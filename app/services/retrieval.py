"""BM25、Qdrant 向量检索与 BGE 重排。"""

import hashlib
import math
import re
import threading
import time
from dataclasses import dataclass

from app.config import Settings
from app.domain import Chunk, Paper


RETRIEVAL_STRATEGIES = ("bm25", "dense", "hybrid", "hybrid-rerank")
INDEX_VERSION = 3


class RetrievalBackendError(RuntimeError):
    """严格评测模式下，检索后端不可用时抛出的错误。"""


@dataclass
class SearchResult:
    """检索到的证据块及其最终得分。"""

    chunk: Chunk
    score: float


class HybridRetriever:
    """使用本地 BGE 编码、Qdrant 持久化索引和可选 CrossEncoder 重排。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedding_model = None
        self.reranker = None
        self.qdrant_client = None
        self.last_error: str | None = None
        self._embedding_lock = threading.RLock()
        self._reranker_lock = threading.RLock()
        self._qdrant_lock = threading.RLock()

    def index(self, paper: Paper) -> bool:
        """上传后一次性编码全文并写入 Qdrant；模型关闭时安全跳过。"""
        index_chunks = self.index_chunks(paper)
        if not self.settings.vector_enabled or not index_chunks:
            return False
        try:
            with self._embedding_lock:
                vectors = self._embedding_model().encode(
                    [self._document_text(chunk) for chunk in index_chunks],
                    normalize_embeddings=True,
                    batch_size=16,
                    show_progress_bar=False,
                )
            client = self._qdrant()
            from qdrant_client.models import Distance, PointStruct, VectorParams

            collection = self._collection_name(paper.id)
            with self._qdrant_lock:
                if client.collection_exists(collection):
                    client.delete_collection(collection)
                client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
                )
            points = [
                PointStruct(
                    id=index,
                    vector=vector.tolist(),
                    payload={"chunk_id": chunk.id, "page": chunk.page, "section": chunk.section},
                )
                for index, (chunk, vector) in enumerate(zip(index_chunks, vectors))
            ]
            with self._qdrant_lock:
                client.upsert(collection_name=collection, points=points, wait=True)
            self.last_error = None
            return True
        except Exception as error:
            self.last_error = f"Qdrant 索引失败：{type(error).__name__}: {error}"
            return False

    def index_chunks(self, paper: Paper) -> list[Chunk]:
        """组合原文本与 Figure/Table 检索表示，统一写入既有 BM25/Dense 基础设施。"""
        return [*paper.chunks, *self.figure_chunks(paper), *self.table_chunks(paper)]

    def figure_chunks(self, paper: Paper) -> list[Chunk]:
        chunks = []
        for item in paper.figures:
            if item.kind != "picture":
                continue
            description = item.description or {}
            lines = [
                f"Caption: {item.caption}",
                f"Summary: {description.get('summary', '')}",
                f"Entities: {self._flatten(description.get('entities', []))}",
                f"Components: {self._flatten(description.get('components', []))}",
                f"Relations: {self._flatten(description.get('relations', []))}",
                f"Visual Evidence: {self._flatten(description.get('visual_evidence', []))}",
                f"Keywords: {self._flatten(description.get('retrieval_keywords', []))}",
            ]
            chunks.append(Chunk(
                item.id, paper.id, item.page, item.section or "Figure",
                "\n".join(lines), item.bbox,
            ))
        return chunks

    def table_chunks(self, paper: Paper) -> list[Chunk]:
        return [
            Chunk(
                item.id, paper.id, item.page, item.section or f"Table: {item.caption}",
                f"Caption: {item.caption}\n{item.content}", item.bbox,
            )
            for item in paper.figures if item.kind == "table" and item.content.strip()
        ]

    def _flatten(self, value) -> str:
        if not isinstance(value, list):
            return str(value or "")
        return "; ".join(
            str(item) if not isinstance(item, dict)
            else ", ".join(f"{key}: {part}" for key, part in item.items())
            for item in value
        )

    def search(
        self,
        chunks: list[Chunk],
        query: str,
        limit: int = 6,
        section_hints: tuple[str, ...] = (),
        fallback_section_hints: tuple[str, ...] = (),
        lexical_query: str | None = None,
        strategy: str = "hybrid-rerank",
        strict: bool = False,
    ) -> list[SearchResult]:
        """按指定策略检索；严格模式禁止模型或 Qdrant 故障被静默降级。"""
        results, _trace = self.search_with_trace(
            chunks,
            query,
            limit=limit,
            section_hints=section_hints,
            fallback_section_hints=fallback_section_hints,
            lexical_query=lexical_query,
            strategy=strategy,
            strict=strict,
        )
        return results

    def search_with_trace(
        self,
        chunks: list[Chunk],
        query: str,
        limit: int = 6,
        section_hints: tuple[str, ...] = (),
        fallback_section_hints: tuple[str, ...] = (),
        lexical_query: str | None = None,
        strategy: str = "hybrid-rerank",
        strict: bool = False,
    ) -> tuple[list[SearchResult], dict]:
        """检索并返回本次请求真实使用的后端、候选数量和降级原因。"""
        if strategy not in RETRIEVAL_STRATEGIES:
            raise ValueError(f"未知检索策略：{strategy}")
        needs_dense = strategy in ("dense", "hybrid", "hybrid-rerank")
        needs_reranker = strategy == "hybrid-rerank"
        if strict and needs_dense and not self.settings.vector_enabled:
            raise RetrievalBackendError(f"策略 {strategy} 需要启用向量检索")
        if strict and needs_reranker and not self.settings.reranker_enabled:
            raise RetrievalBackendError("策略 hybrid-rerank 需要启用 reranker")

        warnings: list[str] = []
        if needs_dense and not self.settings.vector_enabled:
            warnings.append("向量检索未启用")
        if needs_reranker and not self.settings.reranker_enabled:
            warnings.append("重排模型未启用")
        scoped_chunks = self._scope_sections(chunks, section_hints, fallback_section_hints)
        lexical_started = time.perf_counter()
        lexical = (
            self._bm25(scoped_chunks, lexical_query or query)
            if strategy in ("bm25", "hybrid", "hybrid-rerank")
            else []
        )
        lexical_latency_ms = round((time.perf_counter() - lexical_started) * 1000, 1)
        semantic_started = time.perf_counter()
        semantic = (
            self._semantic_with_error(scoped_chunks, query, strict, warnings)
            if needs_dense
            else []
        )
        semantic_latency_ms = round((time.perf_counter() - semantic_started) * 1000, 1)
        if strategy == "bm25":
            ranked = lexical
        elif strategy == "dense":
            ranked = semantic
        else:
            ranked = self._rrf(lexical, semantic)
        if section_hints:
            for result in ranked:
                if any(hint in result.chunk.section.lower() for hint in section_hints):
                    result.score += 0.05
            ranked.sort(key=lambda result: result.score, reverse=True)
        fusion_candidates = len(ranked)
        reranked = False
        rerank_candidate_count = 0
        rerank_latency_ms = 0.0
        if needs_reranker and self.settings.reranker_enabled:
            self.last_error = None
            rerank_candidates = ranked[: self.settings.reranker_candidate_limit]
            rerank_candidate_count = len(rerank_candidates)
            rerank_started = time.perf_counter()
            ranked = self._rerank(query, rerank_candidates, strict=strict)
            rerank_latency_ms = round((time.perf_counter() - rerank_started) * 1000, 1)
            if self.last_error:
                warnings.append(self.last_error)
            else:
                reranked = bool(ranked)
        output = ranked[:limit]
        if warnings:
            self.last_error = warnings[-1]
        effective_strategy = self._effective_strategy(
            bool(lexical), bool(semantic), reranked
        )
        trace = {
            "requested_strategy": strategy,
            "effective_strategy": effective_strategy,
            "degraded": effective_strategy != strategy,
            "warnings": warnings,
            "index_version": INDEX_VERSION,
            "scoped_chunk_count": len(scoped_chunks),
            "lexical_candidate_count": len(lexical),
            "semantic_candidate_count": len(semantic),
            "fusion_candidate_count": fusion_candidates,
            "rerank_candidate_count": rerank_candidate_count,
            "reranked": reranked,
            "returned_count": len(output),
            "bm25_latency_ms": lexical_latency_ms,
            "dense_latency_ms": semantic_latency_ms,
            "rerank_latency_ms": rerank_latency_ms,
        }
        return output, trace

    def _semantic_with_error(
        self,
        chunks: list[Chunk],
        query: str,
        strict: bool,
        warnings: list[str],
    ) -> list[SearchResult]:
        """执行语义检索，并将非严格模式下的失败写入本次检索轨迹。"""
        self.last_error = None
        results = self._semantic(chunks, query, strict=strict)
        if self.last_error:
            warnings.append(self.last_error)
        return results

    def _effective_strategy(
        self,
        has_lexical: bool,
        has_semantic: bool,
        reranked: bool,
    ) -> str:
        """根据实际成功的组件标记本次检索，而不是只报告请求配置。"""
        if reranked:
            return "hybrid-rerank" if has_lexical and has_semantic else "rerank"
        if has_lexical and has_semantic:
            return "hybrid"
        if has_semantic:
            return "dense"
        if has_lexical:
            return "bm25"
        return "none"

    def _scope_sections(self, chunks: list[Chunk], hints: tuple[str, ...], fallback_hints: tuple[str, ...] = ()) -> list[Chunk]:
        """按章节标题起始词硬过滤；主章节缺失时才使用回退章节。"""
        if not hints:
            return chunks
        scoped = [chunk for chunk in chunks if self._section_matches(chunk.section, hints)]
        if scoped:
            return scoped
        fallback = [chunk for chunk in chunks if self._section_matches(chunk.section, fallback_hints)]
        return fallback or chunks

    def _section_matches(self, section: str, hints: tuple[str, ...]) -> bool:
        """只接受章节标题开头匹配，避免 Comparison with Methods 被当作方法章节。"""
        normalized = re.sub(r"^(?:[\d.]+|[ivxlcdm]+)\s+", "", section.strip().lower())
        return any(normalized.startswith(hint.lower()) for hint in hints)

    def status(self) -> dict:
        """返回检索后端状态，便于健康检查和排障。"""
        return {
            "vector_enabled": self.settings.vector_enabled,
            "reranker_enabled": self.settings.reranker_enabled,
            "qdrant_mode": "remote" if self.settings.qdrant_url else "local",
            "qdrant_path": str(self.settings.qdrant_dir),
            "last_error": self.last_error,
        }

    def close(self) -> None:
        """释放本地 Qdrant 文件锁，供服务重载、测试和正常退出使用。"""
        with self._qdrant_lock:
            if self.qdrant_client is not None:
                self.qdrant_client.close()
                self.qdrant_client = None

    def delete_index(self, paper_id: str) -> None:
        """删除指定论文的向量集合；清理失败不能掩盖原始业务错误。"""
        if not self.settings.vector_enabled:
            return
        try:
            client = self._qdrant()
            collection = self._collection_name(paper_id)
            with self._qdrant_lock:
                if client.collection_exists(collection):
                    client.delete_collection(collection)
        except Exception as error:
            self.last_error = f"Qdrant 清理失败：{type(error).__name__}: {error}"

    def _bm25(self, chunks: list[Chunk], query: str) -> list[SearchResult]:
        """计算 BM25，保障术语、缩写和数值的精确召回。"""
        terms = self._tokens(query)
        documents = [self._tokens(self._document_text(chunk)) for chunk in chunks]
        avg_length = sum(map(len, documents)) / max(len(documents), 1)
        document_frequency = {term: sum(term in document for document in documents) for term in terms}
        results: list[SearchResult] = []
        for chunk, document in zip(chunks, documents):
            score = 0.0
            for term in terms:
                frequency = document.count(term)
                if not frequency:
                    continue
                df = document_frequency[term]
                idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
                score += idf * frequency * 2.2 / (
                    frequency + 1.2 * (1 - 0.75 + 0.75 * len(document) / max(avg_length, 1))
                )
            if score > 0:
                results.append(SearchResult(chunk, score))
        return sorted(results, key=lambda result: result.score, reverse=True)

    def _semantic(
        self,
        chunks: list[Chunk],
        query: str,
        strict: bool = False,
    ) -> list[SearchResult]:
        """只编码问题，在 Qdrant 查询已持久化的论文段落向量。"""
        if not self.settings.vector_enabled or not chunks:
            return []
        try:
            paper_id = chunks[0].paper_id
            with self._embedding_lock:
                vector = self._embedding_model().encode(query, normalize_embeddings=True).tolist()
            client = self._qdrant()
            from qdrant_client.models import FieldCondition, Filter, MatchAny

            allowed_ids = [chunk.id for chunk in chunks]
            with self._qdrant_lock:
                response = client.query_points(
                    collection_name=self._collection_name(paper_id),
                    query=vector,
                    limit=min(max(len(chunks), 1), 24),
                    with_payload=True,
                    query_filter=Filter(
                        must=[FieldCondition(key="chunk_id", match=MatchAny(any=allowed_ids))]
                    ),
                )
            chunk_map = {chunk.id: chunk for chunk in chunks}
            results = []
            for point in response.points:
                chunk = chunk_map.get(point.payload.get("chunk_id"))
                if chunk:
                    results.append(SearchResult(chunk, float(point.score)))
            self.last_error = None
            return results
        except Exception as error:
            self.last_error = f"Qdrant 查询失败：{type(error).__name__}: {error}"
            if strict:
                raise RetrievalBackendError(self.last_error) from error
            return []

    def _rrf(self, lexical: list[SearchResult], semantic: list[SearchResult]) -> list[SearchResult]:
        """使用 RRF 融合不同尺度的词法与语义排名。"""
        scores: dict[str, SearchResult] = {}
        result_sets = (lexical, semantic) if semantic else (lexical,)
        for results in result_sets:
            for rank, result in enumerate(results, start=1):
                existing = scores.setdefault(result.chunk.id, SearchResult(result.chunk, 0.0))
                existing.score += 1 / (60 + rank)
        return sorted(scores.values(), key=lambda result: result.score, reverse=True)

    def _rerank(
        self,
        query: str,
        results: list[SearchResult],
        strict: bool = False,
    ) -> list[SearchResult]:
        """使用本地 bge-reranker-v2-m3 对候选段落精排。"""
        if not self.settings.reranker_enabled or not results:
            return results
        try:
            with self._reranker_lock:
                if self.reranker is None:
                    from sentence_transformers import CrossEncoder
                    self.reranker = CrossEncoder(
                        str(self._model_path("bge-reranker-v2-m3")),
                        max_length=self.settings.reranker_max_length,
                    )
                scores = self.reranker.predict([
                    (query, self._document_text(item.chunk)) for item in results
                ])
            self.last_error = None
            return sorted(
                [SearchResult(item.chunk, float(score)) for item, score in zip(results, scores)],
                key=lambda item: item.score,
                reverse=True,
            )
        except Exception as error:
            self.last_error = f"重排失败，已保留融合排序：{type(error).__name__}: {error}"
            if strict:
                raise RetrievalBackendError(self.last_error) from error
            return results

    def _embedding_model(self):
        """按需加载本地 bge-m3，避免服务启动时占用大量内存。"""
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(str(self._model_path("bge-m3")))
        return self.embedding_model

    def _qdrant(self):
        """连接远程 Qdrant，或创建 demo/data/qdrant 下的本地持久化实例。"""
        with self._qdrant_lock:
            if self.qdrant_client is None:
                from qdrant_client import QdrantClient
                self.qdrant_client = (
                    QdrantClient(url=self.settings.qdrant_url)
                    if self.settings.qdrant_url
                    else QdrantClient(path=str(self.settings.qdrant_dir))
                )
        return self.qdrant_client

    def _collection_name(self, paper_id: str) -> str:
        """将 UUID 映射为符合 Qdrant 命名规则的稳定集合名。"""
        digest = hashlib.sha1(paper_id.encode("utf-8")).hexdigest()[:20]
        return f"paper_{digest}"

    def _tokens(self, text: str) -> list[str]:
        """切分英文术语、数字和中文二元词，避免整句中文被视作一个词。"""
        normalized = text.lower()
        english = re.findall(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*|\d+(?:\.\d+)?%?", normalized)
        chinese_sequences = re.findall(r"[\u4e00-\u9fff]+", normalized)
        chinese: list[str] = []
        for sequence in chinese_sequences:
            if len(sequence) == 1:
                chinese.append(sequence)
            else:
                chinese.extend(sequence[index:index + 2] for index in range(len(sequence) - 1))
        return english + chinese

    def _document_text(self, chunk: Chunk) -> str:
        """把章节标题和断行修复版本加入检索表示，同时保留可引用原文。"""
        original = re.sub(r"\s+", " ", chunk.text).strip()
        dehyphenated = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "", original)
        parts = [chunk.section.strip(), original]
        if dehyphenated != original:
            parts.append(dehyphenated)
        return "\n".join(part for part in parts if part)

    def _model_path(self, name: str):
        """优先读取 ModelScope 本地快照，缺失时返回官方模型名。"""
        path = self.settings.models_dir / "modelscope" / name
        return path if (path / "config.json").exists() else f"BAAI/{name}"
