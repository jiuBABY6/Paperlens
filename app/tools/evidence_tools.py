"""Text/Figure/Table 的检索与原文读取工具。"""

import re

from app.domain import Chunk, EvidenceObject, Paper, sentence_evidence
from app.services.retrieval import HybridRetriever


class EvidenceTools:
    FIGURE_REFERENCE = re.compile(
        r"\bfig(?:ure)?s?\.?\s*"
        r"(\d+(?:\s*(?:,|and|&|to|-)\s*\d+)*)|图\s*(\d+)",
        re.I,
    )
    FIGURE_CAPTION = re.compile(
        r"^\s*(?:fig(?:ure)?\.?|图)\s*(\d+)\b",
        re.I,
    )
    TABLE_REFERENCE = re.compile(
        r"\btables?\.?\s*"
        r"(\d+(?:\s*(?:,|and|&|to|-)\s*\d+)*)|表\s*(\d+)",
        re.I,
    )
    TABLE_CAPTION = re.compile(
        r"^\s*(?:table|表)\s*(\d+)\b",
        re.I,
    )

    def __init__(
        self,
        paper: Paper,
        retriever: HybridRetriever,
        figure_service=None,
        search_strategy: str | None = None,
    ) -> None:
        self.paper = paper
        self.retriever = retriever
        self.figure_service = figure_service
        self.search_strategy = search_strategy
        self.sentences = {item.id: item for item in paper.sentences}
        self.chunks = {item.id: item for item in paper.chunks}
        self.figures = {item.id: item for item in paper.figures}
        self.last_search_trace: dict = {}

    def _search(self, chunks: list[Chunk], query: str, top_k: int):
        """统一执行检索并保留真实后端、候选数和降级信息。"""
        options = {"strategy": self.search_strategy} if self.search_strategy else {}
        results, trace = self.retriever.search_with_trace(
            chunks,
            query,
            limit=top_k,
            **options,
        )
        self.last_search_trace = trace
        return results

    def search_text(self, query: str, top_k: int = 10) -> list[tuple[EvidenceObject, float]]:
        candidates = [item for item in self.paper.chunks if not item.section.startswith("Table:")]
        results = self._search(candidates, query, top_k)
        output: list[tuple[EvidenceObject, float]] = []
        for result in results:
            ids = result.chunk.sentence_ids or [result.chunk.id]
            for evidence_id in ids:
                sentence = self.sentences.get(evidence_id)
                if sentence:
                    output.append((sentence_evidence(sentence), result.score))
                    break
        return output

    def read_text(self, chunk_id: str) -> dict | None:
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return None
        return {
            "chunk_id": chunk.id,
            "text": chunk.text,
            "section": chunk.section,
            "page": chunk.page,
            "bbox": chunk.bbox,
            "sentence_ids": chunk.sentence_ids,
        }

    def read_sentence(self, sentence_id: str) -> dict | None:
        sentence = self.sentences.get(sentence_id)
        return sentence_evidence(sentence).to_payload() if sentence else None

    def read_section(self, section_name: str) -> list[dict]:
        name = section_name.strip().lower()
        return [
            self.read_text(item.id) for item in self.paper.chunks
            if name in item.section.lower()
        ]

    def search_figures(self, query: str, top_k: int = 5) -> list[tuple[EvidenceObject, float]]:
        requested_numbers = self._requested_figure_numbers(query)
        exact_items = [
            item
            for item in self.paper.figures
            if item.kind == "picture"
            and self._caption_figure_number(item.caption) in requested_numbers
        ]
        exact_ids = {item.id for item in exact_items}
        # 显式 Figure 编号是结构化约束。只有论文中确实存在对应图注时才过滤；
        # 否则保留语义检索，兼容图注缺失或解析失败的旧论文。
        exact_match_applied = bool(requested_numbers and exact_ids)
        if exact_match_applied:
            selected = exact_items[:top_k]
            self.last_search_trace = self._exact_search_trace(
                "figure", requested_numbers, [item.id for item in selected]
            )
            return [(item.as_evidence(), 1.0) for item in selected]
        chunks = self.retriever.figure_chunks(self.paper)
        results = self._search(chunks, query, top_k)
        self.last_search_trace = {
            **self.last_search_trace,
            "requested_figure_numbers": sorted(requested_numbers),
            "exact_figure_match_applied": exact_match_applied,
            "exact_figure_ids": sorted(exact_ids),
        }
        return [
            (self.figures[item.chunk.id].as_evidence(), item.score)
            for item in results if item.chunk.id in self.figures
        ]

    @classmethod
    def _requested_figure_numbers(cls, query: str) -> set[int]:
        numbers: set[int] = set()
        for english, chinese in cls.FIGURE_REFERENCE.findall(query or ""):
            numbers.update(int(item) for item in re.findall(r"\d+", english or chinese))
        return numbers

    @classmethod
    def _caption_figure_number(cls, caption: str) -> int | None:
        match = cls.FIGURE_CAPTION.search(caption or "")
        return int(match.group(1)) if match else None

    def _exact_search_trace(
        self,
        modality: str,
        requested_numbers: set[int],
        exact_ids: list[str],
    ) -> dict:
        """记录零模型调用的结构化编号命中。"""
        return {
            "requested_strategy": self.search_strategy or "default",
            "effective_strategy": "metadata-exact",
            "degraded": False,
            "warnings": [],
            "scoped_chunk_count": len(exact_ids),
            "lexical_candidate_count": 0,
            "semantic_candidate_count": 0,
            "fusion_candidate_count": 0,
            "rerank_candidate_count": 0,
            "reranked": False,
            "returned_count": len(exact_ids),
            "bm25_latency_ms": 0.0,
            "dense_latency_ms": 0.0,
            "rerank_latency_ms": 0.0,
            f"requested_{modality}_numbers": sorted(requested_numbers),
            f"exact_{modality}_match_applied": True,
            f"exact_{modality}_ids": sorted(exact_ids),
        }

    def read_figure(self, figure_id: str) -> dict | None:
        item = self.figures.get(figure_id)
        if not item or item.kind != "picture":
            return None
        return {
            **item.as_evidence().to_payload(),
            "image_path": item.image_path,
            "caption": item.caption,
            "nearby_text": item.nearby_text,
            "offline_description": item.description,
        }

    def analyze_figure_for_query(self, figure_id: str, question: str) -> dict | None:
        item = self.figures.get(figure_id)
        if not item or item.kind != "picture" or not self.figure_service:
            return None
        return self.figure_service.analyze_for_query(self.paper.id, item, question)

    def search_tables(self, query: str, top_k: int = 5) -> list[tuple[EvidenceObject, float]]:
        requested_numbers = self._requested_table_numbers(query)
        exact_items = [
            item
            for item in self.paper.figures
            if item.kind == "table"
            and self._caption_table_number(item.caption) in requested_numbers
        ]
        if requested_numbers and exact_items:
            selected = exact_items[:top_k]
            self.last_search_trace = self._exact_search_trace(
                "table", requested_numbers, [item.id for item in selected]
            )
            return [(item.as_evidence(), 1.0) for item in selected]
        chunks = self.retriever.table_chunks(self.paper)
        results = self._search(chunks, query, top_k)
        self.last_search_trace = {
            **self.last_search_trace,
            "requested_table_numbers": sorted(requested_numbers),
            "exact_table_match_applied": False,
            "exact_table_ids": [],
        }
        return [
            (self.figures[item.chunk.id].as_evidence(), item.score)
            for item in results if item.chunk.id in self.figures
        ]

    @classmethod
    def _requested_table_numbers(cls, query: str) -> set[int]:
        numbers: set[int] = set()
        for english, chinese in cls.TABLE_REFERENCE.findall(query or ""):
            numbers.update(int(item) for item in re.findall(r"\d+", english or chinese))
        return numbers

    @classmethod
    def _caption_table_number(cls, caption: str) -> int | None:
        match = cls.TABLE_CAPTION.search(caption or "")
        return int(match.group(1)) if match else None

    def read_table(self, table_id: str) -> dict | None:
        item = self.figures.get(table_id)
        if not item or item.kind != "table":
            return None
        return {
            **item.as_evidence().to_payload(),
            "caption": item.caption,
            "markdown": item.content,
            "columns": item.as_evidence().metadata.get("columns", []),
            "rows": item.as_evidence().metadata.get("rows", []),
            "nearby_text": item.nearby_text,
        }

    def analyze_table_image_with_vlm(self, table_id: str, question: str) -> dict:
        """结构化 Table 不可读时，按显式开关读取表格截图。"""
        item = self.figures.get(table_id)
        service = self.figure_service
        if not item or item.kind != "table":
            return {"available": False, "reason": "Table 不存在。"}
        if not service or not service.settings.table_vlm_fallback_enabled:
            return {"available": False, "reason": "Table VLM fallback 未启用。"}
        if not item.image_path or not service.client.available:
            return {"available": False, "reason": "Table 图片或 Qwen-VL 不可用。"}
        prompt = f"""Read the visible scientific table image to answer the question.
Question: {question}
Caption: {item.caption}
Return JSON only: {{"answerable":true,"table_observations":[],"missing_information":[],"confidence":"high|medium|low"}}.
Preserve row/column labels, values, units and comparison direction. Do not infer unreadable cells."""
        service.query_call_count += 1
        result = service.client.analyze(item.image_path, prompt)
        return {
            "available": True,
            "answerable": result.get("answerable") is True,
            "table_observations": result.get("table_observations", []),
            "missing_information": result.get("missing_information", []),
            "confidence": result.get("confidence", "low"),
        }

    def get_evidence(self, evidence_id: str) -> EvidenceObject | None:
        if evidence_id in self.sentences:
            return sentence_evidence(self.sentences[evidence_id])
        if evidence_id in self.chunks:
            chunk = self.chunks[evidence_id]
            return EvidenceObject(
                chunk.id, "text", chunk.page, chunk.bbox, chunk.text, chunk.section,
                {"sentence_ids": chunk.sentence_ids, "chunk_level": True},
            )
        item = self.figures.get(evidence_id)
        return item.as_evidence() if item else None
