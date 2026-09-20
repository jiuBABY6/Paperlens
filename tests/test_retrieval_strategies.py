"""检索策略路由和严格失败行为测试。"""

from dataclasses import replace

import numpy as np
import pytest

from app.config import settings
from app.domain import Chunk
from app.services.retrieval import HybridRetriever, RetrievalBackendError, SearchResult


def build_retriever(monkeypatch):
    retriever = HybridRetriever(
        replace(settings, vector_enabled=True, reranker_enabled=True)
    )
    chunks = [
        Chunk("lexical", "paper", 1, "Method", "lexical", None),
        Chunk("semantic", "paper", 2, "Results", "semantic", None),
    ]
    calls = []

    def bm25(_chunks, _query):
        calls.append("bm25")
        return [SearchResult(chunks[0], 2.0)]

    def semantic(_chunks, _query, strict=False):
        calls.append(("dense", strict))
        return [SearchResult(chunks[1], 0.9)]

    def rerank(_query, results, strict=False):
        calls.append(("rerank", strict))
        return list(reversed(results))

    monkeypatch.setattr(retriever, "_bm25", bm25)
    monkeypatch.setattr(retriever, "_semantic", semantic)
    monkeypatch.setattr(retriever, "_rerank", rerank)
    return retriever, chunks, calls


@pytest.mark.parametrize(
    ("strategy", "expected_calls"),
    [
        ("bm25", ["bm25"]),
        ("dense", [("dense", True)]),
        ("hybrid", ["bm25", ("dense", True)]),
        (
            "hybrid-rerank",
            ["bm25", ("dense", True), ("rerank", True)],
        ),
    ],
)
def test_strategy_runs_only_required_components(
    monkeypatch,
    strategy,
    expected_calls,
) -> None:
    retriever, chunks, calls = build_retriever(monkeypatch)

    results = retriever.search(chunks, "query", strategy=strategy, strict=True)

    assert results
    assert calls == expected_calls


def test_strict_dense_strategy_rejects_disabled_vector_search() -> None:
    retriever = HybridRetriever(
        replace(settings, vector_enabled=False, reranker_enabled=False)
    )
    chunks = [Chunk("chunk", "paper", 1, "Method", "text", None)]

    with pytest.raises(RetrievalBackendError, match="需要启用向量检索"):
        retriever.search(chunks, "query", strategy="dense", strict=True)


def test_strict_rerank_strategy_rejects_disabled_reranker() -> None:
    retriever = HybridRetriever(
        replace(settings, vector_enabled=True, reranker_enabled=False)
    )
    chunks = [Chunk("chunk", "paper", 1, "Method", "text", None)]

    with pytest.raises(RetrievalBackendError, match="需要启用 reranker"):
        retriever.search(chunks, "query", strategy="hybrid-rerank", strict=True)


def test_strict_dense_strategy_surfaces_qdrant_failure(monkeypatch) -> None:
    retriever = HybridRetriever(
        replace(settings, vector_enabled=True, reranker_enabled=False)
    )
    chunks = [Chunk("chunk", "paper", 1, "Method", "text", None)]

    class FakeEmbedding:
        def encode(self, _query, **_kwargs):
            return np.array([1.0, 0.0])

    retriever.embedding_model = FakeEmbedding()

    def unavailable_qdrant():
        raise RuntimeError("storage is already accessed")

    monkeypatch.setattr(retriever, "_qdrant", unavailable_qdrant)

    with pytest.raises(RetrievalBackendError, match="storage is already accessed"):
        retriever.search(chunks, "query", strategy="dense", strict=True)


def test_retrieval_document_includes_section_and_dehyphenated_text() -> None:
    retriever = HybridRetriever(settings)
    chunk = Chunk(
        "chunk", "paper", 9, "Limitations",
        "The increased cost remains an inherent trade- off.", None,
    )

    document = retriever._document_text(chunk)

    assert document.startswith("Limitations\n")
    assert "tradeoff" in document
    assert "trade- off" in document


def test_non_strict_search_reports_actual_degradation() -> None:
    retriever = HybridRetriever(
        replace(settings, vector_enabled=False, reranker_enabled=False)
    )
    chunks = [Chunk("chunk", "paper", 1, "Method", "alpha method", None)]

    results, trace = retriever.search_with_trace(
        chunks, "alpha", strategy="hybrid-rerank", strict=False
    )

    assert results[0].chunk.id == "chunk"
    assert trace["requested_strategy"] == "hybrid-rerank"
    assert trace["effective_strategy"] == "bm25"
    assert trace["degraded"] is True
    assert len(trace["warnings"]) == 2


def test_hybrid_rerank_limits_candidates_and_records_stage_latencies(monkeypatch) -> None:
    local_settings = replace(
        settings,
        vector_enabled=True,
        reranker_enabled=True,
        reranker_candidate_limit=8,
    )
    retriever = HybridRetriever(local_settings)
    chunks = [
        Chunk(f"chunk-{index}", "paper", 1, "Method", f"term {index}", None)
        for index in range(12)
    ]
    ranked = [SearchResult(chunk, float(12 - index)) for index, chunk in enumerate(chunks)]
    seen = {}

    monkeypatch.setattr(retriever, "_bm25", lambda *_args: ranked)
    monkeypatch.setattr(retriever, "_semantic", lambda *_args, **_kwargs: ranked)

    def fake_rerank(_query, candidates, strict=False):
        seen["count"] = len(candidates)
        return candidates

    monkeypatch.setattr(retriever, "_rerank", fake_rerank)
    _results, trace = retriever.search_with_trace(
        chunks, "term", strategy="hybrid-rerank"
    )

    assert seen["count"] == 8
    assert trace["rerank_candidate_count"] == 8
    assert {"bm25_latency_ms", "dense_latency_ms", "rerank_latency_ms"} <= set(trace)
