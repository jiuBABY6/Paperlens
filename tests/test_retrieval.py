"""Qdrant 本地持久化索引的最小回归测试。"""

from dataclasses import replace

import numpy as np
import pytest

from app.config import settings
from app.domain import Chunk, Paper
from app.services.retrieval import HybridRetriever


class FakeEmbedding:
    """避免测试时加载数 GB BGE 权重的确定性向量编码器。"""

    def encode(self, values, **_kwargs):
        """让包含 alpha 的文本靠近 alpha 查询，其余文本靠近 beta 查询。"""
        values = [values] if isinstance(values, str) else values
        vectors = [np.array([1.0, 0.0]) if "alpha" in value.lower() else np.array([0.0, 1.0]) for value in values]
        return vectors[0] if len(vectors) == 1 else np.array(vectors)


def test_qdrant_index_and_search_are_persistent(tmp_path) -> None:
    """索引一次后，语义查询应从本地 Qdrant 取回正确证据。"""
    pytest.importorskip("qdrant_client")
    local_settings = replace(settings, data_dir=tmp_path, vector_enabled=True, reranker_enabled=False, qdrant_url="")
    chunks = [
        Chunk("chunk-alpha", "paper-1", 1, "Method", "alpha mechanism", (0, 0, 10, 10)),
        Chunk("chunk-beta", "paper-1", 2, "Results", "beta baseline", (0, 0, 10, 10)),
    ]
    paper = Paper("paper-1", "demo.pdf", "Demo", "", "demo.pdf", "test", chunks, [])
    writer = HybridRetriever(local_settings)
    writer.embedding_model = FakeEmbedding()
    assert writer.index(paper)
    writer.close()
    reader = HybridRetriever(local_settings)
    reader.embedding_model = FakeEmbedding()
    results = reader.search(chunks, "alpha", limit=1)
    assert results[0].chunk.id == "chunk-alpha"
    reader.close()
