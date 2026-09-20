"""跨语言查询规划和词法检索的回归测试。"""

from dataclasses import replace

from app.config import settings
from app.domain import Chunk
from app.services.reading import ReadingService
from app.services.retrieval import HybridRetriever


def test_chinese_tokenizer_uses_bigrams_instead_of_whole_sentence() -> None:
    retriever = HybridRetriever(settings)
    tokens = retriever._tokens("作者使用了哪些数据集？")
    assert "数据" in tokens
    assert "据集" in tokens
    assert "作者使用了哪些数据集" not in tokens


def test_english_lexical_query_can_recall_english_paper_for_chinese_question() -> None:
    retriever = HybridRetriever(settings)
    chunks = [
        Chunk("dataset", "paper", 1, "Experiments", "Experiments use the CIFAR-10 dataset.", None),
        Chunk("method", "paper", 2, "Method", "The encoder contains three attention layers.", None),
    ]
    results = retriever.search(
        chunks,
        "作者使用了什么数据集？",
        lexical_query="dataset used in experiments CIFAR-10",
        limit=1,
        strategy="bm25",
    )
    assert results[0].chunk.id == "dataset"


def test_query_plan_preserves_semantic_query_and_uses_english_rewrite() -> None:
    service = ReadingService(replace(settings, deepseek_key="test"), HybridRetriever(settings))
    service._request = lambda *_args, **_kwargs: '{"lexical_query":"datasets used in experiments BERT 12.5%"}'

    plan = service.plan_query("BERT 在哪些数据集上提升了 12.5%？")

    assert plan["semantic_query"] == "BERT 在哪些数据集上提升了 12.5%？"
    assert plan["lexical_query"] == "datasets used in experiments BERT 12.5%"
    assert plan["translated"] is True
