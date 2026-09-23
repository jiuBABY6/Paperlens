"""结论、证据与原文一致性的回归测试。"""

from app.config import settings
from app.domain import Chunk, Sentence
from app.services.reading import ReadingService
from app.services.retrieval import HybridRetriever, SearchResult


def test_sentence_id_must_belong_to_the_retrieved_chunk() -> None:
    """模型不能选择 Top-K 段落之外的句子作为证据。"""
    chunk = Chunk("e1", "paper", 1, "Limitations", "Inference time is approximately 8x higher than the baseline.", (0, 0, 1, 1), sentence_ids=["s1"])
    sentence = Sentence("s1", "paper", "e1", 1, "Limitations", "Inference time is approximately 8x higher than the baseline.", (0, 0, 1, 1))
    service = ReadingService(settings, HybridRetriever(settings))
    supported = service._safe_item({"text": "推理开销较高。", "sentence_ids": ["s1"]}, [SearchResult(chunk, 1)], [sentence])
    unsupported = service._safe_item({"text": "存在四项局限。", "sentence_ids": ["s2"]}, [SearchResult(chunk, 1)], [sentence])
    assert supported["citations"]
    assert unsupported["citations"] == []
    assert "证据校验未通过" in unsupported["text"]


def test_section_hints_are_hard_filters_when_the_section_exists() -> None:
    """方法字段不能因为关键词相似而召回实验章节。"""
    chunks = [
        Chunk("method", "paper", 1, "3 Method", "model architecture", None),
        Chunk("result", "paper", 2, "4 Experiments", "model achieves high score", None),
    ]
    retriever = HybridRetriever(settings)
    assert retriever._scope_sections(chunks, ("method",)) == [chunks[0]]


def test_comparison_with_methods_is_not_a_method_section() -> None:
    """标题中包含 Methods 的比较章节不能污染方法概览。"""
    chunks = [
        Chunk("method", "paper", 1, "3 Method", "model architecture", None),
        Chunk("comparison", "paper", 9, "H Comparison with Training-based Methods", "comparison text", None),
    ]
    retriever = HybridRetriever(settings)
    assert retriever._scope_sections(chunks, ("method",)) == [chunks[0]]


def test_question_answer_rejects_sentence_outside_retrieved_chunks(monkeypatch) -> None:
    """自由问答和精读卡片使用相同的 Top-K 证据约束。"""
    from dataclasses import replace

    local_settings = replace(settings, deepseek_key="test")
    service = ReadingService(local_settings, HybridRetriever(local_settings))
    chunk = Chunk("chunk", "paper", 1, "Method", "Supported sentence.", None, sentence_ids=["allowed"])
    sentences = [
        Sentence("allowed", "paper", "chunk", 1, "Method", "Supported sentence.", (0, 0, 1, 1)),
        Sentence("outside", "paper", "other", 2, "Results", "Outside sentence.", (0, 0, 1, 1)),
    ]
    monkeypatch.setattr(
        service,
        "_request",
        lambda *_args, **_kwargs: '{"answerable":true,"claims":[{"text":"Unsupported answer","sentence_ids":["outside"]}]}',
    )

    result = service.answer_with_evidence("question", [chunk], sentences)

    assert result["citations"] == []
    assert result["answerable"] is False
    assert result["status"] == "invalid_claim_evidence"


def test_question_answer_returns_sentence_level_coordinates(monkeypatch) -> None:
    from dataclasses import replace

    local_settings = replace(settings, deepseek_key="test")
    service = ReadingService(local_settings, HybridRetriever(local_settings))
    chunk = Chunk("chunk", "paper", 1, "Method", "Supported sentence.", None, sentence_ids=["allowed"])
    sentence = Sentence(
        "allowed", "paper", "chunk", 1, "Method", "Supported sentence.",
        (10, 20, 30, 40), [(10, 20, 30, 25), (10, 30, 20, 40)],
    )
    responses = iter([
        '{"answerable":true,"claims":[{"text":"有原文支持的回答。","sentence_ids":["allowed"]}]}',
        '{"verdicts":[{"claim_index":0,"supported":true,"sentence_ids":["allowed"],"reason":"direct"}]}',
    ])
    monkeypatch.setattr(service, "_request", lambda *_args, **_kwargs: next(responses))

    result = service.answer_with_evidence("question", [chunk], [sentence])

    assert result["answer"] == "有原文支持的回答。"
    assert result["answerable"] is True
    assert result["status"] == "ok"
    assert result["citations"][0]["id"] == "allowed"
    assert len(result["citations"][0]["bboxes"]) == 2
    assert result["claims"][0]["verification_status"] == "supported"
    assert result["claims"][0]["verification_reason"] == "direct"


def test_simple_answer_limits_claims_and_citations_to_direct_evidence() -> None:
    service = ReadingService(settings, HybridRetriever(settings))
    sentence_map = {
        f"s{index}": Sentence(
            f"s{index}", "paper", "chunk", 1, "Abstract", f"Fact {index}.", None
        )
        for index in range(1, 6)
    }
    raw_claims = [
        {
            "text": f"Claim {index}.",
            "sentence_ids": [f"s{index}", f"s{index + 1}"],
        }
        for index in range(1, 5)
    ]

    claims = service._candidate_claims(
        raw_claims,
        set(sentence_map),
        sentence_map,
    )

    assert len(claims) == 3
    assert all(len(item["sentence_ids"]) <= 2 for item in claims)


def test_question_answer_returns_structured_refusal(monkeypatch) -> None:
    """证据不足必须是机器可判定的拒答，且不能携带引用。"""
    from dataclasses import replace

    local_settings = replace(settings, deepseek_key="test")
    service = ReadingService(local_settings, HybridRetriever(local_settings))
    chunk = Chunk("chunk", "paper", 1, "Method", "Related but insufficient.", None, sentence_ids=["s1"])
    sentence = Sentence("s1", "paper", "chunk", 1, "Method", "Related but insufficient.", None)
    monkeypatch.setattr(
        service,
        "_request",
        lambda *_args, **_kwargs: '{"answerable":false,"claims":[],"refusal_reason":"论文没有报告该数据。"}',
    )

    result = service.answer_with_evidence("线上每天处理多少图片？", [chunk], [sentence])

    assert result["answerable"] is False
    assert result["citations"] == []
    assert result["status"] == "insufficient_evidence"


def test_claim_rejected_when_verifier_finds_only_partial_support(monkeypatch) -> None:
    """合法 sentence_id 不能绕过语义支持校验。"""
    from dataclasses import replace

    local_settings = replace(settings, deepseek_key="test")
    service = ReadingService(local_settings, HybridRetriever(local_settings))
    chunk = Chunk("chunk", "paper", 1, "Datasets", "The study uses six datasets.", None, sentence_ids=["s1"])
    sentence = Sentence("s1", "paper", "chunk", 1, "Datasets", "The study uses six datasets.", None)
    responses = iter([
        '{"answerable":true,"claims":[{"text":"数据集包括 A、B、C。","sentence_ids":["s1"]}]}',
        '{"verdicts":[{"claim_index":0,"supported":false,"sentence_ids":[],"reason":"证据没有列出名称"}]}',
    ])
    monkeypatch.setattr(service, "_request", lambda *_args, **_kwargs: next(responses))

    result = service.answer_with_evidence("使用哪些数据集？", [chunk], [sentence])

    assert result["answerable"] is False
    assert result["citations"] == []
    assert result["status"] == "unsupported_claims"


def test_exact_per_unit_question_rejects_total_only_answer() -> None:
    payload = {
        "answer": "The paper reports a total cost for 100 samples, not a per-explanation cost.",
        "claims": [{"claim": "Only total cost is reported."}],
    }
    assert ReadingService._reported_information_is_absent(
        "What was the exact US-dollar API cost per generated explanation?",
        payload,
        [],
    ) is True
