from dataclasses import replace
from io import BytesIO
from pathlib import Path

from PIL import Image

from app.agent.executor import AgentExecutor
from app.agent.router import QueryRouter
from app.config import settings
from app.domain import Chunk, EvidenceObject, Figure, Paper, Sentence
from app.agent.planner import Planner
from app.services.reading import ReadingService
from app.evaluation import evaluate_records
from app.multimodal.figure_understanding import FigureUnderstandingService
from app.multimodal.qwen_vl_client import QwenVLClient
from app.services.retrieval import HybridRetriever
from app.tools import EvidenceTools


def sample_paper() -> Paper:
    chunk = Chunk("chunk-1", "paper", 2, "Method", "The architecture uses encoder and decoder modules.", (1, 2, 3, 4), sentence_ids=["sent_1"])
    sentence = Sentence("sent_1", "paper", "chunk-1", 2, "Method", chunk.text, chunk.bbox)
    figure = Figure(
        "paper-fig_003", "paper", 3, "picture", "Figure 3: Model architecture encoder decoder",
        (10, 20, 300, 400), "figure.png", "", "Method", chunk.text, ["sent_1"],
        {"status": "ok", "summary": "Encoder and decoder architecture", "components": ["encoder", "decoder"]},
    )
    table = Figure(
        "paper-table_002", "paper", 5, "table", "Table 2: Main results",
        (10, 20, 300, 400), "table.png", "| Method | Score |\n|---|---|\n| Baseline | 80 |\n| Proposed | 85 |",
        "Experiments", "The proposed method improves results.", ["sent_1"],
    )
    return Paper("paper", "paper.pdf", "Title", "Abstract", "paper.pdf", "test", [chunk], [figure, table], [sentence])


def multi_figure_paper() -> Paper:
    paper = sample_paper()
    paper.figures.extend([
        Figure(
            "paper-fig_001", "paper", 1, "picture", "Figure 1: Overview",
            None, "figure-1.png", "", "Method", "", [],
            {"status": "ok", "summary": "Overview"},
        ),
        Figure(
            "paper-fig_004", "paper", 4, "picture", "Figure 4: Results",
            None, "figure-4.png", "", "Results", "", [],
            {"status": "ok", "summary": "Results"},
        ),
    ])
    return paper


def test_router_keeps_simple_text_query_on_standard_rag() -> None:
    router = QueryRouter()
    assert router.route("What dataset does this paper use?")["route"] == "standard_rag"
    multimodal = router.route("Based on Figure 3 and Table 2, which module contributes most?")
    assert multimodal["route"] == "agentic_rag"
    assert multimodal["modalities"] == ["text", "figure", "table"]


def test_router_skips_text_for_pure_visual_question() -> None:
    routed = QueryRouter().route(
        "在图 2 中，Implications 和 Candidate Explanations 分别是什么颜色？"
    )

    assert routed["route"] == "agentic_rag"
    assert routed["modalities"] == ["figure"]
    assert routed["required_modalities"] == ["figure"]
    assert routed["pure_visual"] is True


def test_router_does_not_treat_chinese_word_image_as_figure_reference() -> None:
    routed = QueryRouter().route(
        "根据论文摘要，SimCLIP 使用哪种编码器来建模图像与文本交互？"
    )

    assert routed["modalities"] == ["text"]
    assert routed["required_modalities"] == ["text"]
    assert routed["pure_visual"] is False


def test_router_recognizes_plural_tables_and_explicit_cross_modal_requirements() -> None:
    question = "Based on Figure 1, the method description, and Tables 1 and 2, which design choices matter most?"
    routed = QueryRouter().route(question)
    assert routed["modalities"] == ["text", "figure", "table"]
    assert routed["required_modalities"] == ["text", "figure", "table"]


def test_router_skips_text_for_pure_table_question() -> None:
    routed = QueryRouter().route(
        "According to Table 2, how does LSTM compare with CNN?"
    )

    assert routed["route"] == "agentic_rag"
    assert routed["modalities"] == ["table"]
    assert routed["required_modalities"] == ["table"]
    assert routed["pure_table"] is True


def test_planner_decomposes_methodological_claim_support_question() -> None:
    question = "Are the paper's main methodological claims supported by its experimental results?"
    routed = QueryRouter().route(question)
    assert routed["modalities"] == ["text", "table"]
    assert routed["required_modalities"] == ["text", "table"]
    plan = Planner().plan(question, routed["modalities"])
    assert len(plan["sub_tasks"]) == 3
    descriptions = " ".join(item["description"] for item in plan["sub_tasks"])
    assert "methodological claims" in descriptions
    assert "experimental results" in descriptions
    assert "ablation" in descriptions
    assert any("table" in item["preferred_modalities"] for item in plan["sub_tasks"])


def test_planner_rewrite_preserves_explicit_figure_and_table_numbers() -> None:
    planner = Planner()
    figure_question = "In Figure 1, how many gray LSTM blocks are explicitly drawn?"
    figure_task = planner.plan(figure_question, ["figure"])["sub_tasks"][0]
    figure_query = planner.rewrite(figure_task["description"], figure_question)

    assert "Figure 1" in figure_query
    assert EvidenceTools._requested_figure_numbers(figure_query) == {1}

    table_question = "According to Tables 1 and 2, which system performs best?"
    table_task = planner.plan(table_question, ["table"])["sub_tasks"][0]
    table_query = planner.rewrite(table_task["description"], table_question)

    assert "Tables 1 and 2" in table_query


def test_figure_and_table_tools_return_unified_evidence() -> None:
    local_settings = replace(settings, vector_enabled=False, reranker_enabled=False)
    tools = EvidenceTools(sample_paper(), HybridRetriever(local_settings))
    figure = tools.search_figures("Figure 3 architecture", top_k=3)[0][0]
    table = tools.search_tables("Table 2 Proposed Score", top_k=3)[0][0]
    assert figure.type == "figure" and figure.evidence_id == "paper-fig_003"
    assert table.type == "table" and table.metadata["columns"] == ["Method", "Score"]
    assert tools.last_search_trace["effective_strategy"] == "metadata-exact"
    assert tools.last_search_trace["requested_table_numbers"] == [2]
    assert tools.last_search_trace["exact_table_match_applied"] is True
    assert tools.read_table(table.evidence_id)["rows"][1]["Score"] == "85"


def test_explicit_figure_number_filters_semantic_candidates() -> None:
    local_settings = replace(settings, vector_enabled=False, reranker_enabled=False)
    tools = EvidenceTools(multi_figure_paper(), HybridRetriever(local_settings))

    results = tools.search_figures("What colors are used in Figure 3?", top_k=5)

    assert [item.evidence_id for item, _score in results] == ["paper-fig_003"]
    assert tools.last_search_trace["effective_strategy"] == "metadata-exact"
    assert tools.last_search_trace["reranked"] is False
    assert tools.last_search_trace["semantic_candidate_count"] == 0
    assert tools.last_search_trace["requested_figure_numbers"] == [3]
    assert tools.last_search_trace["exact_figure_match_applied"] is True


def test_unknown_figure_number_falls_back_to_semantic_search() -> None:
    local_settings = replace(settings, vector_enabled=False, reranker_enabled=False)
    tools = EvidenceTools(multi_figure_paper(), HybridRetriever(local_settings))

    results = tools.search_figures("What is shown in Figure 99?", top_k=5)

    assert results
    assert tools.last_search_trace["requested_figure_numbers"] == [99]
    assert tools.last_search_trace["exact_figure_match_applied"] is False


def test_multiple_explicit_figure_numbers_keep_each_named_figure() -> None:
    local_settings = replace(settings, vector_enabled=False, reranker_enabled=False)
    tools = EvidenceTools(multi_figure_paper(), HybridRetriever(local_settings))

    results = tools.search_figures("Compare Figures 1 and 4", top_k=5)

    assert {item.evidence_id for item, _score in results} == {
        "paper-fig_001", "paper-fig_004"
    }
    assert tools.last_search_trace["requested_figure_numbers"] == [1, 4]


class FakeVLClient:
    available = True

    def __init__(self):
        self.token_usage = 0
        self.prompts = []
        self.last_image_preparation = {}

    def analyze(self, _image_path, prompt):
        self.token_usage += 10
        self.prompts.append(prompt)
        if "independent evaluator" in prompt:
            return {
                "score": 0,
                "reference_supported": True,
                "candidate_supported": False,
                "candidate_matches_reference": False,
                "dataset_issue": False,
                "observed_answer": "Implications are yellow and candidates are green.",
                "reason": "The candidate reverses the colors.",
            }
        if "independent visual evidence verifier" in prompt:
            return {
                "supported": False,
                "reason": "The image contradicts the candidate.",
                "unsupported_details": ["reversed colors"],
            }
        if "User question" in prompt:
            return {
                "relevant": True, "answerable": True,
                "figure_evidence": [{"evidence": "encoder connects to decoder", "source": "figure"}],
                "interpretation": "The two modules are connected.",
                "missing_information": [], "confidence": "high",
            }
        return {"figure_type": "architecture", "summary": "two modules", "components": ["encoder", "decoder"]}


def test_qwen_client_upscales_low_resolution_figure_in_memory(tmp_path: Path) -> None:
    path = tmp_path / "small.png"
    Image.new("RGB", (501, 134), "white").save(path)
    client = QwenVLClient(replace(settings, qwen_vl_key="placeholder"))

    mime, payload = client._prepare_image(path)

    with Image.open(BytesIO(payload)) as prepared:
        assert prepared.size == (2871, 768)
    assert mime == "image/png"
    assert client.last_image_preparation == {
        "original_width": 501,
        "original_height": 134,
        "sent_width": 2871,
        "sent_height": 768,
        "upscaled": True,
        "scale": 5.731,
    }


class FakeReading:
    token_usage = 0

    def answer_from_evidence(self, _question, evidence, required_evidence_types=None):
        citations = [item.to_payload() for item in evidence]
        used = list(dict.fromkeys(item.type for item in evidence))
        required = required_evidence_types or ["text"]
        missing = [item for item in required if item not in used]
        return {
            "answer": "grounded", "answerable": True, "claims": [],
            "citations": citations, "status": "ok",
            "claim_coverage": {"complete": not missing, "missing_evidence_types": missing},
        }


def test_agent_trace_records_query_conditioned_figure_reading() -> None:
    paper = sample_paper()
    local_settings = replace(
        settings, vector_enabled=False, reranker_enabled=False,
        qwen_vl_key="placeholder", agent_figure_read_limit=1,
    )
    service = FigureUnderstandingService(local_settings, client=FakeVLClient())
    executor = AgentExecutor(local_settings, HybridRetriever(local_settings), FakeReading(), service)
    router = QueryRouter().route("What are the major components shown in Figure 3?")
    result = executor.run(paper, "What are the major components shown in Figure 3?", router)
    tools = [item["tool"] for item in result["trace"]["steps"]]
    assert "search_figures" in tools
    assert "analyze_figure_for_query" in tools
    assert result["trace"]["qwen_vl_calls"] == 1
    assert result["trace"]["total_steps"] <= local_settings.agent_max_steps
    search_step = next(item for item in result["trace"]["steps"] if item["tool"] == "search_figures")
    assert search_step["retrieval"]["effective_strategy"] == "metadata-exact"
    assert result["retrieval"]["effective_strategy"] == "metadata-exact"
    assert result["trace"]["evidence_memory"][0]["used"] is True
    assert all(item["tool"] != "search_text" for item in result["trace"]["steps"])


def test_offline_visual_judge_is_independent_and_can_reject_answer() -> None:
    local_settings = replace(settings, qwen_vl_key="placeholder")
    service = FigureUnderstandingService(local_settings, client=FakeVLClient())
    figure = sample_paper().figures[0]

    result = service.judge_answer(
        figure,
        "Which colors are used?",
        "Implications are yellow and candidates are green.",
        "Implications are green and candidates are yellow.",
    )

    assert result["kind"] == "visual"
    assert result["correct"] is False
    assert result["score"] == 0
    assert result["dataset_issue"] is False
    assert service.judge_call_count == 1


def test_offline_visual_judge_excludes_bad_gold_from_accuracy() -> None:
    class GoldConflictVLClient(FakeVLClient):
        def analyze(self, _image_path, prompt):
            self.token_usage += 10
            self.prompts.append(prompt)
            return {
                "score": 0,
                "reference_supported": False,
                "candidate_supported": True,
                "candidate_matches_reference": False,
                "dataset_issue": True,
                "observed_answer": "Implications are orange and candidates are light green.",
                "visual_observations": [],
                "reason": "The reference conflicts with the visible colors.",
            }

    local_settings = replace(settings, qwen_vl_key="placeholder")
    service = FigureUnderstandingService(
        local_settings, client=GoldConflictVLClient()
    )

    result = service.judge_answer(
        sample_paper().figures[0],
        "Which colors are used?",
        "Implications are yellow and candidates are green.",
        "Implications are orange and candidates are light green.",
    )

    assert result["dataset_issue"] is True
    assert result["reference_supported"] is False
    assert result["candidate_supported"] is True
    assert result["score"] is None
    assert result["correct"] is None


def test_optional_online_visual_check_is_disabled_by_default_and_enforced_when_enabled() -> None:
    paper = sample_paper()
    question = "What are the major components shown in Figure 3?"
    base = replace(
        settings, vector_enabled=False, reranker_enabled=False,
        qwen_vl_key="placeholder", agent_figure_read_limit=1,
    )

    disabled_service = FigureUnderstandingService(base, client=FakeVLClient())
    disabled = AgentExecutor(
        base, HybridRetriever(base), FakeReading(), disabled_service
    ).run(paper, question, QueryRouter().route(question))
    assert disabled["visual_answer_check"] == {}
    assert disabled["status"] == "ok"

    enabled_settings = replace(base, online_visual_judge_enabled=True)
    enabled_service = FigureUnderstandingService(enabled_settings, client=FakeVLClient())
    enabled = AgentExecutor(
        enabled_settings, HybridRetriever(enabled_settings), FakeReading(), enabled_service
    ).run(paper, question, QueryRouter().route(question))
    assert enabled["visual_answer_check"]["supported"] is False
    assert enabled["status"] == "partial"
    assert enabled["evidence_sufficient"] is False
    assert enabled["trace"]["qwen_vl_calls"] == 2
    assert enabled["visual_verification_status"] == "failed"
    assert "视觉核验未通过。" in enabled["insufficient_evidence"]


def test_frontend_prioritizes_visual_verification_failure_message() -> None:
    html = (Path(__file__).parents[1] / "app" / "static" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "const decision = visualFailed" in html
    assert "? '视觉核验未通过'" in html


def test_required_table_and_cross_modal_flows_use_expected_tools() -> None:
    paper = sample_paper()
    local_settings = replace(
        settings, vector_enabled=False, reranker_enabled=False,
        qwen_vl_key="placeholder", agent_figure_read_limit=1,
    )
    service = FigureUnderstandingService(local_settings, client=FakeVLClient())
    executor = AgentExecutor(local_settings, HybridRetriever(local_settings), FakeReading(), service)
    router = QueryRouter()

    table_question = "According to Table 2, how much does the proposed method outperform the strongest baseline?"
    table_result = executor.run(paper, table_question, router.route(table_question))
    table_tools = [item["tool"] for item in table_result["trace"]["steps"]]
    assert "search_tables" in table_tools and "read_table" in table_tools
    assert "search_text" not in table_tools

    cross_question = "Based on the architecture figure, method description, and Table 2, which modules contribute most?"
    cross_result = executor.run(paper, cross_question, router.route(cross_question))
    cross_tools = [item["tool"] for item in cross_result["trace"]["steps"]]
    assert {"search_text", "search_figures", "analyze_figure_for_query", "read_figure_context", "search_tables", "read_table"} <= set(cross_tools)
    assert cross_result["mode"] == "multimodal_agentic_rag"


def test_reported_information_gap_makes_overall_sufficiency_partial() -> None:
    class GapReading(FakeReading):
        def answer_from_evidence(self, question, evidence, required_evidence_types=None):
            value = super().answer_from_evidence(question, evidence, required_evidence_types)
            value["insufficient_evidence"] = ["No significance test was reported."]
            return value

    paper = sample_paper()
    local_settings = replace(settings, vector_enabled=False, reranker_enabled=False)
    result = AgentExecutor(
        local_settings, HybridRetriever(local_settings), GapReading()
    ).run(
        paper,
        "Are the main methodological claims supported by experimental results?",
        QueryRouter().route("Are the main methodological claims supported by experimental results?"),
    )

    assert result["status"] == "partial"
    assert result["evidence_sufficient"] is False


def test_complex_text_question_stays_text_agentic() -> None:
    question = "Why do the authors claim this method is novel?"
    router = QueryRouter().route(question)
    assert router["modalities"] == ["text"]
    local_settings = replace(settings, vector_enabled=False, reranker_enabled=False)
    result = AgentExecutor(
        local_settings, HybridRetriever(local_settings), FakeReading()
    ).run(sample_paper(), question, router)
    assert result["mode"] == "text_agentic_rag"
    assert all(item["tool"] == "search_text" for item in result["trace"]["steps"])
    assert len(result["plan"]["sub_tasks"]) == 1


def test_final_claim_coverage_reports_missing_required_modalities(monkeypatch) -> None:
    local_settings = replace(settings, deepseek_key="placeholder")
    service = ReadingService(local_settings, HybridRetriever(local_settings))
    evidence = [
        EvidenceObject("sent_1", "text", 1, None, "Method text", "Method"),
        EvidenceObject("fig_1", "figure", 2, None, None, "Method", {"caption": "Architecture"}),
        EvidenceObject("table_1", "table", 3, None, "| A | B |", "Results"),
    ]
    responses = iter([
        '{"answer":"","claims":[{"claim":"The architecture contains module A.","evidence_ids":["fig_1"]}],"insufficient_evidence":[]}',
        '{"verdicts":[{"claim_index":0,"status":"supported","supported_by":["fig_1"],"reason":"visible"}]}',
        '{"claims":[],"insufficient_evidence":[]}',
    ])
    monkeypatch.setattr(service, "_request", lambda *_args, **_kwargs: next(responses))

    result = service.answer_from_evidence(
        "Compare the method description, figure, and table.",
        evidence,
        required_evidence_types=["text", "figure", "table"],
    )

    assert result["status"] == "partial"
    assert result["claim_coverage"]["complete"] is False
    assert result["claim_coverage"]["missing_evidence_types"] == ["text", "table"]
    assert len(result["insufficient_evidence"]) == 2


def test_visual_query_analysis_is_handed_to_generator_and_claim_verifier(monkeypatch) -> None:
    local_settings = replace(settings, deepseek_key="placeholder")
    service = ReadingService(local_settings, HybridRetriever(local_settings))
    evidence = [
        EvidenceObject(
            "fig_1", "figure", 2, None, "Figure 1: LSTM architecture", "Method",
            {
                "caption": "Figure 1: LSTM architecture",
                "query_analysis": {
                    "answerable": True,
                    "visual_observations": [
                        "Three gray LSTM blocks are explicitly drawn before the ellipsis."
                    ],
                    "figure_evidence": [
                        {"evidence": "Three gray LSTM blocks are visible.", "source": "figure"}
                    ],
                    "interpretation": "The figure explicitly shows three gray LSTM blocks.",
                    "confidence": "high",
                    "missing_information": [],
                },
            },
        ),
    ]
    prompts = []
    responses = iter([
        '{"answerable":true,"answer":"Three.","claims":['
        '{"claim":"Figure 1 explicitly shows three gray LSTM blocks before the ellipsis.",'
        '"evidence_ids":["fig_1"]}],"insufficient_evidence":[]}',
        '{"verdicts":[{"claim_index":0,"status":"supported",'
        '"supported_by":["fig_1"],"reason":"The visual observation states three blocks."}]}',
    ])

    def fake_request(prompt, *_args, **_kwargs):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(service, "_request", fake_request)
    result = service.answer_from_evidence(
        "In Figure 1, how many gray LSTM blocks are explicitly drawn before the ellipsis?",
        evidence,
        required_evidence_types=["figure"],
    )

    assert result["answerable"] is True
    assert result["status"] == "ok"
    assert result["claims"][0]["verification_status"] == "supported"
    assert result["citations"][0]["evidence_id"] == "fig_1"
    assert len(prompts) == 2
    assert all("visual_observations" in prompt for prompt in prompts)
    assert all("Three gray LSTM blocks" in prompt for prompt in prompts)


def test_reported_information_absence_returns_structured_refusal(monkeypatch) -> None:
    local_settings = replace(settings, deepseek_key="placeholder")
    service = ReadingService(local_settings, HybridRetriever(local_settings))
    evidence = [
        EvidenceObject(
            "sent_1", "text", 4, None,
            "Table 2 reports accuracy and F1 only.", "Results",
        ),
    ]
    payload = (
        '{"answerable":true,"answer":"论文未报告显著性检验。",'
        '"claims":[{"claim":"论文未报告显著性检验。",'
        '"evidence_ids":["sent_1"]}],'
        '"insufficient_evidence":["No significance test was reported."],'
        '"refusal_reason":""}'
    )
    monkeypatch.setattr(service, "_request", lambda *_args, **_kwargs: payload)

    result = service.answer_from_evidence(
        "论文是否报告了统计显著性检验？", evidence, required_evidence_types=["text"]
    )

    assert result["answerable"] is False
    assert result["status"] == "insufficient_evidence"
    assert "significance test" in result["refusal_reason"]


def test_text_judge_accepts_details_supported_by_actual_citations(monkeypatch) -> None:
    local_settings = replace(settings, deepseek_key="placeholder")
    service = ReadingService(local_settings, HybridRetriever(local_settings))
    captured = {}

    def fake_request(prompt, *_args, **_kwargs):
        captured["prompt"] = prompt
        return '{"score":2,"correct":true,"reason":"supported"}'

    monkeypatch.setattr(service, "_request", fake_request)
    result = service.judge_answer(
        "What is the method principle?",
        "It uses the information bottleneck principle.",
        "It uses the information bottleneck principle and maximizes information gain.",
        ["inspired by the information bottleneck principle"],
        ["the method selects relevant explanations and maximizes information gain"],
    )

    assert result["correct"] is True
    assert "maximizes information gain" in captured["prompt"]
    assert "不得仅因标准答案未提到而扣分" in captured["prompt"]


def test_modality_coverage_repair_adds_only_verified_missing_evidence(monkeypatch) -> None:
    local_settings = replace(settings, deepseek_key="placeholder")
    service = ReadingService(local_settings, HybridRetriever(local_settings))
    evidence = [
        EvidenceObject("sent_1", "text", 1, None, "The method uses a CNN encoder.", "Method"),
        EvidenceObject("fig_1", "figure", 2, None, None, "Method", {"caption": "CNN diagram"}),
        EvidenceObject("table_1", "table", 3, None, "| CNN | 62.9 |", "Results"),
    ]
    responses = iter([
        '{"claims":[{"claim":"The diagram shows a CNN.","evidence_ids":["fig_1"]},{"claim":"CNN reaches 62.9 F1.","evidence_ids":["table_1"]}],"insufficient_evidence":[]}',
        '{"verdicts":[{"claim_index":0,"status":"supported","supported_by":["fig_1"],"reason":"visible"},{"claim_index":1,"status":"supported","supported_by":["table_1"],"reason":"reported"}]}',
        '{"claims":[{"claim":"The method description uses a CNN encoder.","evidence_ids":["sent_1"]}],"insufficient_evidence":[]}',
        '{"verdicts":[{"claim_index":0,"status":"supported","supported_by":["sent_1"],"reason":"stated"}]}',
    ])
    monkeypatch.setattr(service, "_request", lambda *_args, **_kwargs: next(responses))

    result = service.answer_from_evidence(
        "Compare the method description, figure, and table.",
        evidence,
        required_evidence_types=["text", "figure", "table"],
    )

    assert result["status"] == "ok"
    assert result["claim_coverage"]["complete"] is True
    assert result["claim_coverage"]["used_evidence_types"] == ["text", "figure", "table"]
    assert {item["evidence_id"] for item in result["citations"]} == {"sent_1", "fig_1", "table_1"}


def test_answer_language_follows_question_language() -> None:
    service = ReadingService(settings, HybridRetriever(settings))
    assert service._language_instruction("What dataset is used?") == "Write every claim in English."
    assert "中文" in service._language_instruction("使用了什么数据集？")


def test_evaluation_reports_all_metric_groups_and_modes() -> None:
    report = evaluate_records([{
        "mode": "multimodal_agentic_rag",
        "expected_evidence_ids": ["sent_1", "paper-fig_003"],
        "ranked_evidence_ids": ["paper-fig_003", "sent_1"],
        "returned_evidence_ids": ["sent_1", "paper-fig_003"],
        "claims": [{"status": "supported"}],
        "expected_modalities": ["text", "figure"],
        "predicted_modalities": ["text", "figure"],
        "task_success": True, "steps": 3, "tool_calls": ["search_text", "search_figures"],
        "qwen_vl_calls": 1, "latency_ms": 20, "token_usage": 50,
    }])
    assert report["overall"]["retrieval"]["recall_at_k"] == 1.0
    assert report["overall"]["generation"]["claim_support_rate"] == 1.0
    assert report["overall"]["engineering"]["total_qwen_vl_calls"] == 1
    assert set(report["by_mode"]) == {
        "standard_rag", "text_agentic_rag", "multimodal_agentic_rag",
        "text_multi_agent_rag", "multimodal_multi_agent_rag",
    }
