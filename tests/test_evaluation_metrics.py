"""检索排名指标测试。"""

import json
from dataclasses import replace

from app.config import settings
from app.domain import Chunk, Figure, Paper, Sentence
from app.evaluation import evaluate_records
from app.repository import PaperRepository
from app.services.retrieval import SearchResult
import scripts.evaluate as evaluate_module
from scripts.evaluate import (
    alternative_citation_metrics,
    alternative_ranking_metrics,
    ranking_metrics,
    summarize_rag_records,
)


def test_ranking_metrics_reward_early_relevant_chunks() -> None:
    strong = ranking_metrics(["a", "x", "b"], {"a", "b"}, 3)
    weak = ranking_metrics(["x", "a", "b"], {"a", "b"}, 3)

    assert strong["recall"] == 1.0
    assert strong["reciprocal_rank"] == 1.0
    assert strong["ndcg"] > weak["ndcg"]


def test_alternative_evidence_groups_accept_the_best_complete_group() -> None:
    metrics = alternative_ranking_metrics(
        ["alternative", "noise"],
        [{"original"}, {"alternative"}],
        2,
    )

    assert metrics == {"recall": 1.0, "reciprocal_rank": 1.0, "ndcg": 1.0}


def test_alternative_sentence_citations_accept_either_complete_group() -> None:
    metrics = alternative_citation_metrics(
        {"page4-s1", "page4-s2"},
        [{"page4-s1", "page4-s2"}, {"page5-s1", "page5-s2"}],
    )

    assert metrics == {
        "hit": True,
        "complete_hit": True,
        "precision": 1.0,
        "recall": 1.0,
    }


def test_agent_metrics_use_best_alternative_evidence_group() -> None:
    report = evaluate_records([{
        "mode": "text_agentic_rag",
        "expected_evidence_ids": ["page4", "page5"],
        "expected_evidence_groups": [["page4"], ["page5"]],
        "ranked_evidence_ids": ["page4"],
        "returned_evidence_ids": ["page4"],
        "task_success": True,
    }])

    assert report["overall"]["retrieval"]["recall_at_k"] == 1.0
    assert report["overall"]["evidence"]["recall"] == 1.0


def test_cross_modal_top_k_uses_each_modality_ranking_independently() -> None:
    text_ids = [f"text-{index}" for index in range(10)]
    report = evaluate_records([{
        "mode": "multimodal_agentic_rag",
        "expected_evidence_ids": ["figure-2"],
        "expected_evidence_groups": [["figure-2"]],
        "expected_figure_ids": ["figure-2"],
        "ranked_evidence_ids": [*text_ids, "figure-2"],
        "ranked_evidence_ids_by_modality": {
            "text": text_ids,
            "figure": ["figure-2", "figure-4"],
            "table": [],
        },
        "returned_evidence_ids": ["figure-2"],
        "returned_figure_ids": ["figure-2"],
    }], top_k=5)

    assert report["overall"]["retrieval"]["recall_at_k"] == 1.0
    assert report["overall"]["retrieval"]["mrr"] == 1.0
    assert report["overall"]["multimodal"]["figure_retrieval_recall_at_k"] == 1.0
    assert report["overall"]["multimodal"]["figure_retrieval_mrr"] == 1.0


def test_text_retrieval_uses_chunk_ids_but_evidence_uses_sentence_ids() -> None:
    report = evaluate_records([{
        "mode": "standard_rag",
        "expected_evidence_ids": ["paper-p1-b2-s3"],
        "expected_evidence_groups": [["paper-p1-b2-s3"]],
        "expected_chunk_ids": ["paper-p1-b2"],
        "expected_chunk_groups": [["paper-p1-b2"]],
        "expected_text_ids": ["paper-p1-b2-s3"],
        "expected_modalities": ["text"],
        "ranked_evidence_ids": ["paper-p1-b2", "noise"],
        "ranked_evidence_ids_by_modality": {
            "text": ["paper-p1-b2", "noise"], "figure": [], "table": [],
        },
        "returned_evidence_ids": ["paper-p1-b2-s3"],
        "returned_text_ids": ["paper-p1-b2-s3"],
    }])

    assert report["overall"]["retrieval"]["recall_at_k"] == 1.0
    assert report["overall"]["multimodal"]["text_retrieval_recall_at_k"] == 1.0
    assert report["overall"]["evidence"]["recall"] == 1.0


def test_task_success_is_unknown_without_judge() -> None:
    report = evaluate_records([{
        "mode": "multimodal_agentic_rag",
        "execution_success": True,
        "answer_correct": None,
        "task_success": None,
    }])

    agent = report["overall"]["agent"]
    assert agent["execution_success_rate"] == 1.0
    assert agent["answer_correct_rate"] is None
    assert agent["task_success_rate"] is None


def test_modality_routing_uses_router_output_even_when_generation_refuses() -> None:
    report = evaluate_records([{
        "mode": "multimodal_agentic_rag",
        "expected_modalities": ["figure"],
        "routed_modalities": ["figure"],
        "predicted_modalities": [],
        "status": "unsupported_claims",
        "execution_success": True,
    }])

    assert report["overall"]["multimodal"]["modality_routing_accuracy"] == 1.0


def test_rag_summary_scores_answers_refusals_and_gold_citations() -> None:
    records = [
        {
            "status": "ok", "gold_answerable": True, "predicted_answerable": True,
            "answerability_correct": True, "gold_citation_hit": True,
            "gold_citation_precision": 1.0, "gold_citation_recall": 0.5,
            "judge": {"score": 2, "correct": True}, "latency_ms": 100.0,
            "execution_success": True, "answer_correct": True, "task_success": True,
        },
        {
            "status": "ok", "gold_answerable": False, "predicted_answerable": False,
            "answerability_correct": True, "gold_citation_hit": None,
            "gold_citation_precision": None, "gold_citation_recall": None,
            "judge": {"score": None, "correct": None}, "latency_ms": 200.0,
            "execution_success": True, "answer_correct": True, "task_success": True,
        },
    ]

    summary = summarize_rag_records(records)

    assert summary["answerability_accuracy"] == 1.0
    assert summary["refusal_accuracy"] == 1.0
    assert summary["gold_citation_hit_rate"] == 1.0
    assert summary["mean_gold_citation_recall"] == 0.5
    assert summary["llm_answer_score_0_to_2"] == 2.0
    assert summary["answer_evaluated_count"] == 2
    assert summary["answer_correct_count"] == 2


def test_execution_success_rate_includes_failed_attempts() -> None:
    records = [
        {
            "status": "ok", "gold_answerable": True,
            "predicted_answerable": True, "answerability_correct": True,
            "gold_citation_hit": False, "gold_citation_precision": 0.0,
            "gold_citation_recall": 0.0, "judge": {}, "latency_ms": 1.0,
            "execution_success": True, "answer_correct": None, "task_success": None,
        },
        {
            "status": "paper_not_found", "execution_success": False,
            "answer_correct": None, "task_success": None,
        },
    ]

    summary = summarize_rag_records(records)

    assert summary["execution_success_rate"] == 0.5


def test_rag_summary_reports_multi_agent_recovery_metrics() -> None:
    records = [{
        "status": "ok", "gold_answerable": True, "predicted_answerable": True,
        "answerability_correct": True, "gold_citation_hit": True,
        "gold_citation_precision": 1.0, "gold_citation_recall": 1.0,
        "judge": {}, "latency_ms": 10.0, "execution_success": True,
        "answer_correct": None, "task_success": None, "agent_count": 5,
        "retry_count": 1,
        "recovery": {"attempted": True, "successful": True},
        "node_traces": [
            {"status": "failed"}, {"status": "success"}, {"status": "approved"},
        ],
    }]

    summary = summarize_rag_records(records)

    assert summary["mean_agent_count"] == 5.0
    assert summary["retry_case_rate"] == 1.0
    assert summary["node_failure_rate"] == 0.3333
    assert summary["recovery_success_rate"] == 1.0


def test_rag_summary_reports_dataset_issue_without_counting_answer_wrong() -> None:
    records = [{
        "case_id": "bad-gold",
        "status": "ok",
        "gold_answerable": True,
        "predicted_answerable": True,
        "answerability_correct": True,
        "gold_citation_hit": True,
        "gold_citation_precision": 1.0,
        "gold_citation_recall": 1.0,
        "judge": {"score": None, "correct": None, "dataset_issue": True},
        "execution_success": True,
        "answer_correct": None,
        "task_success": None,
        "latency_ms": 1.0,
    }]

    summary = summarize_rag_records(records)

    assert summary["dataset_issue_count"] == 1
    assert summary["dataset_issue_case_ids"] == ["bad-gold"]
    assert summary["answer_correct_count"] == 0
    assert summary["answer_correct_rate"] is None


def test_unanswerable_cases_do_not_lower_retrieval_metrics(monkeypatch, tmp_path) -> None:
    local_settings = replace(
        settings,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        vector_enabled=False,
        reranker_enabled=False,
    )
    repository = PaperRepository(local_settings.database_path)
    chunk = Chunk("gold", "paper", 1, "Method", "alpha retrieval method", None)
    repository.save(Paper("paper", "paper.pdf", "Paper", "", "paper.pdf", "test", [chunk], []), None)
    rows = [
        {
            "paper_id": "paper", "question": "alpha", "answerable": True,
            "expected_chunk_ids": ["gold"],
        },
        {
            "paper_id": "paper", "question": "missing information", "answerable": False,
            "expected_chunk_ids": [],
        },
    ]
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr(evaluate_module, "settings", local_settings)

    report = evaluate_module.evaluate(dataset, 5, strategy="bm25")

    assert report["retrieval_count"] == 1
    assert report["unanswerable_count"] == 1
    assert report["citation_hit_at_5"] == 1.0


def test_evaluate_filters_split_and_reports_per_paper(monkeypatch, tmp_path) -> None:
    local_settings = replace(
        settings,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        vector_enabled=False,
        reranker_enabled=False,
    )
    repository = PaperRepository(local_settings.database_path)
    for paper_id, term in (("paper-dev", "alpha"), ("paper-test", "beta")):
        chunk = Chunk(f"{paper_id}-gold", paper_id, 1, "Method", term, None)
        repository.save(
            Paper(paper_id, "paper.pdf", "Paper", "", "paper.pdf", "test", [chunk], []),
            None,
        )
    rows = [
        {
            "paper_id": "paper-dev", "question": "alpha", "answerable": True,
            "expected_chunk_ids": ["paper-dev-gold"], "split": "dev",
        },
        {
            "paper_id": "paper-test", "question": "beta", "answerable": True,
            "expected_chunk_ids": ["paper-test-gold"], "split": "test",
        },
    ]
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr(evaluate_module, "settings", local_settings)

    report = evaluate_module.evaluate(dataset, 5, split="dev", strategy="bm25")

    assert report["count"] == 1
    assert report["configuration"]["split"] == "dev"
    assert report["configuration"]["strategy"] == "bm25"
    assert [item["paper_id"] for item in report["per_paper"]] == ["paper-dev"]
    assert "cold_start_latency_ms" in report
    assert "mean_warm_latency_ms" in report


def test_evaluate_rag_runs_query_retrieval_answer_and_refusal(monkeypatch, tmp_path) -> None:
    local_settings = replace(
        settings,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        vector_enabled=True,
        reranker_enabled=True,
        deepseek_key="test",
    )
    repository = PaperRepository(local_settings.database_path)
    chunk = Chunk("gold", "paper", 1, "Method", "alpha evidence", None, sentence_ids=["s1"])
    sentence = Sentence("s1", "paper", "gold", 1, "Method", "alpha evidence", None)
    repository.save(
        Paper("paper", "paper.pdf", "Paper", "", "paper.pdf", "test", [chunk], [], [sentence]),
        None,
    )
    rows = [
        {
            "case_id": "answerable", "paper_id": "paper", "question": "alpha?",
            "expected_answer": "alpha", "answerable": True,
            "expected_sentence_ids": ["s1"], "gold_quotes": ["alpha evidence"],
            "split": "dev",
        },
        {
            "case_id": "refusal", "paper_id": "paper", "question": "missing?",
            "expected_answer": "missing", "answerable": False,
            "expected_sentence_ids": [], "gold_quotes": [], "split": "dev",
        },
    ]
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    class FakeRetriever:
        def __init__(self, _settings):
            pass

        def search_with_trace(self, chunks, _query, **_kwargs):
            return [SearchResult(chunks[0], 1.0)], {
                "requested_strategy": "hybrid-rerank",
                "effective_strategy": "hybrid-rerank",
                "degraded": False,
                "warnings": [],
            }

        def close(self):
            pass

    class FakeReading:
        def __init__(self, _settings, _retriever):
            pass

        def plan_query(self, question):
            return {"semantic_query": question, "lexical_query": question, "translated": False}

        def answer_with_evidence(self, question, _chunks, _sentences):
            if question == "missing?":
                return {"answer": "拒答", "answerable": False, "claims": [], "citations": [], "refusal_reason": "missing", "status": "insufficient_evidence"}
            return {"answer": "alpha", "answerable": True, "claims": [], "citations": [{"id": "s1"}], "refusal_reason": "", "status": "ok"}

    monkeypatch.setattr(evaluate_module, "settings", local_settings)
    monkeypatch.setattr(evaluate_module, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(evaluate_module, "ReadingService", FakeReading)

    report = evaluate_module.evaluate_rag(dataset, 5, split="dev")

    assert report["answerability_accuracy"] == 1.0
    assert report["refusal_accuracy"] == 1.0
    assert report["gold_citation_hit_rate"] == 1.0
    assert report["configuration"]["query_rewrite_enabled"] is True


def test_evaluate_rag_records_multimodal_agent_tools_and_trace(monkeypatch, tmp_path) -> None:
    local_settings = replace(
        settings,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        vector_enabled=False,
        reranker_enabled=False,
        deepseek_key="test",
        qwen_vl_key="test",
        agent_orchestrator="legacy",
    )
    repository = PaperRepository(local_settings.database_path)
    figure = Figure(
        "figure-2", "paper", 2, "picture", "Figure 2: Labels",
        None, "figure-2.png", "", "Method", "", [],
    )
    repository.save(
        Paper("paper", "paper.pdf", "Paper", "", "paper.pdf", "test", [], [figure]),
        None,
    )
    row = {
        "case_id": "figure-case",
        "paper_id": "paper",
        "question": "What label is visible in Figure 2?",
        "expected_answer": "Harmful",
        "answerable": True,
        "expected_sentence_ids": [],
        "expected_evidence_ids": ["figure-2"],
        "expected_evidence_groups": [["figure-2"]],
        "expected_modalities": ["figure"],
        "expected_figure_ids": ["figure-2"],
        "expected_text_ids": [],
        "expected_table_ids": [],
        "expected_tools": ["search_figures", "analyze_figure_for_query"],
        "gold_quotes": [],
        "tags": ["visual-only"],
        "split": "dev",
    }
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(json.dumps(row), encoding="utf-8")

    class FakeRetriever:
        def __init__(self, _settings):
            pass

        def close(self):
            pass

    class FakeReading:
        token_usage = 0

        def __init__(self, _settings, _retriever):
            pass

    class FakeAgentExecutor:
        def __init__(self, *_args):
            pass

        def run(self, _paper, _question, _route, strategy=None):
            assert strategy == "bm25"
            trace = {
                "rewrites": [{"query": "Figure 2 label"}],
                "steps": [
                    {"tool": "search_figures", "result_ids": ["figure-2"]},
                    {"tool": "analyze_figure_for_query", "result_ids": ["figure-2"]},
                ],
                "total_steps": 2,
                "qwen_vl_calls": 1,
                "token_usage": 42,
                "retrieval": {"effective_strategy": "bm25", "degraded": False},
            }
            return {
                "answer": "Harmful",
                "answerable": True,
                "claims": [{"verification_status": "supported"}],
                "citations": [{"id": "figure-2", "evidence_id": "figure-2", "type": "figure"}],
                "refusal_reason": "",
                "status": "ok",
                "mode": "multimodal_agentic_rag",
                "evidence_sufficient": True,
                "retrieval": trace["retrieval"],
                "trace": trace,
            }

    class FakeFigureUnderstanding:
        def __init__(self, *_args):
            self.judge_call_count = 0

        def judge_answer(self, _figure, _question, _expected, _actual):
            self.judge_call_count += 1
            return {
                "score": 0,
                "correct": False,
                "reason": "The visible label differs.",
                "kind": "visual",
            }

    monkeypatch.setattr(evaluate_module, "settings", local_settings)
    monkeypatch.setattr(evaluate_module, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(evaluate_module, "ReadingService", FakeReading)
    monkeypatch.setattr(evaluate_module, "AgentExecutor", FakeAgentExecutor)
    monkeypatch.setattr(
        evaluate_module, "FigureUnderstandingService", FakeFigureUnderstanding
    )

    report = evaluate_module.evaluate_rag(dataset, 5, split="dev", strategy="bm25")
    record = report["records"][0]

    assert record["mode"] == "multimodal_agentic_rag"
    assert record["tool_calls"] == ["search_figures", "analyze_figure_for_query"]
    assert record["qwen_vl_calls"] == 1
    assert record["gold_complete_citation_hit"] is True
    assert record["returned_figure_ids"] == ["figure-2"]
    assert record["execution_success"] is True
    assert record["answer_correct"] is None
    assert record["task_success"] is None
    assert report["agentic_metrics"]["overall"]["agent"]["task_success_rate"] is None

    monkeypatch.setattr(
        evaluate_module,
        "build_agent_executor",
        lambda *_args: FakeAgentExecutor(),
    )
    langgraph_report = evaluate_module.evaluate_rag(
        dataset, 5, split="dev", strategy="bm25", orchestrator="langgraph"
    )
    assert langgraph_report["configuration"]["orchestrator"] == "langgraph"

    judged_report = evaluate_module.evaluate_rag(
        dataset, 5, split="dev", strategy="bm25", judge=True
    )
    judged = judged_report["records"][0]
    assert judged["judge"]["kind"] == "visual"
    assert judged["visual_judge_qwen_vl_calls"] == 1
    assert judged["answer_correct"] is False
    assert judged["task_success"] is False
