"""评测单论文问答检索的证据命中率、排名质量与延迟。"""

import argparse
from collections import defaultdict
from dataclasses import replace
import json
import math
import time
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.config import settings
from app.agent.executor import AgentExecutor
from app.agent.router import QueryRouter
from app.evaluation import evaluate_records
from app.multimodal import FigureUnderstandingService
from app.multi_agent import build_agent_executor
from app.repository import PaperRepository
from app.services.reading import ReadingService
from app.services.retrieval import HybridRetriever, RETRIEVAL_STRATEGIES


def load_rows(path: Path, split: str = "all") -> list[dict]:
    """读取 JSONL，并按论文级 dev/test 切分过滤。"""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if split == "all":
        return rows
    return [row for row in rows if row.get("split") == split]


def ranking_metrics(returned: list[str], expected: set[str], limit: int) -> dict:
    """计算一个完整证据组的 Recall、MRR 与 nDCG。"""
    ranked = returned[:limit]
    relevant_ranks = [index for index, item in enumerate(ranked, start=1) if item in expected]
    recall = len(set(ranked).intersection(expected)) / max(len(expected), 1)
    reciprocal_rank = 1 / relevant_ranks[0] if relevant_ranks else 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_count = min(len(expected), limit)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        "recall": round(recall, 4),
        "reciprocal_rank": round(reciprocal_rank, 4),
        "ndcg": round(dcg / ideal_dcg if ideal_dcg else 0.0, 4),
    }


def expected_chunk_groups(row: dict) -> list[set[str]]:
    """读取可替代完整证据组；旧数据自动视为单个证据组。"""
    groups = row.get("expected_chunk_groups")
    if isinstance(groups, list) and groups:
        return [set(group) for group in groups if isinstance(group, list) and group]
    expected = set(row.get("expected_chunk_ids", []))
    return [expected] if expected else []


def expected_sentence_groups(row: dict) -> list[set[str]]:
    """读取可替代 Sentence 证据组；旧数据退化为单个 Sentence 集合。"""
    groups = row.get("expected_sentence_groups")
    if isinstance(groups, list) and groups:
        return [set(group) for group in groups if isinstance(group, list) and group]
    expected = set(row.get("expected_sentence_ids", []))
    return [expected] if expected else []


def expected_evidence_groups(row: dict) -> list[set[str]]:
    """读取统一 Evidence 的可替代组，并兼容只有 Sentence Gold 的旧题集。"""
    groups = row.get("expected_evidence_groups")
    if isinstance(groups, list) and groups:
        return [set(group) for group in groups if isinstance(group, list) and group]
    expected = set(row.get("expected_evidence_ids", []))
    if expected:
        return [expected]
    return expected_sentence_groups(row)


def alternative_citation_metrics(
    returned: set[str],
    groups: list[set[str]],
) -> dict:
    """在可替代 Gold 组中选择覆盖最完整的一组计算端到端引用指标。"""
    if not groups:
        return {"hit": False, "complete_hit": False, "precision": 0.0, "recall": 0.0}
    candidates = []
    for group in groups:
        overlap = returned.intersection(group)
        candidates.append({
            "hit": bool(overlap),
            "complete_hit": group.issubset(returned),
            "precision": round(len(overlap) / len(returned), 4) if returned else 0.0,
            "recall": round(len(overlap) / len(group), 4),
        })
    return max(
        candidates,
        key=lambda item: (
            item["complete_hit"], item["recall"], item["precision"], item["hit"]
        ),
    )


def trace_result_ids(trace: dict) -> list[str]:
    """按 Agent 搜索顺序展开并去重所有候选 Evidence ID。"""
    output: list[str] = []
    for step in trace.get("steps", []):
        if not str(step.get("tool", "")).startswith("search_"):
            continue
        for evidence_id in step.get("result_ids", []):
            if evidence_id and evidence_id not in output:
                output.append(evidence_id)
    return output


def trace_result_ids_by_modality(trace: dict) -> dict[str, list[str]]:
    """分别保留各检索工具内部的排名，避免跨模态拼接破坏 Top-K。"""
    tool_modalities = {
        "search_text": "text",
        "search_figures": "figure",
        "search_tables": "table",
    }
    output = {modality: [] for modality in ("text", "figure", "table")}
    for step in trace.get("steps", []):
        modality = tool_modalities.get(str(step.get("tool", "")))
        if not modality:
            continue
        for evidence_id in step.get("result_ids", []):
            if evidence_id and evidence_id not in output[modality]:
                output[modality].append(evidence_id)
    return output


def citation_evidence_for_judge(citations: list[dict]) -> list[str]:
    """提取回答实际引用的原始证据，供独立文本 Judge 复核补充细节。"""
    output: list[str] = []
    for citation in citations[:10]:
        evidence_id = citation.get("evidence_id", citation.get("id", ""))
        modality = citation.get("type", "text")
        content = next((
            str(citation.get(key, "")).strip()
            for key in ("quote", "text", "content")
            if str(citation.get(key, "")).strip()
        ), "")
        if not content:
            metadata = citation.get("metadata", {})
            if isinstance(metadata, dict):
                content = str(metadata.get("caption", "")).strip()
        if content:
            output.append(f"[{modality}:{evidence_id}] {content[:1500]}")
    return output


def alternative_ranking_metrics(
    returned: list[str],
    groups: list[set[str]],
    limit: int,
) -> dict:
    """对每个可替代证据组评分，并采用覆盖最完整且排名最优的一组。"""
    if not groups:
        return {"recall": 0.0, "reciprocal_rank": 0.0, "ndcg": 0.0}
    candidates = [ranking_metrics(returned, group, limit) for group in groups]
    return max(
        candidates,
        key=lambda item: (item["recall"], item["reciprocal_rank"], item["ndcg"]),
    )


def summarize_records(records: list[dict], limit: int) -> dict:
    """汇总整体或单篇论文的检索和延迟指标。"""
    completed = [record for record in records if record.get("status") == "ok"]
    retrieval_records = [record for record in completed if record["answerable"]]
    latencies = sorted(item["latency_ms"] for item in completed)
    p95_latency = (
        latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)]
        if latencies
        else 0.0
    )
    return {
        "count": len(completed),
        "retrieval_count": len(retrieval_records),
        "unanswerable_count": len(completed) - len(retrieval_records),
        f"citation_hit_at_{limit}": round(
            sum(item["citation_hit"] for item in retrieval_records)
            / max(len(retrieval_records), 1),
            4,
        ),
        f"complete_evidence_hit_at_{limit}": round(
            sum(item["complete_evidence_hit"] for item in retrieval_records)
            / max(len(retrieval_records), 1),
            4,
        ),
        f"recall_at_{limit}": round(
            sum(item["recall"] for item in retrieval_records)
            / max(len(retrieval_records), 1),
            4,
        ),
        "mrr": round(
            sum(item["reciprocal_rank"] for item in retrieval_records)
            / max(len(retrieval_records), 1),
            4,
        ),
        f"ndcg_at_{limit}": round(
            sum(item["ndcg"] for item in retrieval_records)
            / max(len(retrieval_records), 1),
            4,
        ),
        "mean_latency_ms": round(
            sum(item["latency_ms"] for item in completed) / max(len(completed), 1),
            1,
        ),
        "p95_latency_ms": round(p95_latency, 1),
        "max_latency_ms": round(max(latencies, default=0.0), 1),
    }


def summarize_rag_records(records: list[dict]) -> dict:
    """汇总端到端回答、拒答、引用覆盖、裁判分数和延迟。"""
    attempted = list(records)
    completed = [record for record in records if record.get("status") not in ("paper_not_found", "retrieval_error")]
    answerable = [record for record in completed if record["gold_answerable"]]
    unanswerable = [record for record in completed if not record["gold_answerable"]]
    decided = [record for record in completed if record["predicted_answerable"] is not None]
    judged = [record for record in answerable if record.get("judge", {}).get("score") is not None]
    dataset_issues = [
        record for record in completed
        if record.get("judge", {}).get("dataset_issue") is True
    ]
    semantically_decided = [
        record for record in completed if record.get("answer_correct") is not None
    ]
    latencies = sorted(record["latency_ms"] for record in completed)
    p95 = latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)] if latencies else 0.0
    node_traces = [
        node
        for record in completed
        for node in record.get("node_traces", [])
    ]
    recovery_attempts = [
        record for record in completed if record.get("recovery", {}).get("attempted")
    ]

    def average(items: list[dict], field: str) -> float:
        values = [item[field] for item in items if item.get(field) is not None]
        return round(sum(values) / max(len(values), 1), 4)

    return {
        "count": len(completed),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "decision_count": len(decided),
        "execution_success_rate": round(
            sum(item.get("execution_success") is True for item in attempted)
            / max(len(attempted), 1),
            4,
        ),
        "answer_evaluated_count": len(semantically_decided),
        "answer_correct_count": sum(
            item.get("answer_correct") is True for item in semantically_decided
        ),
        "dataset_issue_count": len(dataset_issues),
        "dataset_issue_case_ids": [item.get("case_id") for item in dataset_issues],
        "answer_correct_rate": (
            round(
                sum(item.get("answer_correct") is True for item in semantically_decided)
                / len(semantically_decided),
                4,
            )
            if semantically_decided else None
        ),
        "task_success_rate": (
            round(
                sum(item.get("task_success") is True for item in semantically_decided)
                / len(semantically_decided),
                4,
            )
            if semantically_decided else None
        ),
        "answerability_accuracy": round(
            sum(item["answerability_correct"] for item in decided) / max(len(decided), 1), 4
        ),
        "refusal_accuracy": round(
            sum(item["predicted_answerable"] is False for item in unanswerable)
            / max(len(unanswerable), 1), 4
        ),
        "answer_delivery_rate": round(
            sum(item["predicted_answerable"] is True for item in answerable)
            / max(len(answerable), 1), 4
        ),
        "gold_citation_hit_rate": round(
            sum(item["gold_citation_hit"] for item in answerable) / max(len(answerable), 1), 4
        ),
        "complete_gold_citation_hit_rate": round(
            sum(item.get("gold_complete_citation_hit", False) for item in answerable)
            / max(len(answerable), 1),
            4,
        ),
        "mean_gold_citation_precision": average(answerable, "gold_citation_precision"),
        "mean_gold_citation_recall": average(answerable, "gold_citation_recall"),
        "llm_judged_count": len(judged),
        "llm_answer_score_0_to_2": (
            average([{"score": item["judge"]["score"]} for item in judged], "score")
            if judged else None
        ),
        "llm_exact_correct_rate": (
            round(sum(item["judge"]["correct"] is True for item in judged) / len(judged), 4)
            if judged else None
        ),
        "mean_end_to_end_latency_ms": round(
            sum(latencies) / max(len(latencies), 1), 1
        ),
        "p95_end_to_end_latency_ms": round(p95, 1),
        "max_end_to_end_latency_ms": round(max(latencies, default=0.0), 1),
        "mean_agent_count": average(completed, "agent_count"),
        "retry_case_rate": round(
            sum(int(item.get("retry_count", 0)) > 0 for item in completed)
            / max(len(completed), 1),
            4,
        ),
        "mean_retry_count": average(completed, "retry_count"),
        "node_failure_rate": round(
            sum(item.get("status") in {"failed", "unavailable"} for item in node_traces)
            / max(len(node_traces), 1),
            4,
        ),
        "recovery_attempt_count": len(recovery_attempts),
        "recovery_success_rate": (
            round(
                sum(item.get("recovery", {}).get("successful") is True for item in recovery_attempts)
                / len(recovery_attempts),
                4,
            ) if recovery_attempts else None
        ),
    }


def evaluate(
    dataset: Path,
    limit: int,
    rewrite_queries: bool = False,
    split: str = "all",
    strategy: str = "hybrid-rerank",
) -> dict:
    """对指定 split 执行严格、可复现的检索评测。"""
    if split not in ("dev", "test", "all"):
        raise ValueError(f"未知数据集切分：{split}")
    if strategy not in RETRIEVAL_STRATEGIES:
        raise ValueError(f"未知检索策略：{strategy}")

    repository = PaperRepository(settings.database_path)
    retriever = HybridRetriever(settings)
    reading = ReadingService(settings, retriever)
    rows = load_rows(dataset, split)
    if not rows:
        raise ValueError(f"评测集中没有 split={split} 的样本")

    records: list[dict] = []
    try:
        for row in rows:
            result = repository.get(row["paper_id"])
            if not result:
                records.append({
                    "paper_id": row["paper_id"],
                    "question": row.get("question"),
                    "status": "paper_not_found",
                })
                continue
            paper, _ = result
            query_plan = (
                reading.plan_query(row["question"])
                if rewrite_queries
                else {
                    "semantic_query": row["question"],
                    "lexical_query": row["question"],
                    "translated": False,
                }
            )
            started = time.perf_counter()
            hits = retriever.search(
                paper.chunks,
                query_plan["semantic_query"],
                limit=limit,
                lexical_query=query_plan["lexical_query"],
                section_hints=tuple(query_plan.get("section_hints", ())),
                strategy=strategy,
                strict=True,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            groups = expected_chunk_groups(row)
            expected_union = set().union(*groups) if groups else set()
            returned = [item.chunk.id for item in hits]
            returned_set = set(returned[:limit])
            answerable = bool(row.get("answerable", True))
            metrics = (
                alternative_ranking_metrics(returned, groups, limit)
                if answerable
                else {"recall": None, "reciprocal_rank": None, "ndcg": None}
            )
            records.append({
                "paper_id": paper.id,
                "question": row["question"],
                "answerable": answerable,
                "expected_chunk_ids": sorted(expected_union),
                "expected_chunk_groups": [sorted(group) for group in groups],
                "returned_chunk_ids": returned,
                "citation_hit": (
                    any(group.intersection(returned_set) for group in groups)
                    if answerable
                    else None
                ),
                "complete_evidence_hit": (
                    any(group.issubset(returned_set) for group in groups)
                    if answerable
                    else None
                ),
                "query_plan": query_plan,
                **metrics,
                "latency_ms": elapsed_ms,
                "status": "ok",
            })
    finally:
        retriever.close()

    completed = [record for record in records if record.get("status") == "ok"]
    summary = summarize_records(completed, limit)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in completed:
        grouped[record["paper_id"]].append(record)
    per_paper = [
        {"paper_id": paper_id, **summarize_records(items, limit)}
        for paper_id, items in grouped.items()
    ]
    cold_start = completed[0]["latency_ms"] if completed else 0.0
    warm_records = completed[1:]
    mean_warm = (
        sum(item["latency_ms"] for item in warm_records) / len(warm_records)
        if warm_records
        else 0.0
    )
    return {
        "dataset": str(dataset),
        **summary,
        "configuration": {
            "split": split,
            "strategy": strategy,
            "vector_enabled": settings.vector_enabled,
            "reranker_enabled": settings.reranker_enabled,
            "query_rewrite_enabled": rewrite_queries,
            "limit": limit,
        },
        "cold_start_latency_ms": round(cold_start, 1),
        "mean_warm_latency_ms": round(mean_warm, 1),
        "per_paper": per_paper,
        "records": records,
    }


def evaluate_rag(
    dataset: Path,
    limit: int,
    split: str = "dev",
    strategy: str = "hybrid-rerank",
    judge: bool = False,
    case_id: str | None = None,
    orchestrator: str | None = None,
    parallel: bool | None = None,
) -> dict:
    """执行与网页一致的路由、Agent 工具、回答、验证和拒答链路。"""
    if split not in ("dev", "test", "all"):
        raise ValueError(f"未知数据集切分：{split}")
    if strategy not in RETRIEVAL_STRATEGIES:
        raise ValueError(f"未知检索策略：{strategy}")
    selected_orchestrator = orchestrator or settings.agent_orchestrator
    if selected_orchestrator not in {"legacy", "langgraph"}:
        raise ValueError(f"未知编排器：{selected_orchestrator}")
    active_settings = replace(
        settings,
        agent_orchestrator=selected_orchestrator,
        multi_agent_parallel_enabled=(
            settings.multi_agent_parallel_enabled if parallel is None else parallel
        ),
    )
    repository = PaperRepository(active_settings.database_path)
    retriever = HybridRetriever(active_settings)
    reading = ReadingService(active_settings, retriever)
    query_router = QueryRouter()
    figure_understanding = FigureUnderstandingService(active_settings, repository)
    agent_executor = (
        build_agent_executor(active_settings, retriever, reading, figure_understanding)
        if selected_orchestrator == "langgraph"
        else AgentExecutor(active_settings, retriever, reading, figure_understanding)
    )
    rows = load_rows(dataset, split)
    if case_id:
        rows = [row for row in rows if row.get("case_id") == case_id]
    if not rows:
        raise ValueError(f"评测集中没有 split={split} 的样本")

    records: list[dict] = []
    try:
        for row in rows:
            result = repository.get(row["paper_id"])
            if not result:
                records.append({
                    "case_id": row.get("case_id"),
                    "paper_id": row["paper_id"],
                    "execution_success": False,
                    "answer_correct": False if judge else None,
                    "task_success": False if judge else None,
                    "status": "paper_not_found",
                })
                continue
            paper, _ = result
            started = time.perf_counter()
            initial_reading_tokens = getattr(reading, "token_usage", 0)
            route = query_router.route(row["question"])
            if route["route"] == "agentic_rag":
                grounded = agent_executor.run(
                    paper,
                    row["question"],
                    route,
                    strategy=strategy,
                )
                trace = grounded.get("trace", {})
                query_plan = {"router": route, "rewrites": trace.get("rewrites", [])}
            else:
                query_plan = reading.plan_query(row["question"])
                hits, retrieval_trace = retriever.search_with_trace(
                    paper.chunks,
                    query_plan["semantic_query"],
                    limit=limit,
                    lexical_query=query_plan["lexical_query"],
                    section_hints=tuple(query_plan.get("section_hints", ())),
                    strategy=strategy,
                    strict=True,
                )
                grounded = reading.answer_with_evidence(
                    row["question"], [item.chunk for item in hits], paper.sentences
                )
                elapsed = round((time.perf_counter() - started) * 1000, 1)
                trace = {
                    "run_id": f"eval-{row.get('case_id', len(records) + 1)}",
                    "query": row["question"],
                    "route": "standard_rag",
                    "router": route,
                    "plan": {},
                    "rewrites": [query_plan],
                    "steps": [{
                        "step": 1,
                        "subtask_id": "standard_rag",
                        "tool": "search_text",
                        "arguments": {"query": query_plan["semantic_query"], "top_k": limit},
                        "result_ids": [item.chunk.id for item in hits],
                        "latency_ms": elapsed,
                        "cached": False,
                        "retrieval": retrieval_trace,
                    }],
                    "final_evidence_ids": [
                        item.get("evidence_id", item.get("id"))
                        for item in grounded.get("citations", [])
                    ],
                    "total_steps": 1,
                    "total_latency_ms": elapsed,
                    "token_usage": (
                        getattr(reading, "token_usage", 0) - initial_reading_tokens
                    ),
                    "qwen_vl_calls": 0,
                    "sufficiency": {
                        "sufficient": grounded.get("answerable") is True,
                        "covered_subtasks": (
                            ["standard_rag"] if grounded.get("answerable") is True else []
                        ),
                        "missing_subtasks": (
                            [] if grounded.get("answerable") is True else ["standard_rag"]
                        ),
                        "missing_information": [],
                    },
                    "evidence_memory": [],
                    "retrieval": retrieval_trace,
                    "orchestrator": "standard_rag",
                    "node_traces": [],
                    "agent_count": 0,
                    "retry_count": 0,
                    "recovery": {},
                }
                grounded = {
                    **grounded,
                    "route": "standard_rag",
                    "mode": "standard_rag",
                    "router": route,
                    "trace": trace,
                    "retrieval": retrieval_trace,
                    "evidence_sufficient": grounded.get("answerable") is True,
                    "latency_ms": elapsed,
                }
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            gold_answerable = bool(row.get("answerable", True))
            predicted_answerable = grounded.get("answerable")
            sentence_groups = expected_sentence_groups(row)
            evidence_groups = expected_evidence_groups(row)
            gold_sentence_ids = set().union(*sentence_groups) if sentence_groups else set()
            citations = grounded.get("citations", [])
            citation_ids = {
                item.get("evidence_id", item.get("id"))
                for item in grounded.get("citations", [])
                if item.get("evidence_id", item.get("id"))
            }
            citation_metrics = alternative_citation_metrics(citation_ids, evidence_groups)
            answer_judge = {
                "score": None, "correct": None, "reason": "未启用",
                "kind": None, "dataset_issue": None,
            }
            visual_judge_qwen_vl_calls = 0
            if judge and gold_answerable and predicted_answerable is True:
                if "visual-only" in row.get("tags", []):
                    expected_figures = set(row.get("expected_figure_ids", []))
                    figure = next((
                        item for item in paper.figures if item.id in expected_figures
                    ), None)
                    if figure:
                        before = figure_understanding.judge_call_count
                        answer_judge = figure_understanding.judge_answer(
                            figure,
                            row["question"],
                            str(row.get("expected_answer", "")),
                            grounded.get("answer", ""),
                        )
                        visual_judge_qwen_vl_calls = (
                            figure_understanding.judge_call_count - before
                        )
                    else:
                        answer_judge = {
                            "score": None,
                            "correct": None,
                            "reason": "评测集指定的 Figure 在论文记录中不存在。",
                            "kind": "visual",
                            "dataset_issue": True,
                        }
                else:
                    answer_judge = {
                        **reading.judge_answer(
                            row["question"],
                            str(row.get("expected_answer", "")),
                            grounded.get("answer", ""),
                            list(row.get("gold_quotes", [])),
                            citation_evidence_for_judge(citations),
                        ),
                        "kind": "text",
                    }
            trace = grounded.get("trace", trace)
            tool_calls = [item.get("tool", "") for item in trace.get("steps", [])]
            returned_by_type = {
                modality: sorted({
                    item.get("evidence_id", item.get("id"))
                    for item in citations
                    if item.get("type", "text") == modality
                    and item.get("evidence_id", item.get("id"))
                })
                for modality in ("text", "figure", "table")
            }
            execution_success = (
                predicted_answerable in (True, False)
                and grounded.get("status") not in {
                    "generation_error", "protocol_error", "retrieval_error"
                }
            )
            # 未启用 Judge 时，不根据“链路跑完/证据齐全”推断答案正确。
            answer_correct = None
            if judge:
                if not execution_success:
                    answer_correct = False
                elif not gold_answerable:
                    answer_correct = predicted_answerable is False
                elif predicted_answerable is not True:
                    answer_correct = False
                else:
                    answer_correct = answer_judge.get("correct")
            task_success = (
                execution_success and answer_correct
                if answer_correct is not None
                else None
            )
            ranked_by_modality = trace_result_ids_by_modality(trace)
            records.append({
                "case_id": row.get("case_id"),
                "paper_id": paper.id,
                "question": row["question"],
                "mode": grounded.get("mode", grounded.get("route", "standard_rag")),
                "gold_answerable": gold_answerable,
                "predicted_answerable": predicted_answerable,
                "answerability_correct": predicted_answerable == gold_answerable,
                "answer": grounded.get("answer", ""),
                "refusal_reason": grounded.get("refusal_reason", ""),
                "answer_status": grounded.get("status"),
                "claims": grounded.get("claims", []),
                "citation_ids": sorted(citation_ids),
                "gold_sentence_ids": sorted(gold_sentence_ids),
                "gold_sentence_groups": [sorted(group) for group in sentence_groups],
                "gold_evidence_groups": [sorted(group) for group in evidence_groups],
                "gold_citation_hit": citation_metrics["hit"] if gold_answerable else None,
                "gold_complete_citation_hit": (
                    citation_metrics["complete_hit"] if gold_answerable else None
                ),
                "gold_citation_precision": (
                    citation_metrics["precision"] if gold_answerable else None
                ),
                "gold_citation_recall": (
                    citation_metrics["recall"] if gold_answerable else None
                ),
                "query_plan": query_plan,
                "router": route,
                "retrieval": grounded.get("retrieval", trace.get("retrieval", {})),
                "trace": trace,
                "agent_trace": trace,
                "tool_calls": tool_calls,
                "steps": int(trace.get("total_steps", len(tool_calls))),
                "qwen_vl_calls": int(trace.get("qwen_vl_calls", 0)),
                "visual_judge_qwen_vl_calls": visual_judge_qwen_vl_calls,
                "total_qwen_vl_calls": (
                    int(trace.get("qwen_vl_calls", 0)) + visual_judge_qwen_vl_calls
                ),
                "token_usage": int(trace.get("token_usage", 0)),
                "orchestrator": trace.get("orchestrator", selected_orchestrator),
                "agent_count": int(trace.get("agent_count", 0)),
                "retry_count": int(trace.get("retry_count", 0)),
                "recovery": dict(trace.get("recovery", {})),
                "node_traces": list(trace.get("node_traces", [])),
                "expected_tools": list(row.get("expected_tools", [])),
                "expected_modalities": list(row.get("expected_modalities", [])),
                "predicted_modalities": [
                    modality for modality in ("text", "figure", "table")
                    if returned_by_type[modality]
                ],
                "routed_modalities": list(route.get("modalities", ["text"])),
                "required_modalities": list(route.get("required_modalities", ["text"])),
                "expected_evidence_ids": list(row.get("expected_evidence_ids", [])),
                "expected_evidence_groups": [sorted(group) for group in evidence_groups],
                "expected_chunk_ids": list(row.get("expected_chunk_ids", [])),
                "expected_chunk_groups": [
                    sorted(group) for group in expected_chunk_groups(row)
                ],
                "ranked_evidence_ids": trace_result_ids(trace),
                "ranked_evidence_ids_by_modality": ranked_by_modality,
                "ranked_text_ids": ranked_by_modality["text"],
                "ranked_figure_ids": ranked_by_modality["figure"],
                "ranked_table_ids": ranked_by_modality["table"],
                "returned_evidence_ids": sorted(citation_ids),
                "expected_text_ids": list(row.get("expected_text_ids", [])),
                "expected_figure_ids": list(row.get("expected_figure_ids", [])),
                "expected_table_ids": list(row.get("expected_table_ids", [])),
                "returned_text_ids": returned_by_type["text"],
                "returned_figure_ids": returned_by_type["figure"],
                "returned_table_ids": returned_by_type["table"],
                "execution_success": execution_success,
                "answer_correct": answer_correct,
                "task_success": task_success,
                "judge": answer_judge,
                "latency_ms": elapsed_ms,
                "status": "ok",
            })
    finally:
        retriever.close()

    summary = summarize_rag_records(records)
    agentic_metrics = evaluate_records(
        [record for record in records if record.get("status") == "ok"],
        top_k=limit,
    )
    return {
        "dataset": str(dataset),
        **summary,
        "configuration": {
            "mode": "rag",
            "split": split,
            "strategy": strategy,
            "query_rewrite_enabled": True,
            "claim_verification_enabled": True,
            "llm_judge_enabled": judge,
            "visual_judge_enabled_for_visual_only": judge,
            "online_visual_judge_enabled": active_settings.online_visual_judge_enabled,
            "orchestrator": selected_orchestrator,
            "parallel_enabled": active_settings.multi_agent_parallel_enabled,
            "case_id": case_id,
            "limit": limit,
        },
        "agentic_metrics": agentic_metrics,
        "records": records,
    }


def main() -> None:
    """解析命令行参数并输出机器可读报告。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--strategy", choices=RETRIEVAL_STRATEGIES, default="hybrid-rerank")
    parser.add_argument("--mode", choices=("retrieval", "rag"), default="retrieval")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rewrite", action="store_true", help="为中文问题生成英文 BM25 查询")
    parser.add_argument("--judge", action="store_true", help="RAG 模式下额外调用 LLM 评判答案正确性")
    parser.add_argument("--case-id", help="RAG 模式下只运行一个指定 case，便于低成本冒烟测试")
    parser.add_argument(
        "--orchestrator",
        choices=("legacy", "langgraph"),
        default="langgraph",
        help="RAG 模式使用的 Agent 编排器；命令行默认运行 LangGraph 多智能体链路",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=None,
        help="LangGraph 模式下并行执行无依赖 Specialist 任务",
    )
    args = parser.parse_args()
    if args.mode != "rag" and (args.judge or args.case_id or args.parallel):
        parser.error("--judge、--case-id 和 --parallel 只能与 --mode rag 一起使用")
    try:
        report = (
            evaluate_rag(
                args.dataset,
                args.limit,
                split=args.split,
                strategy=args.strategy,
                judge=args.judge,
                case_id=args.case_id,
                orchestrator=args.orchestrator,
                parallel=args.parallel,
            )
            if args.mode == "rag"
            else evaluate(
                args.dataset,
                args.limit,
                args.rewrite,
                split=args.split,
                strategy=args.strategy,
            )
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
