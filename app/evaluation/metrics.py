"""无需外部模型即可复现的检索、证据、Agent 与工程指标。"""

from collections import defaultdict
import re
from statistics import mean


MODES = (
    "standard_rag",
    "text_agentic_rag",
    "multimodal_agentic_rag",
    "text_multi_agent_rag",
    "multimodal_multi_agent_rag",
)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _average(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _optional_average(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None


def _evidence_groups(row: dict) -> list[set[str]]:
    groups = row.get("expected_evidence_groups")
    if isinstance(groups, list) and groups:
        return [set(group) for group in groups if isinstance(group, list) and group]
    gold = set(row.get("expected_evidence_ids", []))
    return [gold] if gold else []


def _text_retrieval_id(evidence_id: str, row: dict) -> str:
    """把句子级 Gold 映射到文本检索实际返回的 Chunk ID。"""
    text_ids = set(row.get("expected_text_ids", []))
    if evidence_id not in text_ids:
        return evidence_id
    chunk_ids = set(row.get("expected_chunk_ids", []))
    inferred = re.sub(r"-s\d+$", "", evidence_id)
    if inferred in chunk_ids:
        return inferred
    if len(chunk_ids) == 1:
        return next(iter(chunk_ids))
    return inferred if inferred != evidence_id else evidence_id


def _retrieval_groups(row: dict) -> list[set[str]]:
    """检索按 Chunk/Figure/Table 评分，引用仍由 Sentence Evidence 评分。"""
    groups = _evidence_groups(row)
    if groups:
        return [
            {_text_retrieval_id(evidence_id, row) for evidence_id in group}
            for group in groups
        ]
    chunk_groups = row.get("expected_chunk_groups")
    if isinstance(chunk_groups, list) and chunk_groups:
        return [set(group) for group in chunk_groups if isinstance(group, list) and group]
    chunks = set(row.get("expected_chunk_ids", []))
    return [chunks] if chunks else []


def _expected_ids_by_modality(row: dict) -> dict[str, set[str]]:
    return {
        "text": {
            _text_retrieval_id(evidence_id, row)
            for evidence_id in row.get("expected_text_ids", [])
        },
        "figure": set(row.get("expected_figure_ids", [])),
        "table": set(row.get("expected_table_ids", [])),
    }


def _modality_for_id(evidence_id: str, row: dict) -> str:
    expected = _expected_ids_by_modality(row)
    for modality, ids in expected.items():
        if evidence_id in ids:
            return modality
    if "-fig_" in evidence_id:
        return "figure"
    if "-table_" in evidence_id:
        return "table"
    return "text"


def _ranked_by_modality(row: dict) -> tuple[dict[str, list[str]], bool]:
    raw = row.get("ranked_evidence_ids_by_modality")
    explicit = isinstance(raw, dict)
    output: dict[str, list[str]] = {}
    for modality in ("text", "figure", "table"):
        values = raw.get(modality, []) if explicit else row.get(f"ranked_{modality}_ids", [])
        output[modality] = list(values) if isinstance(values, list) else []
    return output, explicit or any(output.values())


def _retrieval_for_group(row: dict, gold: set[str], top_k: int) -> tuple[float, float]:
    """在每个模态自己的 Top-K 中计算命中，避免搜索调用顺序制造假阴性。"""
    ranked = list(row.get("ranked_evidence_ids", row.get("returned_evidence_ids", [])))
    ranked_by_modality, has_modality_rankings = _ranked_by_modality(row)
    hits = 0
    reciprocal_ranks: list[float] = []
    for evidence_id in gold:
        modality = _modality_for_id(evidence_id, row)
        candidates = (
            ranked_by_modality[modality]
            if has_modality_rankings and modality
            else ranked
        )[:top_k]
        if evidence_id in candidates:
            hits += 1
            reciprocal_ranks.append(1 / (candidates.index(evidence_id) + 1))
    return _ratio(hits, len(gold)), max(reciprocal_ranks, default=0.0)


def evaluate_records(records: list[dict], top_k: int = 5) -> dict:
    """计算总体及 Legacy/Multi-Agent RAG 模式分组指标。"""
    return {
        "overall": _summarize(records, top_k),
        "by_mode": {
            mode: _summarize([item for item in records if item.get("mode") == mode], top_k)
            for mode in MODES
        },
    }


def _summarize(records: list[dict], top_k: int) -> dict:
    retrieval_recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    evidence_recalls: list[float] = []
    f1s: list[float] = []
    support_rates: list[float] = []
    modality_hits: dict[str, list[float]] = defaultdict(list)
    modality_mrrs: dict[str, list[float]] = defaultdict(list)
    routing: list[float] = []
    repeated_rates: list[float] = []
    for row in records:
        groups = _evidence_groups(row)
        retrieval_groups = _retrieval_groups(row)
        selected = set(row.get("returned_evidence_ids", []))
        if retrieval_groups:
            retrieval_candidates = []
            for gold in retrieval_groups:
                retrieval_recall, rank = _retrieval_for_group(row, gold, top_k)
                retrieval_candidates.append((retrieval_recall, rank))
            best_retrieval = max(retrieval_candidates, key=lambda item: (item[0], item[1]))
            retrieval_recalls.append(best_retrieval[0])
            reciprocal_ranks.append(best_retrieval[1])
        if groups:
            candidates = []
            for gold in groups:
                precision = _ratio(len(selected & gold), len(selected))
                recall = _ratio(len(selected & gold), len(gold))
                candidates.append({
                    "precision": precision,
                    "recall": recall,
                })
            best = max(
                candidates,
                key=lambda item: (item["recall"], item["precision"]),
            )
            precision = best["precision"]
            recall = best["recall"]
            precisions.append(precision)
            evidence_recalls.append(recall)
            f1s.append(_ratio(2 * precision * recall, precision + recall))
        claims = row.get("claims", [])
        if claims:
            support_rates.append(_ratio(
                sum(item.get("verification_status", item.get("status")) == "supported" for item in claims),
                len(claims),
            ))
        expected_modalities = set(row.get("expected_modalities", []))
        routed_modalities = row.get("routed_modalities")
        # Routing accuracy evaluates the Router decision, independently from
        # whether later generation/refusal produced any final citations.
        actual_modalities = set(
            routed_modalities
            if isinstance(routed_modalities, list)
            else row.get("predicted_modalities", [])
        )
        if expected_modalities:
            routing.append(float(expected_modalities == actual_modalities))
        expected_modalities = set(row.get("expected_modalities", []))
        expected_by_modality = _expected_ids_by_modality(row)
        for modality in ("text", "figure", "table"):
            expected = expected_by_modality[modality]
            if expected and (not expected_modalities or modality in expected_modalities):
                ranked_by_modality, has_rankings = _ranked_by_modality(row)
                ranked = (
                    ranked_by_modality[modality]
                    if has_rankings
                    else list(row.get("ranked_evidence_ids", []))
                )[:top_k]
                modality_hits[modality].append(
                    _ratio(len(expected & set(ranked)), len(expected))
                )
                modality_mrrs[modality].append(next(
                    (1 / (index + 1) for index, item in enumerate(ranked) if item in expected),
                    0.0,
                ))
        calls = row.get("tool_calls", [])
        if isinstance(calls, list) and calls:
            normalized = [str(item) for item in calls]
            repeated_rates.append(_ratio(len(normalized) - len(set(normalized)), len(normalized)))
    return {
        "count": len(records),
        "retrieval": {"recall_at_k": _average(retrieval_recalls), "mrr": _average(reciprocal_ranks), "k": top_k},
        "evidence": {
            "precision": _average(precisions),
            "recall": _average(evidence_recalls),
            "f1": _average(f1s),
        },
        "generation": {
            "faithfulness": _average(support_rates),
            "claim_support_rate": _average(support_rates),
        },
        "agent": {
            "execution_success_rate": _optional_average([
                float(item.get("execution_success") is True)
                for item in records if item.get("execution_success") is not None
            ]),
            "answer_correct_rate": _optional_average([
                float(item.get("answer_correct") is True)
                for item in records if item.get("answer_correct") is not None
            ]),
            "task_success_rate": _optional_average([
                float(item.get("task_success") is True)
                for item in records if item.get("task_success") is not None
            ]),
            "average_steps": _average([float(item.get("steps", 0)) for item in records]),
            "average_tool_calls": _average([float(len(item.get("tool_calls", []))) for item in records]),
            "repeated_tool_call_rate": _average(repeated_rates),
        },
        "multimodal": {
            "text_retrieval_recall_at_k": _average(modality_hits["text"]),
            "figure_retrieval_recall_at_k": _average(modality_hits["figure"]),
            "table_retrieval_recall_at_k": _average(modality_hits["table"]),
            "text_retrieval_mrr": _average(modality_mrrs["text"]),
            "figure_retrieval_mrr": _average(modality_mrrs["figure"]),
            "table_retrieval_mrr": _average(modality_mrrs["table"]),
            "modality_routing_accuracy": _average(routing),
        },
        "engineering": {
            "average_latency_ms": _average([float(item.get("latency_ms", 0)) for item in records]),
            "average_token_usage": _average([float(item.get("token_usage", 0)) for item in records]),
            "average_qwen_vl_calls": _average([float(item.get("qwen_vl_calls", 0)) for item in records]),
            "total_qwen_vl_calls": sum(int(item.get("qwen_vl_calls", 0)) for item in records),
            "average_visual_judge_qwen_vl_calls": _average([
                float(item.get("visual_judge_qwen_vl_calls", 0)) for item in records
            ]),
            "total_visual_judge_qwen_vl_calls": sum(
                int(item.get("visual_judge_qwen_vl_calls", 0)) for item in records
            ),
            "total_qwen_vl_calls_including_judge": sum(
                int(item.get("total_qwen_vl_calls", item.get("qwen_vl_calls", 0)))
                for item in records
            ),
        },
    }
