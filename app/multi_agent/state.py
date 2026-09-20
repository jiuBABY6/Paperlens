"""LangGraph 运行状态与可序列化辅助函数。"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from app.domain import EvidenceObject


def merge_evidence(*collections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 evidence_id 稳定去重，后出现的完整字段覆盖旧字段。"""
    ordered: dict[str, dict[str, Any]] = {}
    for collection in collections:
        for value in collection or []:
            evidence_id = str(value.get("evidence_id", value.get("id", "")))
            if not evidence_id:
                continue
            ordered[evidence_id] = {**ordered.get(evidence_id, {}), **value}
    return list(ordered.values())


def merge_evidence_reducer(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """LangGraph reducer：保持二元签名，同时复用统一证据去重逻辑。"""
    return merge_evidence(left, right)


class MultiAgentState(TypedDict, total=False):
    run_id: str
    paper_id: str
    question: str
    requested_strategy: str | None
    route: dict[str, Any]
    plan: dict[str, Any]
    pending_tasks: list[dict[str, Any]]
    completed_tasks: list[dict[str, Any]]
    agent_results: Annotated[dict[str, dict[str, Any]], lambda left, right: {**left, **right}]
    evidence: Annotated[list[dict[str, Any]], merge_evidence_reducer]
    evidence_ids: list[str]
    critique: dict[str, Any]
    retry_targets: list[str]
    retry_count: int
    answer: dict[str, Any]
    errors: Annotated[list[dict[str, Any]], operator.add]
    trace_steps: Annotated[list[dict[str, Any]], operator.add]
    node_traces: Annotated[list[dict[str, Any]], operator.add]
    total_steps: Annotated[int, operator.add]
    token_usage: Annotated[int, operator.add]
    qwen_vl_calls: Annotated[int, operator.add]
    model_calls: Annotated[int, operator.add]
    started_at: float
    deadline_at: float
    execution_mode: str
    legacy_result: dict[str, Any]


def evidence_to_state(item: EvidenceObject, score: float | None = None) -> dict[str, Any]:
    """把领域 Evidence 转换为 Checkpoint 可序列化字典。"""
    return item.to_payload(score=score)


def evidence_from_state(value: dict[str, Any]) -> EvidenceObject:
    """从 LangGraph State 恢复统一 EvidenceObject。"""
    bbox = value.get("bbox")
    return EvidenceObject(
        evidence_id=str(value.get("evidence_id", value.get("id", ""))),
        type=str(value.get("type", "text")),
        page=int(value.get("page", 0) or 0),
        bbox=tuple(bbox) if bbox else None,
        content=value.get("content", value.get("text")),
        section=value.get("section"),
        metadata=dict(value.get("metadata", {})),
    )
