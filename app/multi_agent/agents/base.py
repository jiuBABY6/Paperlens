"""Specialist Agent 公共契约。"""

from __future__ import annotations

import time
from typing import Any

from app.multi_agent.recovery import classify_error


VALID_AGENT_STATUSES = {
    "success", "partial", "failed", "unavailable", "not_applicable"
}


class BaseSpecialistAgent:
    name = "base_agent"
    modality = "text"

    def __init__(self, settings, tools) -> None:
        self.settings = settings
        self.tools = tools

    def run(self, task: dict[str, Any], question: str, attempt: int = 0) -> dict[str, Any]:
        raise NotImplementedError

    def result(
        self,
        task: dict[str, Any],
        *,
        status: str,
        evidence: list[dict[str, Any]] | None = None,
        observations: list[Any] | None = None,
        inferences: list[Any] | None = None,
        candidate_claims: list[dict[str, Any]] | None = None,
        missing_information: list[str] | None = None,
        confidence: float = 0.0,
        retryable: bool = False,
        tool_calls: list[dict[str, Any]] | None = None,
        started: float | None = None,
        token_usage: int = 0,
        model_calls: int = 0,
        qwen_vl_calls: int = 0,
        errors: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if status not in VALID_AGENT_STATUSES:
            raise ValueError(f"未知 Agent 状态：{status}")
        evidence = list(evidence or [])
        return {
            "agent": self.name,
            "modality": self.modality,
            "task_id": task["task_id"],
            "status": status,
            "observations": list(observations or []),
            "inferences": list(inferences or []),
            "candidate_claims": list(candidate_claims or []),
            "evidence": evidence,
            "evidence_ids": [
                item.get("evidence_id", item.get("id")) for item in evidence
                if item.get("evidence_id", item.get("id"))
            ],
            "missing_information": list(missing_information or []),
            "confidence": max(0.0, min(float(confidence), 1.0)),
            "retryable": bool(retryable),
            "tool_calls": list(tool_calls or []),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1) if started else 0.0,
            "token_usage": int(token_usage),
            "model_calls": int(model_calls),
            "qwen_vl_calls": int(qwen_vl_calls),
            "errors": list(errors or []),
        }

    def failed_result(
        self,
        task: dict[str, Any],
        error: Exception,
        started: float,
    ) -> dict[str, Any]:
        detail = classify_error(error)
        return self.result(
            task,
            status="failed",
            missing_information=[f"{self.name} 执行失败。"],
            retryable=bool(detail["retryable"]),
            started=started,
            errors=[detail],
        )

    def tool_step(
        self,
        task: dict[str, Any],
        tool: str,
        arguments: dict[str, Any],
        result_ids: list[str],
        started: float,
        *,
        cached: bool = False,
        retrieval: dict[str, Any] | None = None,
        status: str = "success",
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "subtask_id": task["task_id"],
            "task_id": task["task_id"],
            "agent": self.name,
            "node": self.name,
            "tool": tool,
            "arguments": arguments,
            "result_ids": result_ids,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "cached": bool(cached),
            "retrieval": dict(retrieval or {}),
            "status": status,
            "error": error,
        }
