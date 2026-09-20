"""构建兼容旧前端和 Evaluation 的多智能体 Trace。"""

from __future__ import annotations

from typing import Any


def legacy_compatible_trace(
    trace: dict[str, Any],
    *,
    orchestrator: str,
    node_traces: list[dict[str, Any]] | None = None,
    retry_count: int = 0,
    recovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在不破坏旧 Trace 字段的前提下增加编排层信息。"""
    return {
        **trace,
        "orchestrator": orchestrator,
        "node_traces": list(node_traces or []),
        "agent_count": len({
            item.get("agent") for item in (node_traces or []) if item.get("agent")
        }),
        "retry_count": int(retry_count),
        "recovery": dict(recovery or {}),
    }
