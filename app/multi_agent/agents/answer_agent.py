"""Answer Synthesis Agent：只使用 Critic 审核后的共享证据生成答案。"""

from __future__ import annotations

from typing import Any

from app.multi_agent.state import evidence_from_state


class AnswerSynthesisAgent:
    name = "answer_agent"

    def __init__(self, reading) -> None:
        self.reading = reading

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        evidence = [evidence_from_state(item) for item in state.get("evidence", [])]
        usable = [
            item for item in evidence
            if item.type != "figure"
            or item.metadata.get("query_analysis", {}).get("answerable") is True
        ]
        result = self.reading.answer_from_evidence(
            state["question"],
            usable,
            required_evidence_types=state.get("route", {}).get(
                "required_modalities", ["text"]
            ),
        )
        critique = state.get("critique", {})
        if critique.get("decision") == "partial":
            result["status"] = "partial"
            result["insufficient_evidence"] = list(dict.fromkeys([
                *result.get("insufficient_evidence", []),
                *critique.get("reasons", []),
            ]))
        return result

