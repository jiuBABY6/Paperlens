"""Evidence Critic：在答案生成前审核证据覆盖、冲突与返工条件。"""

from __future__ import annotations

import re
from typing import Any


class EvidenceCriticAgent:
    name = "evidence_critic"

    def __init__(self, settings) -> None:
        self.settings = settings

    @staticmethod
    def _usable_evidence_types(evidence: list[dict[str, Any]]) -> set[str]:
        usable: set[str] = set()
        for item in evidence:
            modality = item.get("type")
            metadata = item.get("metadata", {})
            if modality == "figure":
                analysis = metadata.get("query_analysis", {}) if isinstance(metadata, dict) else {}
                if isinstance(analysis, dict) and analysis.get("answerable") is True:
                    usable.add("figure")
            elif modality == "table":
                markdown = metadata.get("markdown", "") if isinstance(metadata, dict) else ""
                visual = metadata.get("table_visual_analysis", {}) if isinstance(metadata, dict) else {}
                if (
                    str(markdown or item.get("content", "")).strip()
                    or isinstance(visual, dict) and visual.get("answerable") is True
                ):
                    usable.add("table")
            elif modality == "text" and str(item.get("content", "")).strip():
                usable.add("text")
        return usable

    @staticmethod
    def _reported_conflicts(agent_results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []
        metric_values: dict[str, tuple[str, str]] = {}
        for result in agent_results.values():
            for conflict in result.get("conflicts", []):
                conflicts.append({
                    "type": "reported_conflict",
                    "detail": str(conflict)[:500],
                    "task_id": result.get("task_id"),
                })
            for observation in result.get("observations", []):
                if not isinstance(observation, dict):
                    continue
                metric = str(observation.get("metric", "")).strip().lower()
                value = str(observation.get("value", "")).strip()
                if not metric or not re.search(r"[-+]?\d", value):
                    continue
                previous = metric_values.get(metric)
                if previous and previous[0] != value:
                    conflicts.append({
                        "type": "numeric_conflict",
                        "metric": metric,
                        "values": [previous[0], value],
                        "task_ids": [previous[1], str(result.get("task_id", ""))],
                    })
                else:
                    metric_values[metric] = (value, str(result.get("task_id", "")))
        return conflicts

    def review(self, state: dict[str, Any]) -> dict[str, Any]:
        required = list(dict.fromkeys(state.get("route", {}).get("required_modalities", ["text"])))
        evidence = state.get("evidence", [])
        results = state.get("agent_results", {})
        usable = self._usable_evidence_types(evidence)
        missing = [item for item in required if item not in usable]
        conflicts = self._reported_conflicts(results)

        retry_candidates: list[str] = []
        for task in state.get("plan", {}).get("sub_tasks", []):
            result = results.get(task.get("task_id"), {})
            if (
                task.get("modality") in missing
                and result.get("retryable") is True
                and result.get("status") in {"partial", "failed", "unavailable"}
            ):
                retry_candidates.append(str(task["task_id"]))

        retry_count = int(state.get("retry_count", 0))
        if (missing or conflicts) and retry_candidates and retry_count < self.settings.multi_agent_max_retries:
            decision = "retry"
            retry_targets = retry_candidates[:1]
        elif not evidence:
            decision = "refuse"
            retry_targets = []
        elif missing or conflicts:
            decision = "partial"
            retry_targets = []
        else:
            decision = "approved"
            retry_targets = []

        reasons = [f"缺少可用的 {item} Evidence。" for item in missing]
        reasons.extend(
            f"检测到 {item['type']}：{item.get('metric', item.get('detail', '证据冲突'))}。"
            for item in conflicts
        )
        return {
            "decision": decision,
            "required_modalities": required,
            "usable_modalities": sorted(usable),
            "missing_modalities": missing,
            "conflicts": conflicts,
            "retry_targets": retry_targets,
            "reasons": reasons,
        }
