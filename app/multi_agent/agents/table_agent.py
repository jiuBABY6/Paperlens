"""Table Specialist Agent。"""

from __future__ import annotations

import time
from typing import Any

from app.multi_agent.agents.base import BaseSpecialistAgent
from app.multi_agent.state import evidence_to_state


class TableAnalysisAgent(BaseSpecialistAgent):
    name = "table_agent"
    modality = "table"

    def run(self, task: dict[str, Any], question: str, attempt: int = 0) -> dict[str, Any]:
        started = time.perf_counter()
        query = str(task.get("query") or task.get("instruction") or question)
        tool_calls: list[dict[str, Any]] = []
        calls_before = (
            self.tools.figure_service.interactive_call_count
            if self.tools.figure_service else 0
        )
        try:
            search_started = time.perf_counter()
            results = self.tools.search_tables(query=query, top_k=5)
            tool_calls.append(self.tool_step(
                task,
                "search_tables",
                {"query": query, "top_k": 5, "attempt": attempt},
                [item.evidence_id for item, _score in results],
                search_started,
                retrieval=self.tools.last_search_trace,
            ))
            evidence: list[dict[str, Any]] = []
            observations: list[Any] = []
            usable = False
            read_limit = max(0, min(2, int(task.get("max_tool_steps", 3)) - 1))
            for item, score in results[:read_limit]:
                read_started = time.perf_counter()
                value = self.tools.read_table(item.evidence_id)
                tool_calls.append(self.tool_step(
                    task,
                    "read_table",
                    {"table_id": item.evidence_id},
                    [item.evidence_id] if value else [],
                    read_started,
                    status="success" if value else "partial",
                ))
                if value:
                    usable = usable or bool(str(value.get("markdown", "")).strip())
                    observations.append({
                        "table_id": item.evidence_id,
                        "columns": value.get("columns", []),
                        "row_count": len(value.get("rows", [])),
                    })
                evidence.append(evidence_to_state(item, score))
            if not usable and results:
                item, _score = results[0]
                fallback_started = time.perf_counter()
                fallback = self.tools.analyze_table_image_with_vlm(
                    item.evidence_id, question
                )
                tool_calls.append(self.tool_step(
                    task,
                    "analyze_table_image_with_vlm",
                    {"table_id": item.evidence_id, "question": question},
                    [item.evidence_id] if fallback.get("answerable") else [],
                    fallback_started,
                    status="success" if fallback.get("answerable") else "unavailable",
                ))
                if fallback.get("available"):
                    item.metadata["table_visual_analysis"] = fallback
                    observations.extend(fallback.get("table_observations", []))
                    usable = fallback.get("answerable") is True
                    if evidence:
                        evidence[0] = evidence_to_state(item, _score)
            calls_after = (
                self.tools.figure_service.interactive_call_count
                if self.tools.figure_service else calls_before
            )
            return self.result(
                task,
                status="success" if usable else "partial",
                evidence=evidence,
                observations=observations,
                missing_information=[] if usable else ["没有找到可读取的结构化 Table 证据。"],
                confidence=0.9 if usable else (0.3 if evidence else 0.0),
                retryable=not usable,
                tool_calls=tool_calls,
                started=started,
                model_calls=max(0, calls_after - calls_before),
                qwen_vl_calls=max(0, calls_after - calls_before),
            )
        except Exception as error:
            return self.failed_result(task, error, started)
