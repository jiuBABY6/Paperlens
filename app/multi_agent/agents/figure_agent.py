"""Figure Specialist Agent。"""

from __future__ import annotations

import time
from typing import Any

from app.multi_agent.agents.base import BaseSpecialistAgent
from app.multi_agent.state import evidence_to_state


class FigureAnalysisAgent(BaseSpecialistAgent):
    name = "figure_agent"
    modality = "figure"

    def run(self, task: dict[str, Any], question: str, attempt: int = 0) -> dict[str, Any]:
        started = time.perf_counter()
        query = str(task.get("query") or task.get("instruction") or question)
        calls_before = (
            self.tools.figure_service.interactive_call_count
            if self.tools.figure_service else 0
        )
        tool_calls: list[dict[str, Any]] = []
        try:
            search_started = time.perf_counter()
            results = self.tools.search_figures(query=query, top_k=5)
            tool_calls.append(self.tool_step(
                task,
                "search_figures",
                {"query": query, "top_k": 5, "attempt": attempt},
                [item.evidence_id for item, _score in results],
                search_started,
                retrieval=self.tools.last_search_trace,
            ))
            observations: list[Any] = []
            missing: list[str] = []
            evidence: list[dict[str, Any]] = []
            answerable = False
            read_limit = min(
                self.settings.agent_figure_read_limit,
                max(0, int(task.get("max_qwen_calls", self.settings.agent_figure_read_limit))),
            )
            for item, score in results[:read_limit]:
                analysis_started = time.perf_counter()
                analysis = self.tools.analyze_figure_for_query(item.evidence_id, question)
                if analysis:
                    item.metadata["query_analysis"] = analysis
                    observations.extend(analysis.get("visual_observations", []))
                    missing.extend(str(value) for value in analysis.get("missing_information", []))
                    answerable = answerable or analysis.get("answerable") is True
                tool_calls.append(self.tool_step(
                    task,
                    "analyze_figure_for_query",
                    {"figure_id": item.evidence_id, "question": question},
                    [item.evidence_id] if analysis else [],
                    analysis_started,
                    cached=bool(analysis and analysis.get("cached")),
                    status="success" if analysis else "unavailable",
                ))
                evidence.append(evidence_to_state(item, score))
            if not results:
                missing.append("没有检索到相关 Figure。")
            elif not answerable:
                missing.append("命中的 Figure 尚未提供足以回答问题的原图观察。")
            calls_after = (
                self.tools.figure_service.interactive_call_count
                if self.tools.figure_service else calls_before
            )
            return self.result(
                task,
                status="success" if answerable else "partial",
                evidence=evidence,
                observations=observations,
                missing_information=list(dict.fromkeys(missing)),
                confidence=0.9 if answerable else (0.3 if evidence else 0.0),
                retryable=not answerable,
                tool_calls=tool_calls,
                started=started,
                model_calls=max(0, calls_after - calls_before),
                qwen_vl_calls=max(0, calls_after - calls_before),
            )
        except Exception as error:
            return self.failed_result(task, error, started)
