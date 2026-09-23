"""Text Specialist Agent。"""

from __future__ import annotations

import time
from typing import Any

from app.multi_agent.agents.base import BaseSpecialistAgent
from app.multi_agent.state import evidence_to_state


class TextResearchAgent(BaseSpecialistAgent):
    name = "text_agent"
    modality = "text"

    def run(self, task: dict[str, Any], question: str, attempt: int = 0) -> dict[str, Any]:
        started = time.perf_counter()
        function_result, fallback = self.try_function_calling(task, question, started)
        if function_result:
            return function_result
        query = str(task.get("query") or task.get("instruction") or question)
        tool_started = time.perf_counter()
        try:
            results = self.tools.search_text(query=query, top_k=10)
            evidence = [evidence_to_state(item, score) for item, score in results]
            step = self.tool_step(
                task,
                "search_text",
                {"query": query, "top_k": 10, "attempt": attempt},
                [item.evidence_id for item, _score in results],
                tool_started,
                retrieval=self.tools.last_search_trace,
            )
            result = self.result(
                task,
                status="success" if evidence else "partial",
                evidence=evidence,
                observations=[item.get("content", "") for item in evidence if item.get("content")],
                missing_information=[] if evidence else ["没有检索到相关文本证据。"],
                confidence=0.85 if evidence else 0.0,
                retryable=not evidence,
                tool_calls=[step],
                started=started,
            )
            if fallback:
                result["tool_calls"].insert(0, fallback)
            return result
        except Exception as error:
            return self.failed_result(task, error, started)
