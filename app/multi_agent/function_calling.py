"""Bounded native Function Calling loop for one specialist and one paper."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from app.multi_agent.state import evidence_to_state
from app.multi_agent.tool_registry import schemas_for, validate_arguments


class FunctionCallingError(RuntimeError):
    pass


class FunctionCallingSpecialistExecutor:
    def __init__(
        self, reading, tools, agent_name: str, settings,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.reading = reading
        self.tools = tools
        self.agent_name = agent_name
        self.settings = settings
        self.event_callback = event_callback

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        if self.event_callback:
            self.event_callback(event, data)

    def run(self, task: dict[str, Any], question: str) -> dict[str, Any]:
        messages = [{
            "role": "system",
            "content": (
                f"You are {self.agent_name}. Use only the supplied current-paper tools. "
                "Do not answer from memory. Stop once sufficient evidence is found."
            ),
        }, {"role": "user", "content": f"Task: {task.get('instruction', '')}\nQuestion: {question}"}]
        calls: list[dict[str, Any]] = []
        evidence: dict[str, dict[str, Any]] = {}
        observations: list[Any] = []
        seen: set[tuple[str, str]] = set()
        qwen_calls = 0
        model_rounds = 0
        max_steps = min(
            int(task.get("max_tool_steps", self.settings.function_call_max_steps)),
            self.settings.function_call_max_steps,
        )
        for _round in range(self.settings.function_call_max_model_rounds):
            model_rounds += 1
            self._emit("function_call.requested", {
                "agent": self.agent_name,
                "task_id": task.get("task_id"),
                "round": model_rounds,
            })
            message = self.reading.tool_completion(
                messages,
                schemas_for(
                    self.agent_name,
                    strict=bool(getattr(self.settings, "function_call_strict", True)),
                ),
            )
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                break
            messages.append({
                "role": "assistant", "content": message.get("content") or "",
                "tool_calls": tool_calls,
            })
            for call in tool_calls:
                if len(calls) >= max_steps:
                    raise FunctionCallingError("function_call_budget_exhausted")
                function = call.get("function", {})
                name = str(function.get("name", ""))
                try:
                    if not call.get("id") or not isinstance(function, dict) or not name:
                        raise ValueError("malformed_tool_call")
                    raw = function.get("arguments", "{}")
                    args = json.loads(raw) if isinstance(raw, str) else raw
                    args = validate_arguments(self.agent_name, name, args)
                except Exception as error:
                    self._emit("function_call.rejected", {
                        "agent": self.agent_name,
                        "task_id": task.get("task_id"),
                        "tool": name,
                        "tool_call_id": call.get("id", ""),
                        "error": str(error)[:300],
                    })
                    raise FunctionCallingError(f"invalid_tool_call:{type(error).__name__}") from error
                signature = (name, json.dumps(args, ensure_ascii=False, sort_keys=True))
                if signature in seen:
                    raise FunctionCallingError("duplicate_tool_call")
                seen.add(signature)
                if name in {"analyze_figure_for_query", "analyze_table_image_with_vlm"}:
                    allowed = int(task.get("max_qwen_calls", self.settings.multi_agent_max_qwen_vl_calls))
                    if qwen_calls >= allowed:
                        raise FunctionCallingError("qwen_vl_budget_exhausted")
                    qwen_calls += 1
                started = time.perf_counter()
                event_data = {
                    "agent": self.agent_name,
                    "task_id": task.get("task_id"),
                    "tool": name,
                    "tool_call_id": call.get("id", ""),
                    "arguments": args,
                }
                self._emit("tool.started", event_data)
                try:
                    result = getattr(self.tools, name)(**args)
                except Exception as error:
                    self._emit("tool.failed", {
                        **event_data,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                        "error": type(error).__name__,
                    })
                    raise
                result_ids = self._collect(result, evidence, observations)
                self._attach_query_analysis(name, args, result, evidence, result_ids)
                record = {
                    "subtask_id": task["task_id"], "task_id": task["task_id"],
                    "agent": self.agent_name, "node": self.agent_name,
                    "tool": name, "arguments": args, "result_ids": result_ids,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "cached": bool(isinstance(result, dict) and result.get("cached")),
                    "retrieval": dict(self.tools.last_search_trace) if name.startswith("search_") else {},
                    "status": "success" if result else "partial", "error": None,
                    "protocol": "function_calling", "tool_call_id": call.get("id", ""),
                }
                calls.append(record)
                self._emit("tool.completed", {
                    **event_data,
                    "result_ids": result_ids,
                    "latency_ms": record["latency_ms"],
                    "status": record["status"],
                    "cached": record["cached"],
                })
                messages.append({
                    "role": "tool", "tool_call_id": call.get("id", ""),
                    "content": json.dumps(
                        self._result_envelope(name, result, result_ids),
                        ensure_ascii=False,
                    ),
                })
        called_tools = {item["tool"] for item in calls}
        if self.agent_name == "figure_agent" and "analyze_figure_for_query" not in called_tools:
            raise FunctionCallingError("required_visual_analysis_missing")
        if self.agent_name == "table_agent" and not (
            {"read_table", "analyze_table_image_with_vlm"} & called_tools
        ):
            raise FunctionCallingError("required_table_read_missing")
        return {
            "evidence": list(evidence.values()), "observations": observations,
            "tool_calls": calls, "model_calls": model_rounds,
            "qwen_vl_calls": qwen_calls,
        }

    def _attach_query_analysis(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        evidence: dict[str, dict[str, Any]],
        result_ids: list[str],
    ) -> None:
        """Preserve query-conditioned visual output on the Evidence consumed by Critic."""
        if not isinstance(result, dict):
            return
        if tool_name == "analyze_figure_for_query":
            evidence_id = str(arguments.get("figure_id", ""))
            metadata_key = "query_analysis"
        elif tool_name == "analyze_table_image_with_vlm":
            evidence_id = str(arguments.get("table_id", ""))
            metadata_key = "table_visual_analysis"
        else:
            return
        if evidence_id not in evidence:
            item = self.tools.get_evidence(evidence_id)
            if item:
                evidence[evidence_id] = evidence_to_state(item)
                result_ids.append(evidence_id)
        if evidence_id in evidence:
            metadata = dict(evidence[evidence_id].get("metadata", {}))
            metadata[metadata_key] = result
            evidence[evidence_id]["metadata"] = metadata

    def _collect(self, result, evidence: dict[str, dict], observations: list[Any]) -> list[str]:
        ids: list[str] = []
        if isinstance(result, list):
            for value in result:
                if isinstance(value, tuple) and len(value) == 2 and hasattr(value[0], "evidence_id"):
                    item, score = value
                    payload = evidence_to_state(item, score)
                    evidence[item.evidence_id] = payload
                    ids.append(item.evidence_id)
        elif isinstance(result, dict):
            evidence_id = result.get("evidence_id") or result.get("id") or result.get("figure_id") or result.get("table_id")
            if evidence_id:
                item = self.tools.get_evidence(str(evidence_id))
                if item:
                    evidence[item.evidence_id] = evidence_to_state(item)
                    ids.append(item.evidence_id)
            observations.append(result)
        return ids

    def _result_envelope(
        self, tool_name: str, result: Any, result_ids: list[str]
    ) -> dict[str, Any]:
        """Return a stable, bounded tool message contract to the provider model."""
        value = self._safe_result(result)
        encoded = json.dumps(value, ensure_ascii=False)
        limit = int(getattr(self.settings, "function_call_tool_result_max_chars", 16000))
        truncated = len(encoded) > limit
        if truncated:
            value = {"preview": encoded[:limit], "truncated": True}
        return {
            "ok": True,
            "tool": tool_name,
            "data": value,
            "evidence_ids": result_ids,
            "error": None,
            "truncated": truncated,
        }

    @staticmethod
    def _safe_result(result) -> Any:
        if isinstance(result, list):
            return [
                {"evidence_id": item.evidence_id, "type": item.type, "page": item.page, "score": score}
                for item, score in result[:10]
            ]
        if isinstance(result, dict):
            return result
        return {"available": bool(result)}
