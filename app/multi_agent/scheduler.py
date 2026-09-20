"""Specialist Agent 的有界调度器。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
import time
from typing import Any


class SpecialistScheduler:
    def __init__(self, *, parallel_enabled: bool = False) -> None:
        self.parallel_enabled = parallel_enabled

    def execute(
        self,
        tasks: list[dict[str, Any]],
        agents: dict[str, Any],
        question: str,
        *,
        attempt: int = 0,
        timeout_seconds: float | None = None,
    ) -> list[dict[str, Any]]:
        selected = [task for task in tasks if task.get("agent") in agents]
        if not self.parallel_enabled or len(selected) <= 1:
            output = []
            for task in selected:
                started = time.monotonic()
                result = agents[task["agent"]].run(task, question, attempt)
                elapsed = time.monotonic() - started
                if timeout_seconds is not None and elapsed > timeout_seconds:
                    result = {
                        **result,
                        "status": "failed",
                        "retryable": True,
                        "errors": [*result.get("errors", []), {
                            "type": "TimeoutError",
                            "category": "timeout",
                            "message": f"节点执行超过 {timeout_seconds:.3f}s。",
                            "retryable": True,
                        }],
                    }
                output.append(result)
            return output

        output: dict[str, dict[str, Any]] = {}
        pool = ThreadPoolExecutor(max_workers=min(3, len(selected)))
        futures = {
            pool.submit(agents[task["agent"]].run, task, question, attempt): task
            for task in selected
        }
        done, pending = wait(futures, timeout=timeout_seconds)
        for future in done:
            task = futures[future]
            output[task["task_id"]] = future.result()
        for future in pending:
            task = futures[future]
            future.cancel()
            output[task["task_id"]] = {
                "agent": task["agent"],
                "modality": task.get("modality", ""),
                "task_id": task["task_id"],
                "status": "failed",
                "observations": [],
                "inferences": [],
                "candidate_claims": [],
                "evidence": [],
                "evidence_ids": [],
                "missing_information": ["并行节点执行超时。"],
                "confidence": 0.0,
                "retryable": True,
                "tool_calls": [],
                "latency_ms": round((timeout_seconds or 0) * 1000, 1),
                "token_usage": 0,
                "model_calls": 0,
                "qwen_vl_calls": 0,
                "errors": [{
                    "type": "TimeoutError", "category": "timeout",
                    "message": "并行节点执行超时。", "retryable": True,
                }],
            }
        pool.shutdown(wait=not pending, cancel_futures=True)
        return [output[task["task_id"]] for task in selected]
