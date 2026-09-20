import httpx
import time

from app.multi_agent.recovery import call_with_retry, classify_error
from app.multi_agent.scheduler import SpecialistScheduler


def test_remote_retry_is_bounded_and_exponential() -> None:
    attempts = []
    delays = []

    def operation():
        attempts.append(len(attempts) + 1)
        if len(attempts) < 3:
            raise httpx.ReadTimeout("temporary")
        return "ok"

    assert call_with_retry(
        operation,
        max_retries=2,
        base_delay_seconds=0.25,
        sleep=delays.append,
    ) == "ok"
    assert attempts == [1, 2, 3]
    assert delays == [0.25, 0.5]


def test_non_retryable_error_fails_immediately() -> None:
    attempts = []

    def operation():
        attempts.append(1)
        raise ValueError("bad response")

    try:
        call_with_retry(operation, max_retries=3, base_delay_seconds=0)
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError should not be retried")
    assert len(attempts) == 1
    assert classify_error(ValueError("bad"))["retryable"] is False


def test_http_status_classification_only_retries_transient_codes() -> None:
    request = httpx.Request("POST", "https://example.test")
    limited = httpx.HTTPStatusError(
        "limited", request=request, response=httpx.Response(429, request=request)
    )
    invalid = httpx.HTTPStatusError(
        "invalid", request=request, response=httpx.Response(400, request=request)
    )
    assert classify_error(limited)["retryable"] is True
    assert classify_error(invalid)["retryable"] is False


class SlowAgent:
    def run(self, task, _question, _attempt):
        import time

        time.sleep(0.02)
        return {
            "agent": task["agent"], "task_id": task["task_id"],
            "status": "success", "retryable": False, "errors": [],
        }


def test_scheduler_marks_node_timeout_as_retryable_failure() -> None:
    task = {"task_id": "slow", "agent": "text_agent"}
    result = SpecialistScheduler().execute(
        [task], {"text_agent": SlowAgent()}, "question", timeout_seconds=0.001
    )[0]
    assert result["status"] == "failed"
    assert result["retryable"] is True
    assert result["errors"][-1]["category"] == "timeout"


class TimedAgent:
    def __init__(self, delay: float) -> None:
        self.delay = delay

    def run(self, task, _question, _attempt):
        time.sleep(self.delay)
        return {
            "agent": task["agent"], "modality": task["modality"],
            "task_id": task["task_id"], "status": "success",
            "evidence": [], "evidence_ids": [], "observations": [],
            "inferences": [], "candidate_claims": [],
            "missing_information": [], "confidence": 1.0,
            "retryable": False, "tool_calls": [], "latency_ms": self.delay * 1000,
            "token_usage": 0, "model_calls": 0, "qwen_vl_calls": 0, "errors": [],
        }


def test_parallel_scheduler_preserves_order_and_reduces_latency() -> None:
    tasks = [
        {"task_id": f"task_{index}", "agent": f"agent_{index}", "modality": "text"}
        for index in range(3)
    ]
    agents = {task["agent"]: TimedAgent(0.06) for task in tasks}

    started = time.perf_counter()
    serial = SpecialistScheduler(parallel_enabled=False).execute(
        tasks, agents, "question", timeout_seconds=1
    )
    serial_seconds = time.perf_counter() - started
    started = time.perf_counter()
    parallel = SpecialistScheduler(parallel_enabled=True).execute(
        tasks, agents, "question", timeout_seconds=1
    )
    parallel_seconds = time.perf_counter() - started

    assert [item["task_id"] for item in parallel] == [item["task_id"] for item in serial]
    assert [item["status"] for item in parallel] == [item["status"] for item in serial]
    assert parallel_seconds < serial_seconds * 0.7
