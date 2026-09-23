from types import SimpleNamespace

from app.services.reading import ReadingService


def _service() -> ReadingService:
    return ReadingService(SimpleNamespace(
        function_call_circuit_failure_threshold=2,
        function_call_circuit_cooldown_seconds=30.0,
    ), None)


def test_function_calling_circuit_opens_only_after_threshold(monkeypatch) -> None:
    service = _service()
    clock = {"now": 100.0}
    monkeypatch.setattr("app.services.reading.time.monotonic", lambda: clock["now"])
    service._record_tool_call_failure()
    assert service._tool_call_circuit_is_open() is False
    service._record_tool_call_failure()
    assert service._tool_call_circuit_is_open() is True
    assert service.tool_calling_available is False
    clock["now"] = 131.0
    assert service._tool_call_circuit_is_open() is False
    assert service.tool_calling_available is None


def test_function_calling_success_resets_circuit() -> None:
    service = _service()
    service._record_tool_call_failure()
    service._record_tool_call_success()
    assert service._tool_call_failures == 0
    assert service.tool_calling_available is True
