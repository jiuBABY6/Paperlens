import json
from types import SimpleNamespace

import pytest

from app.domain import EvidenceObject
from app.multi_agent.function_calling import FunctionCallingError, FunctionCallingSpecialistExecutor
from app.multi_agent.tool_registry import schemas_for, validate_arguments


class _Reading:
    def __init__(self, messages):
        self.messages = iter(messages)

    def tool_completion(self, _messages, _tools):
        return next(self.messages)


class _Tools:
    last_search_trace = {"effective_strategy": "hybrid-rerank"}

    def search_text(self, query, top_k=10):
        return [(EvidenceObject("s1", "text", 1, None, f"evidence:{query}"), 0.9)]

    def get_evidence(self, _evidence_id):
        return None


def _settings():
    return SimpleNamespace(
        function_call_max_steps=4,
        function_call_max_model_rounds=3,
        multi_agent_max_qwen_vl_calls=2,
    )


def test_registry_rejects_cross_agent_and_paper_id() -> None:
    with pytest.raises(PermissionError):
        validate_arguments("text_agent", "search_figures", {"query": "x"})
    with pytest.raises(ValueError, match="additional"):
        validate_arguments("text_agent", "search_text", {"query": "method", "paper_id": "other"})
    with pytest.raises(ValueError, match="above_maximum"):
        validate_arguments("text_agent", "search_text", {"query": "method", "top_k": 999})
    schema = schemas_for("text_agent", strict=True)[0]["function"]
    assert schema["strict"] is True
    assert set(schema["parameters"]["required"]) == {"query", "top_k"}


def test_native_tool_call_records_protocol_and_evidence() -> None:
    reading = _Reading([
        {"content": "", "tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": "search_text", "arguments": json.dumps({"query": "method", "top_k": 3})},
        }]},
        {"content": "done"},
    ])
    executor = FunctionCallingSpecialistExecutor(reading, _Tools(), "text_agent", _settings())
    result = executor.run({"task_id": "t1", "instruction": "find method"}, "What is the method?")
    assert result["evidence"][0]["evidence_id"] == "s1"
    assert result["tool_calls"][0]["protocol"] == "function_calling"
    assert result["model_calls"] == 2


def test_function_calling_emits_live_tool_events_and_bounded_envelope() -> None:
    class CapturingReading(_Reading):
        def __init__(self, messages):
            super().__init__(messages)
            self.seen_messages = []

        def tool_completion(self, messages, tools):
            self.seen_messages = list(messages)
            return super().tool_completion(messages, tools)

    reading = CapturingReading([
        {"tool_calls": [{
            "id": "call-live", "function": {
                "name": "search_text",
                "arguments": '{"query":"method","top_k":3}',
            },
        }]},
        {"content": "done"},
    ])
    events = []
    executor = FunctionCallingSpecialistExecutor(
        reading, _Tools(), "text_agent", _settings(),
        event_callback=lambda event, data: events.append((event, data)),
    )
    executor.run({"task_id": "t1", "instruction": "find method"}, "method?")
    names = [event for event, _data in events]
    assert "tool.started" in names and "tool.completed" in names
    tool_message = next(item for item in reading.seen_messages if item["role"] == "tool")
    envelope = json.loads(tool_message["content"])
    assert envelope["ok"] is True
    assert envelope["tool"] == "search_text"
    assert envelope["evidence_ids"] == ["s1"]


def test_duplicate_tool_call_is_stopped() -> None:
    call = {"id": "call", "type": "function", "function": {
        "name": "search_text", "arguments": '{"query":"method"}'
    }}
    executor = FunctionCallingSpecialistExecutor(
        _Reading([{"tool_calls": [call]}, {"tool_calls": [{**call, "id": "call-2"}]}]),
        _Tools(), "text_agent", _settings(),
    )
    with pytest.raises(FunctionCallingError, match="duplicate"):
        executor.run({"task_id": "t1", "instruction": "find"}, "question")


def test_figure_analysis_is_attached_to_retrieved_evidence() -> None:
    class FigureTools(_Tools):
        def search_figures(self, query, top_k=5):
            return [(EvidenceObject("fig-2", "figure", 2, None, "Figure 2", metadata={}), 1.0)]

        def analyze_figure_for_query(self, figure_id, question):
            return {"answerable": True, "visual_observations": ["orange box"]}

        def get_evidence(self, evidence_id):
            return EvidenceObject(evidence_id, "figure", 2, None, "Figure 2", metadata={})

    messages = [
        {"tool_calls": [{"id": "s", "function": {"name": "search_figures", "arguments": '{"query":"Figure 2"}'}}]},
        {"tool_calls": [{"id": "a", "function": {"name": "analyze_figure_for_query", "arguments": '{"figure_id":"fig-2","question":"color?"}'}}]},
        {"content": "done"},
    ]
    executor = FunctionCallingSpecialistExecutor(_Reading(messages), FigureTools(), "figure_agent", _settings())
    result = executor.run({"task_id": "t1", "instruction": "inspect", "max_qwen_calls": 1}, "color?")
    assert result["evidence"][0]["metadata"]["query_analysis"]["answerable"] is True


def test_figure_model_cannot_stop_after_metadata_read() -> None:
    class FigureTools(_Tools):
        def search_figures(self, query, top_k=5):
            return [(EvidenceObject("fig-2", "figure", 2, None, "Figure 2", metadata={}), 1.0)]

        def read_figure(self, figure_id):
            return {"evidence_id": figure_id, "caption": "Figure 2"}

        def get_evidence(self, evidence_id):
            return EvidenceObject(evidence_id, "figure", 2, None, "Figure 2", metadata={})

    messages = [
        {"tool_calls": [{"id": "s", "function": {"name": "search_figures", "arguments": '{"query":"Figure 2"}'}}]},
        {"tool_calls": [{"id": "r", "function": {"name": "read_figure", "arguments": '{"figure_id":"fig-2"}'}}]},
        {"content": "done"},
    ]
    executor = FunctionCallingSpecialistExecutor(_Reading(messages), FigureTools(), "figure_agent", _settings())
    with pytest.raises(FunctionCallingError, match="required_visual_analysis"):
        executor.run({"task_id": "t1", "instruction": "inspect", "max_qwen_calls": 1}, "color?")
