import json
from pathlib import Path
import uuid

from fastapi.testclient import TestClient

import app.main as main
from app.observability import mark_span_error, safe_fields


class _RecordedSpan:
    def __init__(self) -> None:
        self.recorded = []
        self.attributes = {}
        self.status = None

    def record_exception(self, error) -> None:
        self.recorded.append(error)

    def set_attribute(self, key, value) -> None:
        self.attributes[key] = value

    def set_status(self, status) -> None:
        self.status = status


def test_structured_fields_redact_secrets_and_content() -> None:
    value = safe_fields({
        "api_key": "secret", "prompt": "paper body", "count": 2,
        "nested": {"authorization": "Bearer abc"},
    })
    assert value["api_key"] == "[REDACTED]"
    assert value["prompt"].startswith("[OMITTED:")
    assert value["nested"]["authorization"] == "[REDACTED]"
    assert value["count"] == 2


def test_handled_span_failure_is_explicitly_marked_error() -> None:
    current = _RecordedSpan()
    error = RuntimeError("sensitive provider response")
    mark_span_error(current, error)

    assert current.recorded == [error]
    assert current.attributes == {"error.type": "RuntimeError"}
    assert current.status.status_code.name == "ERROR"
    assert current.status.description == "RuntimeError"


def test_request_id_readiness_and_metrics_are_exposed_with_bounded_route_labels() -> None:
    client = TestClient(main.app)
    paper_id = f"missing-{uuid.uuid4()}"
    response = client.get(
        f"/api/papers/{paper_id}", headers={"X-Request-ID": "request-12345678"}
    )
    assert response.status_code == 404
    assert response.headers["X-Request-ID"] == "request-12345678"
    assert client.get("/api/ready").status_code == 200

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "paperlens_http_requests_total" in metrics.text
    assert "paperlens_model_requests_total" in metrics.text
    assert paper_id not in metrics.text
    assert "_api_papers_paper_id_" in metrics.text


def test_grafana_dashboards_use_window_counts_and_zero_safe_stats() -> None:
    root = Path(__file__).parents[1] / "observability" / "grafana" / "dashboards"
    agents = json.loads((root / "paperlens-agents-tools.json").read_text("utf-8"))
    system = json.loads((root / "paperlens-system-overview.json").read_text("utf-8"))
    models = json.loads((root / "paperlens-model-memory.json").read_text("utf-8"))

    agent_queries = {panel["title"]: panel["targets"][0]["expr"] for panel in agents["panels"]}
    system_queries = {panel["title"]: panel["targets"][0]["expr"] for panel in system["panels"]}
    model_queries = {panel["title"]: panel["targets"][0]["expr"] for panel in models["panels"]}

    assert "increase(paperlens_tool_calls_total[5m])" in agent_queries[
        "Tool calls / 5m by status"
    ]
    assert agent_queries["Function fallback / process lifetime"].endswith(
        "or vector(0)"
    )
    assert agent_queries["Rejected tool calls / process lifetime"].endswith(
        "or vector(0)"
    )
    assert system_queries["Runs / 5m"].endswith("or vector(0)")
    assert "increase(paperlens_runs_total[5m])" in system_queries["Run outcomes / 5m"]
    assert "increase(paperlens_model_requests_total[5m])" in model_queries[
        "Model requests / 5m"
    ]
    assert "increase(paperlens_memory_operations_total[5m])" in model_queries[
        "Memory operations / 5m"
    ]
