"""LLMOps observability: bounded labels, structured logs, metrics and tracing."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import re
import time
import uuid
from typing import Any, Iterator

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


request_id_var: ContextVar[str] = ContextVar("paperlens_request_id", default="")
run_id_var: ContextVar[str] = ContextVar("paperlens_run_id", default="")
conversation_id_var: ContextVar[str] = ContextVar("paperlens_conversation_id", default="")
paper_id_var: ContextVar[str] = ContextVar("paperlens_paper_id", default="")

_SECRET_PATTERN = re.compile(r"(api[_-]?key|authorization|token|secret|password)", re.I)
_CONTENT_PATTERN = re.compile(r"(prompt|content|paper_text|evidence_text|messages)", re.I)


def _safe_value(key: str, value: Any) -> Any:
    """Prevent secrets and full paper/model content from entering logs or spans."""
    if _SECRET_PATTERN.search(key):
        return "[REDACTED]"
    if _CONTENT_PATTERN.search(key):
        if isinstance(value, str):
            return f"[OMITTED:{len(value)} chars]"
        return "[OMITTED]"
    if isinstance(value, dict):
        return {str(k): _safe_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(key, item) for item in value[:20]]
    if isinstance(value, str):
        return value[:500]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:500]


def safe_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _safe_value(str(key), value) for key, value in fields.items()}


def span_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """OpenTelemetry attributes accept scalars or homogeneous scalar sequences."""
    safe = safe_fields(fields)
    return {
        key: value if isinstance(value, (str, bool, int, float)) else json.dumps(
            value, ensure_ascii=False, separators=(",", ":")
        )[:1000]
        for key, value in safe.items() if value is not None
    }


class JsonFormatter(logging.Formatter):
    """Small dependency-free JSON formatter with request/run correlation fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
            "run_id": run_id_var.get(),
            "conversation_id": conversation_id_var.get(),
            "paper_id": paper_id_var.get(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(safe_fields(fields))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)[:2000]
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(*, enabled: bool, level: str = "INFO") -> None:
    if not enabled:
        return
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = next(
        (item for item in root.handlers if getattr(item, "_paperlens_json", False)),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        handler._paperlens_json = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    handler.setFormatter(JsonFormatter())


@dataclass
class ContextTokens:
    request_id: Any = None
    run_id: Any = None
    conversation_id: Any = None
    paper_id: Any = None


def bind_context(
    *, request_id: str | None = None, run_id: str | None = None,
    conversation_id: str | None = None, paper_id: str | None = None,
) -> ContextTokens:
    return ContextTokens(
        request_id=request_id_var.set(request_id) if request_id is not None else None,
        run_id=run_id_var.set(run_id) if run_id is not None else None,
        conversation_id=(
            conversation_id_var.set(conversation_id)
            if conversation_id is not None else None
        ),
        paper_id=paper_id_var.set(paper_id) if paper_id is not None else None,
    )


def reset_context(tokens: ContextTokens) -> None:
    for variable, token in (
        (request_id_var, tokens.request_id),
        (run_id_var, tokens.run_id),
        (conversation_id_var, tokens.conversation_id),
        (paper_id_var, tokens.paper_id),
    ):
        if token is not None:
            variable.reset(token)


class LLMOps:
    """Application-owned metrics registry; labels are deliberately low-cardinality."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.http_requests = Counter(
            "paperlens_http_requests_total", "HTTP requests", ["method", "route", "status"],
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "paperlens_http_request_duration_seconds", "HTTP request duration",
            ["method", "route"], registry=self.registry,
        )
        self.runs = Counter(
            "paperlens_runs_total", "RAG runs", ["route", "status"], registry=self.registry,
        )
        self.run_duration = Histogram(
            "paperlens_run_duration_seconds", "RAG run duration", ["route"],
            registry=self.registry,
        )
        self.active_runs = Gauge(
            "paperlens_active_runs", "Currently executing conversation runs",
            registry=self.registry,
        )
        self.events = Counter(
            "paperlens_events_total", "Workflow and SSE events", ["event"],
            registry=self.registry,
        )
        self.node_duration = Histogram(
            "paperlens_node_duration_seconds", "LangGraph node duration", ["node", "status"],
            registry=self.registry,
        )
        self.tool_calls = Counter(
            "paperlens_tool_calls_total", "Tool calls", ["agent", "tool", "status", "protocol"],
            registry=self.registry,
        )
        self.tool_duration = Histogram(
            "paperlens_tool_duration_seconds", "Tool duration", ["tool", "status"],
            registry=self.registry,
        )
        self.function_calls = Counter(
            "paperlens_function_calls_total", "Function calling outcomes", ["agent", "status"],
            registry=self.registry,
        )
        self.function_fallbacks = Counter(
            "paperlens_function_fallback_total", "Function calling fixed fallbacks", ["agent", "reason"],
            registry=self.registry,
        )
        self.function_rejections = Counter(
            "paperlens_function_rejections_total", "Rejected model tool calls", ["agent", "reason"],
            registry=self.registry,
        )
        self.function_circuit_state = Gauge(
            "paperlens_function_circuit_state", "Function calling circuit state (1=open)",
            registry=self.registry,
        )
        self.function_circuit_opened = Counter(
            "paperlens_function_circuit_open_total", "Function calling circuit openings",
            registry=self.registry,
        )
        self.model_requests = Counter(
            "paperlens_model_requests_total", "Model requests", ["provider", "model", "status", "operation"],
            registry=self.registry,
        )
        self.model_duration = Histogram(
            "paperlens_model_duration_seconds", "Model request duration", ["provider", "model", "operation"],
            registry=self.registry,
        )
        self.tokens = Counter(
            "paperlens_tokens_total", "Reported model tokens", ["provider", "model", "operation"],
            registry=self.registry,
        )
        self.answer_outcomes = Counter(
            "paperlens_answer_outcomes_total", "Answer quality signals",
            ["answerable", "evidence_sufficient", "status"], registry=self.registry,
        )
        self.memory_operations = Counter(
            "paperlens_memory_operations_total", "Long-term memory operations",
            ["operation", "status"], registry=self.registry,
        )
        self.memory_items = Gauge(
            "paperlens_memory_items", "Current long-term memory items", ["status"],
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)

    def record_event(self, event: str, data: dict[str, Any] | None = None) -> None:
        data = data or {}
        self.events.labels(event=_label(event)).inc()
        latency = _milliseconds(data.get("latency_ms"))
        if event == "langgraph.node.completed" and latency is not None:
            self.node_duration.labels(
                node=_label(data.get("node", "unknown")),
                status=_label(data.get("status", "unknown")),
            ).observe(latency)
        if event in {"tool.completed", "tool.failed"}:
            status = _label(data.get("status", "failed" if event.endswith("failed") else "success"))
            self.tool_calls.labels(
                agent=_label(data.get("agent", "unknown")),
                tool=_label(data.get("tool", "unknown")),
                status=status,
                protocol=_label(data.get("protocol", "unknown")),
            ).inc()
            if latency is not None:
                self.tool_duration.labels(
                    tool=_label(data.get("tool", "unknown")), status=status
                ).observe(latency)
        if event == "function_call.rejected":
            self.function_rejections.labels(
                agent=_label(data.get("agent", "unknown")),
                reason=_reason(data.get("error", "invalid_call")),
            ).inc()
        trace.get_current_span().add_event(event, span_fields(data))

    def record_model(
        self, *, provider: str, model: str, operation: str,
        status: str, duration_seconds: float, tokens: int = 0,
    ) -> None:
        labels = {
            "provider": _label(provider), "model": _label(model),
            "status": _label(status), "operation": _label(operation),
        }
        self.model_requests.labels(**labels).inc()
        self.model_duration.labels(
            provider=labels["provider"], model=labels["model"], operation=labels["operation"]
        ).observe(max(0.0, duration_seconds))
        if tokens > 0:
            self.tokens.labels(
                provider=labels["provider"], model=labels["model"], operation=labels["operation"]
            ).inc(tokens)


def _label(value: Any) -> str:
    value = str(value or "unknown").strip().lower()
    return re.sub(r"[^a-z0-9_.:-]+", "_", value)[:80] or "unknown"


def _reason(value: Any) -> str:
    return _label(str(value).split(":", 1)[0])


def _milliseconds(value: Any) -> float | None:
    try:
        return max(0.0, float(value)) / 1000.0
    except (TypeError, ValueError):
        return None


llmops = LLMOps()
_tracing_configured = False


def configure_tracing(*, enabled: bool, service_name: str, endpoint: str = "") -> None:
    global _tracing_configured
    if not enabled or _tracing_configured:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _tracing_configured = True


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    tracer = trace.get_tracer("paperlens")
    safe = span_fields(attributes)
    with tracer.start_as_current_span(name, attributes=safe) as current:
        try:
            yield current
            # A caller may catch an exception inside the span and mark it before
            # leaving the context. Do not overwrite that explicit ERROR state.
            if current.status.status_code is StatusCode.UNSET:
                current.set_status(Status(StatusCode.OK))
        except Exception as error:
            mark_span_error(current, error)
            raise


def mark_span_error(current: Any, error: BaseException) -> None:
    """Mark a handled failure without leaking exception text into telemetry."""
    current.record_exception(error)
    current.set_attribute("error.type", type(error).__name__)
    current.set_status(Status(StatusCode.ERROR, type(error).__name__))


def record_http_request(
    *, method: str, route: str, status: int | str, duration_seconds: float
) -> None:
    """Record one request after Starlette has resolved its route template.

    Recording after ``call_next`` is intentional: before routing, ``request.url.path``
    contains paper/conversation UUIDs and would create unbounded Prometheus labels.
    """
    llmops.http_requests.labels(
        method=_label(method), route=_label(route), status=_label(status)
    ).inc()
    llmops.http_duration.labels(
        method=_label(method), route=_label(route)
    ).observe(max(0.0, duration_seconds))


def new_request_id(value: str | None = None) -> str:
    candidate = str(value or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_.:-]{8,128}", candidate):
        return candidate
    return f"req-{uuid.uuid4()}"
