"""可序列化的 Agent 执行轨迹。"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TraceStep:
    step: int
    subtask_id: str
    tool: str
    arguments: dict[str, Any]
    result_ids: list[str]
    latency_ms: float
    cached: bool = False
    retrieval: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrace:
    run_id: str
    query: str
    route: str
    router: dict[str, Any]
    plan: dict[str, Any] = field(default_factory=dict)
    rewrites: list[dict[str, Any]] = field(default_factory=list)
    steps: list[TraceStep] = field(default_factory=list)
    final_evidence_ids: list[str] = field(default_factory=list)
    total_steps: int = 0
    total_latency_ms: float = 0.0
    token_usage: int = 0
    qwen_vl_calls: int = 0
    visual_answer_check: dict[str, Any] = field(default_factory=dict)
    sufficiency: dict[str, Any] = field(default_factory=dict)
    evidence_memory: list[dict[str, Any]] = field(default_factory=list)
    retrieval: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
