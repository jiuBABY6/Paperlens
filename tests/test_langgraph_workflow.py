from dataclasses import replace

from app.config import settings
from app.domain import Chunk, Figure, Paper, Sentence
from app.multi_agent.graph import LangGraphExecutor, build_agent_executor
from app.services.retrieval import HybridRetriever


class FakeReading:
    token_usage = 0

    def answer_from_evidence(self, _question, evidence, required_evidence_types=None):
        citations = [item.to_payload() for item in evidence]
        return {
            "answer": "Grounded answer.",
            "answerable": True,
            "claims": [{
                "claim": "Grounded answer.",
                "evidence_ids": [evidence[0].evidence_id],
            }],
            "citations": citations,
            "insufficient_evidence": [],
            "refusal_reason": "",
            "status": "ok",
            "verification": [],
            "claim_coverage": {
                "required_evidence_types": required_evidence_types or ["text"],
                "used_evidence_types": ["text"],
                "missing_evidence_types": [],
                "complete": True,
            },
        }


class ToolCallingReading(FakeReading):
    def __init__(self) -> None:
        self.token_usage = 0
        self.request_count = 0

    def tool_completion(self, messages, _tools):
        self.request_count += 1
        if not any(item.get("role") == "tool" for item in messages):
            return {
                "content": "",
                "tool_calls": [{
                    "id": "search-1",
                    "function": {
                        "name": "search_text",
                        "arguments": '{"query":"novel method","top_k":3}',
                    },
                }],
            }
        return {"content": "done", "tool_calls": []}


def sample_paper() -> Paper:
    chunk = Chunk("paper-p1-b1", "paper", 1, "Methods", "The method is novel.", None)
    sentence = Sentence(
        "paper-p1-b1-s0", "paper", chunk.id, 1, "Methods", chunk.text, None
    )
    chunk.sentence_ids = [sentence.id]
    return Paper("paper", "paper.pdf", "Paper", "", "paper.pdf", "test", [chunk], [], [sentence])


class FakeFigureClient:
    token_usage = 0


class FakeFigureService:
    def __init__(self) -> None:
        self.client = FakeFigureClient()
        self.interactive_call_count = 0

    def analyze_for_query(self, _paper_id, figure, _question):
        self.interactive_call_count += 1
        return {
            "answerable": True,
            "visual_observations": [f"Observed {figure.caption}"],
            "missing_information": [],
            "cached": False,
        }


class RecoveringFigureService(FakeFigureService):
    def analyze_for_query(self, _paper_id, figure, _question):
        self.interactive_call_count += 1
        answerable = self.interactive_call_count > 1
        return {
            "answerable": answerable,
            "visual_observations": [f"Observed {figure.caption}"] if answerable else [],
            "missing_information": [] if answerable else ["First visual read failed."],
            "cached": False,
        }


def cross_modal_paper() -> Paper:
    paper = sample_paper()
    paper.figures = [
        Figure(
            "paper-fig_001", "paper", 2, "picture", "Figure 1: Architecture",
            None, "figure.png", "", "Methods", "The architecture is described here.", [],
        ),
        Figure(
            "paper-table_001", "paper", 3, "table", "Table 1: Results",
            None, "table.png", "| Method | Score |\n|---|---|\n| Ours | 90 |",
            "Results", "The results are reported here.", [],
        ),
    ]
    return paper


def test_executor_factory_preserves_legacy_by_default() -> None:
    local = replace(settings, agent_orchestrator="legacy", vector_enabled=False, reranker_enabled=False)
    executor = build_agent_executor(local, HybridRetriever(local), FakeReading())
    assert executor.__class__.__name__ == "AgentExecutor"


def test_langgraph_skeleton_runs_through_compatible_adapter() -> None:
    local = replace(settings, agent_orchestrator="langgraph", vector_enabled=False, reranker_enabled=False)
    executor = build_agent_executor(local, HybridRetriever(local), FakeReading())
    result = executor.run(
        sample_paper(),
        "Why do the authors claim this method is novel?",
        {
            "complexity": "complex",
            "route": "agentic_rag",
            "modalities": ["text"],
            "required_modalities": ["text"],
            "pure_visual": False,
            "pure_table": False,
            "reason": "test",
        },
    )

    assert isinstance(executor, LangGraphExecutor)
    assert result["orchestrator"] == "langgraph"
    assert result["trace"]["orchestrator"] == "langgraph"
    assert result["trace"]["node_traces"][0]["node"] == "supervisor_plan"
    assert result["trace"]["node_traces"][1]["node"] == "specialist_dispatch"
    assert result["trace"]["node_traces"][1]["agent"] == "text_agent"
    assert result["mode"] == "text_multi_agent_rag"
    assert result["answer"] == "Grounded answer."


def test_langgraph_publishes_node_specialist_tool_and_verified_answer_events() -> None:
    local = replace(
        settings, agent_orchestrator="langgraph", vector_enabled=False,
        reranker_enabled=False, langgraph_checkpoint_enabled=False,
        specialist_execution_mode="fixed",
    )
    executor = build_agent_executor(local, HybridRetriever(local), FakeReading())
    events = []
    executor.run(
        sample_paper(),
        "What method is novel?",
        {
            "complexity": "complex", "route": "agentic_rag",
            "modalities": ["text"], "required_modalities": ["text"],
            "pure_visual": False, "pure_table": False, "reason": "event test",
        },
        event_callback=lambda event, data: events.append((event, data)),
    )
    names = [event for event, _data in events]
    assert "langgraph.node.started" in names
    assert "langgraph.node.completed" in names
    assert "specialist.started" in names and "specialist.completed" in names
    assert "tool.completed" in names
    assert "answer.verified" in names


def test_final_answer_has_reserved_model_budget_after_specialist_calls() -> None:
    local = replace(
        settings,
        agent_orchestrator="langgraph",
        vector_enabled=False,
        reranker_enabled=False,
        langgraph_checkpoint_enabled=False,
        specialist_execution_mode="function_calling_with_fallback",
        multi_agent_max_model_calls=1,
    )
    reading = ToolCallingReading()
    executor = build_agent_executor(local, HybridRetriever(local), reading)

    result = executor.run(
        sample_paper(),
        "Why is this method novel?",
        {
            "complexity": "complex", "route": "agentic_rag",
            "modalities": ["text"], "required_modalities": ["text"],
            "pure_visual": False, "pure_table": False, "reason": "budget test",
        },
    )

    assert result["answer"] == "Grounded answer."
    assert result["status"] == "ok"
    assert result["trace"]["model_calls"] == 2


def test_langgraph_dispatches_all_specialists_for_cross_modal_question() -> None:
    local = replace(
        settings,
        agent_orchestrator="langgraph",
        vector_enabled=False,
        reranker_enabled=False,
        agent_figure_read_limit=1,
    )
    executor = build_agent_executor(
        local, HybridRetriever(local), FakeReading(), FakeFigureService()
    )
    result = executor.run(
        cross_modal_paper(),
        "Based on Figure 1, the method description, and Table 1, what is supported?",
        {
            "complexity": "complex",
            "route": "agentic_rag",
            "modalities": ["text", "figure", "table"],
            "required_modalities": ["text", "figure", "table"],
            "pure_visual": False,
            "pure_table": False,
            "reason": "cross-modal test",
        },
    )

    agents = {
        item["agent"] for item in result["trace"]["node_traces"]
        if item["node"] == "specialist_dispatch"
    }
    evidence_types = {item["type"] for item in result["evidence_memory"]}
    tools = {item["tool"] for item in result["trace"]["steps"]}
    assert agents == {"text_agent", "figure_agent", "table_agent"}
    assert evidence_types == {"text", "figure", "table"}
    assert {
        "search_text", "search_figures", "analyze_figure_for_query",
        "search_tables", "read_table",
    } <= tools


def test_langgraph_records_parallel_execution_mode() -> None:
    local = replace(
        settings,
        agent_orchestrator="langgraph",
        vector_enabled=False,
        reranker_enabled=False,
        langgraph_checkpoint_enabled=False,
        multi_agent_parallel_enabled=True,
        agent_figure_read_limit=1,
    )
    executor = build_agent_executor(
        local, HybridRetriever(local), FakeReading(), FakeFigureService()
    )
    result = executor.run(
        cross_modal_paper(),
        "Based on Figure 1, the method description, and Table 1, what is supported?",
        {
            "complexity": "complex", "route": "agentic_rag",
            "modalities": ["text", "figure", "table"],
            "required_modalities": ["text", "figure", "table"],
            "pure_visual": False, "pure_table": False, "reason": "parallel test",
        },
    )

    assert result["trace"]["execution_mode"] == "parallel"
    assert {item["type"] for item in result["evidence_memory"]} == {
        "text", "figure", "table"
    }


def test_langgraph_critic_repairs_one_failed_visual_task() -> None:
    local = replace(
        settings,
        agent_orchestrator="langgraph",
        vector_enabled=False,
        reranker_enabled=False,
        multi_agent_max_retries=1,
        agent_figure_read_limit=1,
    )
    executor = build_agent_executor(
        local, HybridRetriever(local), FakeReading(), RecoveringFigureService()
    )
    paper = cross_modal_paper()
    paper.figures = [paper.figures[0]]
    result = executor.run(
        paper,
        "What is visible in Figure 1?",
        {
            "complexity": "complex",
            "route": "agentic_rag",
            "modalities": ["figure"],
            "required_modalities": ["figure"],
            "pure_visual": True,
            "pure_table": False,
            "reason": "visual repair test",
        },
    )

    assert result["critique"]["decision"] == "approved"
    assert result["trace"]["retry_count"] == 1
    assert result["trace"]["recovery"]["successful"] is True
    assert result["trace"]["qwen_vl_calls"] == 2
    assert any(
        item["node"] == "repair_dispatch"
        for item in result["trace"]["node_traces"]
    )


def test_langgraph_persists_sqlite_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "langgraph.sqlite3"
    local = replace(
        settings,
        agent_orchestrator="langgraph",
        vector_enabled=False,
        reranker_enabled=False,
        langgraph_checkpoint_enabled=True,
        langgraph_checkpoint_path_value=str(checkpoint),
    )
    executor = build_agent_executor(local, HybridRetriever(local), FakeReading())
    result = executor.run(
        sample_paper(),
        "What method is novel?",
        {
            "complexity": "complex", "route": "agentic_rag",
            "modalities": ["text"], "required_modalities": ["text"],
            "pure_visual": False, "pure_table": False, "reason": "checkpoint test",
        },
    )

    assert result["answer"] == "Grounded answer."
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0
    restored = executor.load_checkpoint(result["trace"]["run_id"])
    assert restored is not None
    assert restored["answer"]["answer"] == "Grounded answer."


def test_qwen_budget_prevents_visual_model_call() -> None:
    service = FakeFigureService()
    local = replace(
        settings,
        agent_orchestrator="langgraph",
        vector_enabled=False,
        reranker_enabled=False,
        langgraph_checkpoint_enabled=False,
        multi_agent_max_qwen_vl_calls=0,
        multi_agent_max_retries=0,
    )
    executor = build_agent_executor(local, HybridRetriever(local), FakeReading(), service)
    paper = cross_modal_paper()
    paper.figures = [paper.figures[0]]
    result = executor.run(
        paper,
        "What is visible in Figure 1?",
        {
            "complexity": "complex", "route": "agentic_rag",
            "modalities": ["figure"], "required_modalities": ["figure"],
            "pure_visual": True, "pure_table": False, "reason": "budget test",
        },
    )

    assert service.interactive_call_count == 0
    assert result["critique"]["decision"] == "refuse"
    assert result["answerable"] is False
    assert result["trace"]["qwen_vl_calls"] == 0
