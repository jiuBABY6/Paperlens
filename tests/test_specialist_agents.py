from dataclasses import replace

from app.config import settings
from app.domain import EvidenceObject
from app.multi_agent.agents import FigureAnalysisAgent, TableAnalysisAgent, TextResearchAgent


class FakeFigureService:
    interactive_call_count = 0


class FakeTools:
    def __init__(self) -> None:
        self.last_search_trace = {"effective_strategy": "bm25", "warnings": [], "degraded": False}
        self.figure_service = FakeFigureService()

    def search_text(self, query, top_k=10):
        return [(EvidenceObject("s1", "text", 1, None, "Text evidence", "Methods"), 0.8)]

    def search_figures(self, query, top_k=5):
        return [(EvidenceObject(
            "fig_001", "figure", 2, None, None, "Method",
            {"caption": "Figure 1"},
        ), 1.0)]

    def analyze_figure_for_query(self, figure_id, question):
        self.figure_service.interactive_call_count += 1
        return {
            "answerable": True,
            "visual_observations": [{"label": "A", "position": "left"}],
            "missing_information": [],
            "confidence": "high",
            "cached": False,
        }

    def search_tables(self, query, top_k=5):
        return [(EvidenceObject(
            "table_001", "table", 3, None, "| A | B |\n|---|---|\n|1|2|", "Results",
            {"caption": "Table 1", "columns": ["A", "B"], "rows": [{"A": "1", "B": "2"}]},
        ), 1.0)]

    def read_table(self, table_id):
        return {"markdown": "| A | B |", "columns": ["A", "B"], "rows": [{"A": "1", "B": "2"}]}


def task(agent: str, modality: str) -> dict:
    return {
        "task_id": f"task_{modality}",
        "agent": agent,
        "modality": modality,
        "instruction": "inspect evidence",
        "query": "Figure 1 Table 1 method",
    }


def test_each_specialist_uses_only_its_modality_tools() -> None:
    local = replace(settings, agent_figure_read_limit=1)

    text = TextResearchAgent(local, FakeTools()).run(task("text_agent", "text"), "question")
    figure = FigureAnalysisAgent(local, FakeTools()).run(task("figure_agent", "figure"), "question")
    table = TableAnalysisAgent(local, FakeTools()).run(task("table_agent", "table"), "question")

    assert text["status"] == "success"
    assert [item["tool"] for item in text["tool_calls"]] == ["search_text"]
    assert figure["status"] == "success"
    assert [item["tool"] for item in figure["tool_calls"]] == [
        "search_figures", "analyze_figure_for_query"
    ]
    assert figure["qwen_vl_calls"] == 1
    assert table["status"] == "success"
    assert [item["tool"] for item in table["tool_calls"]] == [
        "search_tables", "read_table"
    ]
