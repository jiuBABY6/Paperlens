from dataclasses import replace

from app.agent.router import QueryRouter
from app.config import settings
from app.multi_agent.supervisor import SupervisorAgent


def test_supervisor_assigns_only_figure_agent_for_pure_visual_query() -> None:
    local = replace(settings, multi_agent_parallel_enabled=True)
    question = "在图 2 中，两个标签框分别是什么颜色？"
    plan = SupervisorAgent(local).plan(question, QueryRouter().route(question))

    assert plan["selected_agents"] == ["figure_agent"]
    assert plan["execution"] == "sequential"
    assert {item["modality"] for item in plan["sub_tasks"]} == {"figure"}


def test_supervisor_expands_cross_modal_plan_into_specialists() -> None:
    local = replace(settings, multi_agent_parallel_enabled=True)
    question = "Based on Figure 1, the method description, and Table 2, compare the design choices."
    route = QueryRouter().route(question)
    plan = SupervisorAgent(local).plan(question, route)

    assert plan["execution"] == "parallel"
    assert set(plan["selected_agents"]) == {"text_agent", "figure_agent", "table_agent"}
    assert all(item["agent"].endswith("_agent") for item in plan["sub_tasks"])
    assert all(item["query"] for item in plan["sub_tasks"])
