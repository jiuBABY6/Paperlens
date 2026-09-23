"""LangGraph 多智能体 Executor 与 Legacy/LangGraph 选择器。"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agent.executor import AgentExecutor
from app.multi_agent.agents import (
    AnswerSynthesisAgent,
    EvidenceCriticAgent,
    FigureAnalysisAgent,
    TableAnalysisAgent,
    TextResearchAgent,
)
from app.multi_agent.scheduler import SpecialistScheduler
from app.multi_agent.state import MultiAgentState, merge_evidence
from app.multi_agent.supervisor import SupervisorAgent
from app.multi_agent.trace_adapter import legacy_compatible_trace
from app.tools.evidence_tools import EvidenceTools
from app.multi_agent.function_calling import FunctionCallingSpecialistExecutor


class LangGraphExecutor:
    """以 LangGraph 编排职责隔离的多模态 Specialist Agents。"""

    def __init__(self, settings, retriever, reading, figure_service=None) -> None:
        self.settings = settings
        self.retriever = retriever
        self.reading = reading
        self.figure_service = figure_service
        self.supervisor = SupervisorAgent(settings)
        self.critic = EvidenceCriticAgent(settings)
        self.answer_agent = AnswerSynthesisAgent(reading)

    def _agent(
        self, name: str, paper, strategy: str | None,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        tools = EvidenceTools(
            paper,
            self.retriever,
            self.figure_service,
            search_strategy=strategy,
        )
        classes = {
            "text_agent": TextResearchAgent,
            "figure_agent": FigureAnalysisAgent,
            "table_agent": TableAnalysisAgent,
        }
        agent = classes[name](self.settings, tools)
        agent.function_executor = FunctionCallingSpecialistExecutor(
            self.reading, tools, name, self.settings, event_callback=event_callback
        )
        return agent

    def load_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        """读取持久化运行状态，供故障诊断和后续恢复入口使用。"""
        if not self.settings.langgraph_checkpoint_enabled:
            return None
        path = self.settings.langgraph_checkpoint_path
        if not path.is_file():
            return None
        with SqliteSaver.from_conn_string(str(path)) as checkpointer:
            value = checkpointer.get_tuple({
                "configurable": {"thread_id": run_id}
            })
        if value is None:
            return None
        return dict(value.checkpoint.get("channel_values", {}))

    def _build_graph(self, paper, checkpointer=None, event_callback=None):
        def emit(event: str, data: dict[str, Any]) -> None:
            if event_callback:
                event_callback(event, data)

        def plan_node(state: MultiAgentState) -> dict[str, Any]:
            started = time.perf_counter()
            emit("langgraph.node.started", {"node": "supervisor_plan"})
            plan = self.supervisor.plan(state["question"], state["route"])
            latency = round((time.perf_counter() - started) * 1000, 1)
            emit("langgraph.node.completed", {
                "node": "supervisor_plan", "status": "success",
                "selected_agents": plan["selected_agents"], "latency_ms": latency,
            })
            return {
                "plan": plan,
                "pending_tasks": plan["sub_tasks"],
                "execution_mode": plan["execution"],
                "node_traces": [{
                    "node": "supervisor_plan",
                    "agent": "supervisor_agent",
                    "status": "success",
                    "latency_ms": latency,
                    "selected_agents": plan["selected_agents"],
                    "task_ids": [item["task_id"] for item in plan["sub_tasks"]],
                    "evidence_ids": [],
                    "errors": [],
                }],
            }

        def dispatch(
            state: MultiAgentState,
            tasks: list[dict[str, Any]],
            *,
            attempt: int,
            node_name: str,
        ) -> dict[str, Any]:
            emit("langgraph.node.started", {
                "node": node_name,
                "attempt": attempt,
                "task_ids": [item.get("task_id") for item in tasks],
            })
            budget_error = ""
            if time.time() >= state.get("deadline_at", float("inf")):
                budget_error = "Multi-Agent 全局执行超时。"
            elif int(state.get("total_steps", 0)) >= self.settings.multi_agent_max_steps:
                budget_error = "Multi-Agent 工具步数预算已耗尽。"
            elif int(state.get("model_calls", 0)) >= self.settings.multi_agent_max_model_calls:
                budget_error = "Multi-Agent 模型调用预算已耗尽。"

            remaining_qwen = max(
                0,
                self.settings.multi_agent_max_qwen_vl_calls
                - int(state.get("qwen_vl_calls", 0)),
            )
            remaining_steps = max(
                0,
                self.settings.multi_agent_max_steps
                - int(state.get("total_steps", 0)),
            )
            tasks = [
                {
                    **task,
                    "max_qwen_calls": min(remaining_qwen, max(0, remaining_steps - 1)),
                    "max_tool_steps": remaining_steps,
                }
                if task.get("agent") == "figure_agent" else task
                for task in tasks
            ]
            tasks = [
                {**task, "max_tool_steps": remaining_steps}
                if task.get("agent") != "figure_agent" else task
                for task in tasks
            ]
            for task in tasks:
                emit("specialist.started", {
                    "node": node_name,
                    "attempt": attempt,
                    "agent": task.get("agent"),
                    "task_id": task.get("task_id"),
                    "modality": task.get("modality"),
                })
            if budget_error:
                results = [{
                    "agent": task["agent"],
                    "modality": task["modality"],
                    "task_id": task["task_id"],
                    "status": "failed",
                    "observations": [],
                    "inferences": [],
                    "candidate_claims": [],
                    "evidence": [],
                    "evidence_ids": [],
                    "missing_information": [budget_error],
                    "confidence": 0.0,
                    "retryable": False,
                    "tool_calls": [],
                    "latency_ms": 0.0,
                    "token_usage": 0,
                    "model_calls": 0,
                    "qwen_vl_calls": 0,
                    "errors": [{
                        "type": "BudgetExceeded",
                        "category": "budget",
                        "message": budget_error,
                        "retryable": False,
                    }],
                } for task in tasks]
            else:
                agents = {
                    name: self._agent(
                        name, paper, state.get("requested_strategy"), event_callback
                    )
                    for name in {task["agent"] for task in tasks}
                }
                scheduler = SpecialistScheduler(
                    parallel_enabled=state.get("execution_mode") == "parallel"
                )
                remaining_timeout = max(0.001, state.get("deadline_at", time.time()) - time.time())
                results = scheduler.execute(
                    tasks,
                    agents,
                    state["question"],
                    attempt=attempt,
                    timeout_seconds=remaining_timeout,
                )
            evidence = merge_evidence(*[item.get("evidence", []) for item in results])
            steps = [step for item in results for step in item.get("tool_calls", [])]
            completed = [
                {**task, "status": next(
                    (item["status"] for item in results if item["task_id"] == task["task_id"]),
                    "failed",
                )}
                for task in tasks
            ]
            for item in results:
                emit("specialist.completed", {
                    "node": node_name,
                    "attempt": attempt,
                    "agent": item.get("agent"),
                    "task_id": item.get("task_id"),
                    "status": item.get("status"),
                    "latency_ms": item.get("latency_ms"),
                    "evidence_ids": item.get("evidence_ids", []),
                })
            function_call_ids = {
                step.get("tool_call_id") for step in steps
                if step.get("protocol") == "function_calling"
            }
            for step in steps:
                if step.get("tool_call_id") in function_call_ids:
                    continue
                emit("tool.completed", {
                    "agent": step.get("agent"),
                    "task_id": step.get("task_id", step.get("subtask_id")),
                    "tool": step.get("tool"),
                    "result_ids": step.get("result_ids", []),
                    "latency_ms": step.get("latency_ms"),
                    "status": step.get("status", "success"),
                    "protocol": step.get("protocol", "fixed"),
                })
            emit("langgraph.node.completed", {
                "node": node_name,
                "attempt": attempt,
                "task_count": len(tasks),
                "statuses": [item.get("status") for item in results],
                "tool_call_count": len(steps),
            })
            return {
                "agent_results": {item["task_id"]: item for item in results},
                "evidence": evidence,
                "evidence_ids": [item["evidence_id"] for item in evidence],
                "completed_tasks": completed,
                "pending_tasks": [],
                "trace_steps": steps,
                "total_steps": len(steps),
                "token_usage": sum(item.get("token_usage", 0) for item in results),
                "qwen_vl_calls": sum(item.get("qwen_vl_calls", 0) for item in results),
                "model_calls": sum(item.get("model_calls", 0) for item in results),
                "errors": [error for item in results for error in item.get("errors", [])],
                "node_traces": [{
                    "node": node_name,
                    "agent": item["agent"],
                    "task_id": item["task_id"],
                    "status": item["status"],
                    "latency_ms": item["latency_ms"],
                    "evidence_ids": item["evidence_ids"],
                    "errors": item["errors"],
                } for item in results],
            }

        def dispatch_node(state: MultiAgentState) -> dict[str, Any]:
            return dispatch(
                state,
                state.get("pending_tasks", []),
                attempt=0,
                node_name="specialist_dispatch",
            )

        def critic_node(state: MultiAgentState) -> dict[str, Any]:
            started = time.perf_counter()
            emit("langgraph.node.started", {"node": "evidence_critic"})
            critique = self.critic.review(state)
            latency = round((time.perf_counter() - started) * 1000, 1)
            emit("langgraph.node.completed", {
                "node": "evidence_critic", "status": critique["decision"],
                "latency_ms": latency,
                "missing_modalities": critique["missing_modalities"],
            })
            return {
                "critique": critique,
                "retry_targets": critique["retry_targets"],
                "node_traces": [{
                    "node": "evidence_critic",
                    "agent": "evidence_critic",
                    "status": critique["decision"],
                    "latency_ms": latency,
                    "evidence_ids": [
                        item["evidence_id"] for item in state.get("evidence", [])
                    ],
                    "missing_modalities": critique["missing_modalities"],
                    "conflicts": critique["conflicts"],
                    "errors": [],
                }],
            }

        def route_after_critic(state: MultiAgentState) -> str:
            return str(state.get("critique", {}).get("decision", "refuse"))

        def repair_node(state: MultiAgentState) -> dict[str, Any]:
            tasks = self.supervisor.repair_tasks(
                state.get("plan", {}),
                state.get("retry_targets", []),
                state.get("critique", {}),
            )
            emit("repair.dispatched", {
                "attempt": int(state.get("retry_count", 0)) + 1,
                "task_ids": [item.get("task_id") for item in tasks],
                "retry_targets": state.get("retry_targets", []),
            })
            update = dispatch(
                state,
                tasks,
                attempt=int(state.get("retry_count", 0)) + 1,
                node_name="repair_dispatch",
            )
            update["retry_count"] = int(state.get("retry_count", 0)) + 1
            return update

        def answer_node(state: MultiAgentState) -> dict[str, Any]:
            started = time.perf_counter()
            emit("langgraph.node.started", {"node": "answer_agent"})
            if int(state.get("model_calls", 0)) >= self.settings.multi_agent_max_model_calls:
                grounded = {
                    "answer": "模型调用预算已耗尽，无法继续生成答案。",
                    "answerable": None,
                    "claims": [],
                    "citations": [],
                    "insufficient_evidence": ["Multi-Agent 模型调用预算已耗尽。"],
                    "refusal_reason": "",
                    "status": "generation_unavailable",
                    "verification": [],
                }
            else:
                grounded = self.answer_agent.run(state)
            latency = round((time.perf_counter() - started) * 1000, 1)
            emit("answer.verified", {
                "node": "answer_agent",
                "status": grounded.get("status", "unknown"),
                "answerable": grounded.get("answerable"),
                "citation_count": len(grounded.get("citations", [])),
                "latency_ms": latency,
            })
            emit("langgraph.node.completed", {
                "node": "answer_agent", "status": grounded.get("status", "unknown"),
                "latency_ms": latency,
            })
            return {
                "answer": grounded,
                "node_traces": [{
                    "node": "answer_agent",
                    "agent": "answer_agent",
                    "status": grounded.get("status", "unknown"),
                    "latency_ms": latency,
                    "evidence_ids": [
                        item.get("evidence_id", item.get("id"))
                        for item in grounded.get("citations", [])
                    ],
                    "errors": [],
                }],
            }

        def refusal_node(state: MultiAgentState) -> dict[str, Any]:
            reasons = state.get("critique", {}).get("reasons", [])
            reason = "；".join(str(item) for item in reasons) or "没有可用于回答的论文证据。"
            emit("answer.verified", {
                "node": "structured_refusal", "status": "insufficient_evidence",
                "answerable": False, "citation_count": 0,
            })
            return {
                "answer": {
                    "answer": f"论文证据不足，无法回答。{reason}",
                    "answerable": False,
                    "claims": [],
                    "citations": [],
                    "insufficient_evidence": reasons,
                    "refusal_reason": reason,
                    "status": "insufficient_evidence",
                    "verification": [],
                },
                "node_traces": [{
                    "node": "structured_refusal",
                    "agent": "answer_agent",
                    "status": "refuse",
                    "latency_ms": 0.0,
                    "evidence_ids": [],
                    "errors": [],
                }],
            }

        builder = StateGraph(MultiAgentState)
        builder.add_node("plan", plan_node)
        builder.add_node("dispatch_specialists", dispatch_node)
        builder.add_node("evidence_critic", critic_node)
        builder.add_node("repair_dispatch", repair_node)
        builder.add_node("answer", answer_node)
        builder.add_node("structured_refusal", refusal_node)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "dispatch_specialists")
        builder.add_edge("dispatch_specialists", "evidence_critic")
        builder.add_conditional_edges(
            "evidence_critic",
            route_after_critic,
            {
                "approved": "answer",
                "partial": "answer",
                "retry": "repair_dispatch",
                "refuse": "structured_refusal",
            },
        )
        builder.add_edge("repair_dispatch", "evidence_critic")
        builder.add_edge("answer", END)
        builder.add_edge("structured_refusal", END)
        return builder.compile(checkpointer=checkpointer)

    def run(
        self,
        paper,
        question: str,
        router: dict,
        strategy: str | None = None,
        *,
        conversation_id: str | None = None,
        turn_index: int | None = None,
        original_question: str | None = None,
        referenced_evidence_ids: list[str] | None = None,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict:
        started = time.perf_counter()
        initial_tokens = getattr(self.reading, "token_usage", 0)
        initial_model_requests = getattr(self.reading, "request_count", 0)
        if self.figure_service:
            initial_tokens += getattr(self.figure_service.client, "token_usage", 0)
        run_id = str(uuid.uuid4())
        initial_state = {
            "run_id": run_id,
            "paper_id": paper.id,
            "question": question,
            "original_question": original_question or question,
            "resolved_question": question,
            "conversation_id": conversation_id or "",
            "turn_index": turn_index or 0,
            "referenced_evidence_ids": list(referenced_evidence_ids or []),
            "route": router,
            "requested_strategy": strategy,
            "retry_count": 0,
            "errors": [],
            "trace_steps": [],
            "node_traces": [],
            "started_at": time.time(),
            "deadline_at": time.time() + self.settings.multi_agent_timeout_seconds,
            "execution_mode": "sequential",
        }
        graph_config = {
            "configurable": {
                "thread_id": (
                    f"{conversation_id}:{turn_index}"
                    if conversation_id and turn_index else run_id
                )
            },
            "recursion_limit": max(10, self.settings.multi_agent_max_steps * 3),
        }
        if self.settings.langgraph_checkpoint_enabled:
            checkpoint_path = self.settings.langgraph_checkpoint_path
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
                graph = self._build_graph(
                    paper, checkpointer=checkpointer, event_callback=event_callback
                )
                state = graph.invoke(initial_state, config=graph_config)
        else:
            graph = self._build_graph(paper, event_callback=event_callback)
            state = graph.invoke(initial_state, config=graph_config)
        grounded = dict(state.get("answer", {}))
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        steps = [
            {**value, "step": index}
            for index, value in enumerate(state.get("trace_steps", []), start=1)
        ]
        agent_results = state.get("agent_results", {})
        planned_tasks = state.get("plan", {}).get("sub_tasks", [])
        missing_tasks = [
            task["task_id"] for task in planned_tasks
            if agent_results.get(task["task_id"], {}).get("status") != "success"
        ]
        covered_tasks = [
            task["task_id"] for task in planned_tasks
            if agent_results.get(task["task_id"], {}).get("status") == "success"
        ]
        critique = state.get("critique", {})
        sufficiency = {
            "sufficient": critique.get("decision") == "approved",
            "covered_subtasks": covered_tasks,
            "missing_subtasks": missing_tasks,
            "missing_information": list(dict.fromkeys(
                message
                for item in agent_results.values()
                for message in item.get("missing_information", [])
            )),
        }
        if sufficiency["missing_information"]:
            grounded["insufficient_evidence"] = list(dict.fromkeys([
                *grounded.get("insufficient_evidence", []),
                *sufficiency["missing_information"],
            ]))
            if grounded.get("status") == "ok":
                grounded["status"] = "partial"
        retrievals = [step.get("retrieval", {}) for step in steps if step.get("retrieval")]
        strategies = list(dict.fromkeys(
            item.get("effective_strategy", "none") for item in retrievals
        ))
        retrieval = {
            "effective_strategy": "+".join(strategies) if strategies else "none",
            "effective_strategies": strategies,
            "degraded": any(item.get("degraded") for item in retrievals),
            "warnings": list(dict.fromkeys(
                warning for item in retrievals for warning in item.get("warnings", [])
            )),
            "search_count": len(retrievals),
        }
        final_tokens = getattr(self.reading, "token_usage", 0)
        if self.figure_service:
            final_tokens += getattr(self.figure_service.client, "token_usage", 0)
        used_ids = {
            citation.get("evidence_id", citation.get("id"))
            for citation in grounded.get("citations", [])
        }
        trace = legacy_compatible_trace({
            "run_id": run_id,
            "query": question,
            "route": "langgraph_agentic_rag",
            "router": router,
            "plan": state.get("plan", {}),
            "rewrites": [{
                "subtask_id": task["task_id"], "attempt": 0, "query": task["query"]
            } for task in state.get("plan", {}).get("sub_tasks", [])],
            "steps": steps,
            "final_evidence_ids": [
                item.get("evidence_id", item.get("id"))
                for item in grounded.get("citations", [])
            ],
            "total_steps": len(steps),
            "total_latency_ms": elapsed,
            "token_usage": final_tokens - initial_tokens,
            "qwen_vl_calls": int(state.get("qwen_vl_calls", 0)),
            "sufficiency": sufficiency,
            "evidence_memory": [{
                "evidence_id": item["evidence_id"],
                "type": item["type"],
                "used": item["evidence_id"] in used_ids,
            } for item in state.get("evidence", [])],
            "retrieval": retrieval,
            "model_calls": (
                int(state.get("model_calls", 0))
                + max(0, getattr(self.reading, "request_count", 0) - initial_model_requests)
            ),
            "execution_mode": state.get("execution_mode", "sequential"),
            "critique": critique,
            "conversation_id": conversation_id,
            "turn_index": turn_index,
            "original_question": original_question or question,
            "resolved_question": question,
            "referenced_evidence_ids": list(referenced_evidence_ids or []),
            "specialist_execution_mode": getattr(
                self.settings, "specialist_execution_mode", "fixed"
            ),
            "function_calls": [
                step for step in steps if step.get("protocol") == "function_calling"
            ],
            "function_call_fallbacks": sum(
                1 for step in steps if step.get("tool") == "function_calling_fallback"
            ),
        }, orchestrator="langgraph", node_traces=state.get("node_traces", []),
            retry_count=int(state.get("retry_count", 0)),
            recovery={
                "attempted": int(state.get("retry_count", 0)) > 0,
                "successful": (
                    int(state.get("retry_count", 0)) > 0
                    and critique.get("decision") == "approved"
                ),
                "final_decision": critique.get("decision", "unknown"),
            },
        )
        return {
            **grounded,
            "route": "langgraph_agentic_rag",
            "mode": (
                "multimodal_multi_agent_rag"
                if any(item in router["modalities"] for item in ("figure", "table"))
                else "text_multi_agent_rag"
            ),
            "orchestrator": "langgraph",
            "router": router,
            "plan": state.get("plan", {}),
            "agent_results": agent_results,
            "critique": critique,
            "evidence_memory": trace["evidence_memory"],
            "sufficiency": sufficiency,
            "evidence_sufficient": sufficiency["sufficient"],
            "retrieval": retrieval,
            "trace": trace,
            "latency_ms": elapsed,
            "visual_answer_check": {},
            "visual_verification_status": grounded.get(
                "visual_verification_status", "not_run"
            ),
        }


def build_agent_executor(settings, retriever, reading, figure_service=None):
    """根据配置选择已验证的 Legacy 或 LangGraph 编排器。"""
    if settings.agent_orchestrator == "langgraph":
        return LangGraphExecutor(settings, retriever, reading, figure_service)
    return AgentExecutor(settings, retriever, reading, figure_service)
