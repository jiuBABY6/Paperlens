"""有界、多工具、可追踪的 Agentic RAG 执行循环。"""

import json
import time
import uuid
from typing import Callable, Any

from app.agent.evidence_memory import EvidenceMemory
from app.agent.planner import Planner
from app.schemas.trace import AgentTrace, TraceStep
from app.tools.evidence_tools import EvidenceTools


class AgentExecutor:
    def __init__(self, settings, retriever, reading, figure_service=None) -> None:
        self.settings = settings
        self.retriever = retriever
        self.reading = reading
        self.figure_service = figure_service
        self.planner = Planner()

    def run(
        self,
        paper,
        question: str,
        router: dict,
        strategy: str | None = None,
        *,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict:
        started = time.perf_counter()
        plan = self.planner.plan(question, router["modalities"])
        trace = AgentTrace(str(uuid.uuid4()), question, "agentic_rag", router, plan)
        memory = EvidenceMemory()
        tools = EvidenceTools(
            paper,
            self.retriever,
            self.figure_service,
            search_strategy=strategy,
        )
        initial_qwen_calls = (
            self.figure_service.interactive_call_count if self.figure_service else 0
        )
        initial_tokens = getattr(self.reading, "token_usage", 0)
        if self.figure_service:
            initial_tokens += getattr(self.figure_service.client, "token_usage", 0)
        cache: dict[str, tuple[list, dict]] = {}
        for task in plan["sub_tasks"]:
            if trace.total_steps >= self.settings.agent_max_steps:
                break
            query = self.planner.rewrite(task["description"], question)
            trace.rewrites.append({"subtask_id": task["id"], "attempt": 0, "query": query})
            for modality in task["preferred_modalities"]:
                if trace.total_steps >= self.settings.agent_max_steps:
                    break
                results = self._search(tools, memory, trace, cache, task["id"], modality, query)
                if modality == "figure" and results:
                    self._analyze_figure(tools, trace, task["id"], question, results)
                    if "text" in router.get("required_modalities", []):
                        self._read_figure_context(tools, memory, trace, task["id"], results[0][0])
                elif modality == "table" and results:
                    self._read_table(tools, trace, task["id"], results[0][0].evidence_id)
            task["status"] = "completed" if any(
                task["id"] in item.get("subtask_ids", [item["subtask_id"]])
                for item in memory.records()
            ) else "pending"

        sufficiency = self._evidence_sufficiency(memory, plan)
        for task_id in sufficiency["missing_subtasks"]:
            if trace.total_steps >= self.settings.agent_max_steps:
                break
            task = next(item for item in plan["sub_tasks"] if item["id"] == task_id)
            query = self.planner.rewrite(task["description"], question, attempt=1)
            trace.rewrites.append({"subtask_id": task_id, "attempt": 1, "query": query})
            modality = task["preferred_modalities"][0]
            results = self._search(tools, memory, trace, cache, task_id, modality, query)
            if modality == "figure" and results:
                self._analyze_figure(tools, trace, task_id, question, results)
                if "text" in router.get("required_modalities", []):
                    self._read_figure_context(tools, memory, trace, task_id, results[0][0])
            elif modality == "table" and results:
                self._read_table(tools, trace, task_id, results[0][0].evidence_id)
        sufficiency = self._evidence_sufficiency(memory, plan)
        for task in plan["sub_tasks"]:
            task["status"] = (
                "completed" if task["id"] in sufficiency["covered_subtasks"] else "missing_evidence"
            )

        answer_evidence = [
            item for item in memory.evidence()
            if item.type != "figure"
            or item.metadata.get("query_analysis", {}).get("answerable") is True
        ]
        grounded = self.reading.answer_from_evidence(
            question,
            answer_evidence,
            required_evidence_types=router.get("required_modalities", ["text"]),
        )
        grounded["insufficient_evidence"] = list(dict.fromkeys([
            *grounded.get("insufficient_evidence", []),
            *sufficiency["missing_information"],
        ]))
        claim_coverage = grounded.get("claim_coverage", {})
        sufficiency["answer_coverage"] = claim_coverage
        if claim_coverage and not claim_coverage.get("complete", False):
            sufficiency["sufficient"] = False
            grounded["status"] = "partial"
        if grounded.get("insufficient_evidence"):
            sufficiency["sufficient"] = False
            grounded["status"] = "partial"
        visual_answer_check: dict = {}
        if (
            getattr(self.settings, "online_visual_judge_enabled", False)
            and self.figure_service
            and grounded.get("answerable") is True
            and "figure" in router.get("required_modalities", [])
        ):
            cited_figure = next((
                tools.figures.get(item.get("evidence_id", item.get("id")))
                for item in grounded.get("citations", [])
                if item.get("type") == "figure"
                and tools.figures.get(item.get("evidence_id", item.get("id")))
            ), None)
            if cited_figure:
                visual_answer_check = self.figure_service.verify_answer_online(
                    cited_figure,
                    question,
                    grounded.get("answer", ""),
                )
                if visual_answer_check.get("supported") is False:
                    message = "视觉核验未通过。"
                    grounded["insufficient_evidence"] = list(dict.fromkeys([
                        *grounded.get("insufficient_evidence", []),
                        message,
                    ]))
                    cited_figure_id = cited_figure.id
                    for claim in grounded.get("claims", []):
                        if cited_figure_id in claim.get("evidence_ids", []):
                            claim["verification_status"] = "visually_contested"
                    sufficiency["sufficient"] = False
                    grounded["status"] = "partial"
                    grounded["visual_verification_status"] = "failed"
                elif visual_answer_check.get("supported") is True:
                    grounded["visual_verification_status"] = "passed"
                else:
                    grounded["visual_verification_status"] = "inconclusive"
        trace.final_evidence_ids = [item["evidence_id"] for item in grounded.get("citations", [])]
        memory.mark_used(trace.final_evidence_ids)
        trace.sufficiency = sufficiency
        trace.visual_answer_check = visual_answer_check
        trace.evidence_memory = memory.records()
        trace.retrieval = self._retrieval_summary(trace)
        trace.total_latency_ms = round((time.perf_counter() - started) * 1000, 1)
        if self.figure_service:
            trace.qwen_vl_calls = (
                self.figure_service.interactive_call_count - initial_qwen_calls
            )
        final_tokens = getattr(self.reading, "token_usage", 0)
        if self.figure_service:
            final_tokens += getattr(self.figure_service.client, "token_usage", 0)
        trace.token_usage = final_tokens - initial_tokens
        if event_callback:
            for step in trace.steps:
                payload = vars(step)
                event_callback("tool.completed", {
                    "agent": "legacy_agent",
                    "task_id": payload.get("subtask_id"),
                    "tool": payload.get("tool"),
                    "result_ids": payload.get("result_ids", []),
                    "latency_ms": payload.get("latency_ms"),
                    "status": "success",
                    "protocol": "legacy_fixed",
                })
            event_callback("answer.verified", {
                "node": "legacy_answer",
                "status": grounded.get("status", "unknown"),
                "answerable": grounded.get("answerable"),
                "citation_count": len(grounded.get("citations", [])),
            })
        return {
            **grounded,
            "route": "agentic_rag",
            "mode": (
                "multimodal_agentic_rag"
                if any(item in router["modalities"] for item in ("figure", "table"))
                else "text_agentic_rag"
            ),
            "router": router,
            "plan": plan,
            "evidence_memory": memory.records(),
            "sufficiency": sufficiency,
            "evidence_sufficient": sufficiency["sufficient"],
            "retrieval": trace.retrieval,
            "trace": trace.to_dict(),
            "latency_ms": trace.total_latency_ms,
            "visual_answer_check": visual_answer_check,
            "visual_verification_status": grounded.get(
                "visual_verification_status", "not_run"
            ),
        }

    def _search(self, tools, memory, trace, cache, subtask_id: str, modality: str, query: str) -> list:
        name = f"search_{modality if modality != 'figure' else 'figures'}"
        if modality == "table":
            name = "search_tables"
        started = time.perf_counter()
        arguments = {"query": query, "top_k": 5 if modality != "text" else 10}
        key = json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)
        cached = key in cache
        function = getattr(tools, name, None)
        if cached:
            results, retrieval_trace = cache[key]
        else:
            results = function(**arguments) if function else []
            retrieval_trace = dict(tools.last_search_trace) if function else {
                "effective_strategy": "none",
                "degraded": True,
                "warnings": [f"工具 {name} 不可用"],
            }
            cache[key] = (results, retrieval_trace)
        for evidence, score in results:
            memory.add(subtask_id, evidence, score, name)
        trace.total_steps += 1
        trace.steps.append(TraceStep(
            trace.total_steps,
            subtask_id,
            name,
            arguments,
            [item.evidence_id for item, _score in results],
            round((time.perf_counter() - started) * 1000, 1),
            cached,
            retrieval_trace,
        ))
        return results

    def _analyze_figure(self, tools, trace, subtask_id: str, question: str, results: list) -> None:
        for evidence, _score in results[: self.settings.agent_figure_read_limit]:
            if trace.total_steps >= self.settings.agent_max_steps:
                break
            started = time.perf_counter()
            analysis = tools.analyze_figure_for_query(evidence.evidence_id, question)
            if analysis:
                evidence.metadata["query_analysis"] = analysis
            trace.total_steps += 1
            trace.steps.append(TraceStep(
                trace.total_steps,
                subtask_id,
                "analyze_figure_for_query",
                {"figure_id": evidence.evidence_id, "question": question},
                [evidence.evidence_id] if analysis else [],
                round((time.perf_counter() - started) * 1000, 1),
                bool(analysis and analysis.get("cached")),
            ))

    def _read_table(self, tools, trace, subtask_id: str, table_id: str) -> None:
        if trace.total_steps >= self.settings.agent_max_steps:
            return
        started = time.perf_counter()
        value = tools.read_table(table_id)
        trace.total_steps += 1
        trace.steps.append(TraceStep(
            trace.total_steps,
            subtask_id,
            "read_table",
            {"table_id": table_id},
            [table_id] if value else [],
            round((time.perf_counter() - started) * 1000, 1),
            False,
        ))

    def _read_figure_context(self, tools, memory, trace, subtask_id: str, figure) -> None:
        """Follow a selected figure's parser-backed links to nearby source text."""
        if trace.total_steps >= self.settings.agent_max_steps:
            return
        started = time.perf_counter()
        result_ids = []
        for evidence_id in figure.metadata.get("related_sentence_ids", [])[:3]:
            evidence = tools.get_evidence(evidence_id)
            if not evidence or evidence.type != "text" or not evidence.content:
                continue
            memory.add(subtask_id, evidence, 1.0, "read_figure_context")
            result_ids.append(evidence.evidence_id)
        trace.total_steps += 1
        trace.steps.append(TraceStep(
            trace.total_steps,
            subtask_id,
            "read_figure_context",
            {"figure_id": figure.evidence_id, "limit": 3},
            result_ids,
            round((time.perf_counter() - started) * 1000, 1),
            False,
        ))

    def _evidence_sufficiency(self, memory: EvidenceMemory, plan: dict) -> dict:
        """按子任务所需模态判断证据可用性，而不是只统计召回数量。"""
        evidence = {item.evidence_id: item for item in memory.evidence()}
        records = memory.records()
        covered: list[str] = []
        missing: list[str] = []
        for task in plan["sub_tasks"]:
            task_records = [
                item for item in records
                if task["id"] in item.get("subtask_ids", [item["subtask_id"]])
            ]
            usable = False
            for record in task_records:
                item = evidence.get(record["evidence_id"])
                if not item or item.type not in task["preferred_modalities"]:
                    continue
                if item.type == "figure":
                    analysis = item.metadata.get("query_analysis", {})
                    usable = analysis.get("answerable") is True
                elif item.type == "table":
                    usable = bool(item.content and item.content.strip())
                else:
                    usable = bool(item.content and item.content.strip())
                if usable:
                    break
            (covered if usable else missing).append(task["id"])
        return {
            "sufficient": not missing,
            "covered_subtasks": covered,
            "missing_subtasks": missing,
            "missing_information": [f"{item} 尚未找到可用的原始模态证据。" for item in missing],
        }

    def _retrieval_summary(self, trace: AgentTrace) -> dict:
        """汇总 Agent 多次检索实际使用的后端和所有降级原因。"""
        retrievals = [step.retrieval for step in trace.steps if step.retrieval]
        strategies = list(dict.fromkeys(
            item.get("effective_strategy", "none") for item in retrievals
        ))
        warnings = list(dict.fromkeys(
            warning for item in retrievals for warning in item.get("warnings", [])
        ))
        return {
            "effective_strategy": "+".join(strategies) if strategies else "none",
            "effective_strategies": strategies,
            "degraded": any(item.get("degraded") for item in retrievals),
            "warnings": warnings,
            "search_count": len(retrievals),
        }
