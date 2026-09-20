"""复杂度感知、白名单约束的 Supervisor Agent。"""

from __future__ import annotations

from typing import Any

from app.agent.planner import Planner


AGENT_FOR_MODALITY = {
    "text": "text_agent",
    "figure": "figure_agent",
    "table": "table_agent",
}


class SupervisorAgent:
    """把现有可解释 Planner 输出转换为多智能体调度计划。"""

    def __init__(self, settings, planner: Planner | None = None) -> None:
        self.settings = settings
        self.planner = planner or Planner()

    def plan(self, question: str, router: dict[str, Any]) -> dict[str, Any]:
        raw = self.planner.plan(question, router.get("modalities", ["text"]))
        tasks: list[dict[str, Any]] = []
        for task in raw.get("sub_tasks", []):
            modalities = [
                item for item in task.get("preferred_modalities", [])
                if item in AGENT_FOR_MODALITY
            ]
            for modality in modalities:
                task_id = (
                    task["id"] if len(modalities) == 1
                    else f"{task['id']}_{modality}"
                )
                tasks.append({
                    "task_id": task_id,
                    "source_task_id": task["id"],
                    "agent": AGENT_FOR_MODALITY[modality],
                    "modality": modality,
                    "instruction": task["description"],
                    "query": self.planner.rewrite(task["description"], question),
                    "required_modalities": [modality],
                    "dependencies": [],
                    "status": "pending",
                })
        selected_agents = list(dict.fromkeys(task["agent"] for task in tasks))
        execution = (
            "parallel"
            if self.settings.multi_agent_parallel_enabled and len(selected_agents) > 1
            else "sequential"
        )
        return {
            "goal": question,
            "execution": execution,
            "selected_agents": selected_agents,
            "reason": router.get("reason", ""),
            "sub_tasks": tasks,
            "budgets": {
                "max_steps": self.settings.multi_agent_max_steps,
                "max_retries": self.settings.multi_agent_max_retries,
                "max_model_calls": self.settings.multi_agent_max_model_calls,
                "max_qwen_vl_calls": self.settings.multi_agent_max_qwen_vl_calls,
                "timeout_seconds": self.settings.multi_agent_timeout_seconds,
            },
        }

    def repair_tasks(
        self,
        plan: dict[str, Any],
        target_ids: list[str],
        critique: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """只返工 Critic 指定的任务，并把明确缺口加入检索指令。"""
        wanted = set(target_ids[:1])
        reason = " ".join(str(item) for item in critique.get("reasons", []))
        return [{
            **task,
            "query": f"{task['query']} Focus specifically on: {reason}".strip(),
            "status": "pending",
            "repair_reason": reason,
        } for task in plan.get("sub_tasks", []) if task.get("task_id") in wanted]
