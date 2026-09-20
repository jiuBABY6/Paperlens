"""LangGraph 多智能体编排入口（使用惰性导入避免服务层循环依赖）。"""

from typing import Any


__all__ = ["LangGraphExecutor", "build_agent_executor"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from app.multi_agent.graph import LangGraphExecutor, build_agent_executor

        return {
            "LangGraphExecutor": LangGraphExecutor,
            "build_agent_executor": build_agent_executor,
        }[name]
    raise AttributeError(name)
