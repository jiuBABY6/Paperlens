"""职责隔离的 Specialist、Critic 与 Answer Agents。"""

from app.multi_agent.agents.answer_agent import AnswerSynthesisAgent
from app.multi_agent.agents.critic_agent import EvidenceCriticAgent
from app.multi_agent.agents.figure_agent import FigureAnalysisAgent
from app.multi_agent.agents.table_agent import TableAnalysisAgent
from app.multi_agent.agents.text_agent import TextResearchAgent

__all__ = [
    "TextResearchAgent",
    "FigureAnalysisAgent",
    "TableAnalysisAgent",
    "EvidenceCriticAgent",
    "AnswerSynthesisAgent",
]
