"""Build bounded conversation context and resolve history-dependent questions."""

from __future__ import annotations

import json
import re
from typing import Any


_CONTEXT_MARKERS = re.compile(
    r"(?:它|这个|那个|上一个|上一张|刚才|前面|其中|该方法|该表|该图|"
    r"重新|改成|我问的是|what about|how about|it\b|that\b|those\b|previous|above)",
    re.IGNORECASE,
)

_CONVERSATION_HISTORY_QUERY = re.compile(
    r"(?:(?:我|用户)\s*(?:之前|此前|刚才|历史上)\s*"
    r"(?:询问|问|提问|提过)(?:过|了|过了)?\s*(?:哪些|什么).{0,6}(?:问题)?|"
    r"(?:列出|显示|回顾|总结).{0,10}(?:之前|此前|历史|对话).{0,10}(?:问题|提问)|"
    r"\bwhat\s+(?:questions?\s+)?(?:did|have)\s+i\s+(?:ask|asked)\b|"
    r"\blist\s+(?:my\s+)?(?:previous|earlier)\s+questions?\b)",
    re.IGNORECASE,
)

_PAPER_LEARNING_QUERY = re.compile(
    r"(?:我对这篇论文了解多少|我(?:已经|目前)了解了什么|"
    r"(?:我)?(?:还有|哪些).{0,8}(?:没看|未读|没了解)|(?:我的)?(?:学习|阅读)进度|"
    r"总结.{0,8}(?:我的|我对(?:这|该)篇|我在(?:这|该)篇)?.{0,8}"
    r"(?:学习|阅读).{0,6}(?:情况|进度)|"
    r"\bwhat have i learned (?:about|from) (?:this|the) paper\b|"
    r"\bmy (?:reading|learning) progress\b)",
    re.IGNORECASE,
)

_FIRST_PERSON_HISTORY = re.compile(
    r"(?:(?:我|我们|我的|我们的).{0,20}(?:之前|此前|已经|曾经|目前|过往)|"
    r"(?:之前|此前|过往).{0,12}(?:我|我们|我的|我们的)|"
    r"\b(?:i|we|my|our)\b.{0,30}\b(?:previously|earlier|already|so far)\b|"
    r"\b(?:previously|earlier)\b.{0,30}\b(?:i|we|my|our)\b)",
    re.IGNORECASE,
)

_PAPER_LEARNING_SCOPE = re.compile(
    r"(?:论文|文章|文献).{0,24}(?:研究|关注|讨论|学习|阅读|了解|探索|问题)|"
    r"(?:研究|关注|讨论|学习|阅读|了解|探索).{0,24}(?:论文|文章|文献)|"
    r"\b(?:paper|article).{0,40}(?:study|studied|explore|explored|learn|learned|"
    r"read|discuss|discussed|focus|focused|question)|"
    r"\b(?:study|studied|explore|explored|learn|learned|read|discuss|discussed|"
    r"focus|focused).{0,40}(?:paper|article)\b",
    re.IGNORECASE,
)

_PAPER_LEARNING_REQUEST = re.compile(
    r"(?:总结|回顾|列出|梳理|进度|重点|尚未解决|未解决|还没解决|仍待解决|"
    r"\bsummarize\b|\breview\b|\blist\b|\bprogress\b|\bunresolved\b|"
    r"\bopen questions?\b)",
    re.IGNORECASE,
)

_CONTINUE_PREVIOUS_TOPIC = re.compile(
    r"(?:(?:继续|接着)(?:上次|之前|刚才)(?:的话题|的问题|阅读|问答)?|"
    r"\bcontinue (?:the )?(?:previous|last) (?:topic|question|discussion)\b)",
    re.IGNORECASE,
)


def _is_paper_learning_query(question: str) -> bool:
    """Recognize explicit requests about the user's cross-session paper learning state.

    Broad summary forms require first-person history, paper-learning scope, and a
    memory request. This prevents a paper-content question such as “这篇论文尚未
    解决的问题是什么” from being mistaken for a memory-control request.
    """
    if _PAPER_LEARNING_QUERY.search(question):
        return True
    return bool(
        _FIRST_PERSON_HISTORY.search(question)
        and _PAPER_LEARNING_SCOPE.search(question)
        and _PAPER_LEARNING_REQUEST.search(question)
    )


def is_memory_control_query(question: str) -> bool:
    """Return whether a question asks for conversation or learning-memory metadata."""
    return bool(
        _CONVERSATION_HISTORY_QUERY.search(question)
        or _is_paper_learning_query(question)
    )


class QueryResolver:
    """History is used for query interpretation only, never as answer evidence."""

    def __init__(
        self,
        reading,
        *,
        enabled: bool = True,
        recent_turns: int = 6,
        max_chars: int = 12000,
    ) -> None:
        self.reading = reading
        self.enabled = enabled
        self.recent_turns = recent_turns
        self.max_chars = max_chars

    def build_context(self, conversation: dict[str, Any]) -> dict[str, Any]:
        messages = list(conversation.get("messages", []))[-self.recent_turns * 2 :]
        referenced: list[str] = []
        for message in messages:
            referenced.extend(
                str(item.get("evidence_id", item.get("id", "")))
                for item in message.get("citations", [])
                if item.get("evidence_id", item.get("id"))
            )
        payload = {
            "summary": conversation.get("summary", ""),
            "memory": conversation.get("memory", {}),
            "paper_memory": self._bounded_paper_memory(
                conversation.get("paper_memory", {}).get("memory", {})
            ),
            "messages": [
                {"role": item["role"], "content": item["content"]}
                for item in messages
            ],
            "referenced_evidence_ids": list(dict.fromkeys(referenced))[-12:],
        }
        encoded = json.dumps(payload, ensure_ascii=False)
        while len(encoded) > self.max_chars and payload["messages"]:
            payload["messages"].pop(0)
            encoded = json.dumps(payload, ensure_ascii=False)
        return payload

    def resolve(self, question: str, conversation: dict[str, Any]) -> dict[str, Any]:
        context = self.build_context(conversation)
        prior = context["messages"][:-1] if context["messages"] else []
        language = "zh-CN" if re.search(r"[\u4e00-\u9fff]", question) else "en"
        base = {
            "standalone_question": question.strip(),
            "intent": "paper_question",
            "referenced_entities": [],
            "referenced_evidence_ids": context["referenced_evidence_ids"],
            "needs_clarification": False,
            "clarification_question": "",
            "language": language,
            "resolver_used": False,
            "context_message_count": len(prior),
        }
        if _CONVERSATION_HISTORY_QUERY.search(question):
            base["intent"] = "conversation_history"
            return base
        if _is_paper_learning_query(question):
            base["intent"] = "paper_learning_memory"
            return base
        if _CONTINUE_PREVIOUS_TOPIC.search(question):
            last_question = str(
                context.get("paper_memory", {}).get("last_resolved_question", "")
            ).strip()
            if last_question:
                base.update(
                    {
                        "standalone_question": (
                            f"继续深入回答此前关于本论文的问题：{last_question}。"
                            f"当前要求：{question.strip()}"
                        ),
                        "intent": "paper_question",
                        "resolver_used": True,
                        "memory_source": "paper_learning_memory",
                    }
                )
                return base
        if not self.enabled or not prior or not _CONTEXT_MARKERS.search(question):
            return base
        if not getattr(self.reading.settings, "deepseek_key", ""):
            base.update(self._rule_fallback(question, prior, language))
            return base
        prompt = f"""你只负责把多轮追问改写为可独立检索的问题，不回答论文问题。
历史上下文：{json.dumps(context, ensure_ascii=False)}
当前问题：{question}
输出 JSON：{{"standalone_question":"...","referenced_entities":[],"referenced_evidence_ids":[],"needs_clarification":false,"clarification_question":"","language":"{language}"}}
若指代无法唯一确定，needs_clarification=true；不得编造论文事实。"""
        try:
            value = json.loads(
                self.reading._request(
                    prompt,
                    as_json=True,
                    system="你是论文问答的 Query Resolver，只做指代消解。",
                )
            )
            standalone = str(value.get("standalone_question", "")).strip()
            if not standalone and not value.get("needs_clarification"):
                raise ValueError("resolver returned empty query")
            return {
                **base,
                "standalone_question": standalone or question.strip(),
                "referenced_entities": list(value.get("referenced_entities", []))[:20],
                "referenced_evidence_ids": list(
                    dict.fromkeys(
                        [
                            *context["referenced_evidence_ids"],
                            *value.get("referenced_evidence_ids", []),
                        ]
                    )
                )[:20],
                "needs_clarification": bool(value.get("needs_clarification")),
                "clarification_question": str(value.get("clarification_question", ""))[:500],
                "language": str(value.get("language", language)),
                "resolver_used": True,
            }
        except Exception:
            base.update(self._rule_fallback(question, prior, language))
            base["resolver_fallback"] = True
            return base

    @staticmethod
    def answer_conversation_history(
        conversation: dict[str, Any], *, current_message_id: str = "", limit: int = 20
    ) -> tuple[str, dict[str, Any]]:
        """Answer a conversation-control query from persisted user turns only."""
        user_messages = [
            item
            for item in conversation.get("messages", [])
            if item.get("role") == "user" and item.get("id") != current_message_id
        ]
        total = len(user_messages)
        selected = user_messages[-max(1, limit) :]
        if not user_messages:
            answer = "在本次对话中，你此前还没有提问。"
        else:
            omitted = total - len(selected)
            prefix = f"本次对话中，你此前共问过 {total} 个问题："
            if omitted:
                prefix += f"以下列出最近 {len(selected)} 个（更早的 {omitted} 个已省略）："
            answer = prefix + "\n" + "\n".join(
                f"{index}. {item.get('content', '').strip()}"
                for index, item in enumerate(selected, start=omitted + 1)
            )
        return answer, {
            "intent": "conversation_history",
            "language": "zh-CN",
            "total_user_questions": total,
            "returned_user_questions": len(selected),
            "turn_indexes": [item.get("turn_index") for item in selected],
        }

    @staticmethod
    def answer_paper_learning_memory(
        record: dict[str, Any], *, topic_limit: int = 8, unresolved_limit: int = 5
    ) -> tuple[str, dict[str, Any]]:
        """Render verified paper-level progress without model inference.

        The memory summary is navigation metadata, not evidence for paper facts.
        """
        memory = record.get("memory", {}) if isinstance(record, dict) else {}
        interactions = [
            item
            for item in memory.get("interactions", [])
            if not is_memory_control_query(
                str(item.get("question") or item.get("resolved_question") or "")
            )
        ]
        sections = list(memory.get("explored_sections", []))
        figures = list(memory.get("figure_ids", []))
        tables = list(memory.get("table_ids", []))
        unresolved = [
            item
            for item in memory.get("unresolved_questions", [])
            if not is_memory_control_query(str(item.get("question", item)))
        ]
        if not interactions and not unresolved:
            answer = "你还没有在该论文下形成可复用的跨会话学习记录。"
        else:
            lines = ["以下内容来自你在该论文下的跨会话学习记录，不作为论文事实证据。"]
            if interactions:
                lines.append(f"已完成 {len(interactions)} 次有证据支持的论文问答。")
                displayed = min(len(interactions), max(1, topic_limit))
                heading = "重点研究过的内容："
                if displayed < len(interactions):
                    heading = f"重点研究过的内容（最近 {displayed}/共 {len(interactions)} 项）："
                lines.append(heading)
                selected = interactions[-max(1, topic_limit) :]
                for index, item in enumerate(selected, start=1):
                    question = str(
                        item.get("resolved_question") or item.get("question") or ""
                    ).strip()
                    item_sections = [
                        str(section).strip()
                        for section in item.get("sections", [])
                        if str(section).strip()
                    ]
                    suffix = f"（章节：{'、'.join(item_sections[:3])}）" if item_sections else ""
                    lines.append(f"{index}. {question}{suffix}")
            else:
                lines.append("目前还没有通过证据校验的已完成问答。")
            if sections:
                lines.append(f"已涉及章节：{'、'.join(str(value) for value in sections[:12])}。")
            lines.append(
                f"累计覆盖：{len(sections)} 个章节、{len(figures)} 张 Figure、"
                f"{len(tables)} 张 Table。"
            )
            if unresolved:
                lines.append("尚未解决的问题：")
                for index, item in enumerate(unresolved[-max(1, unresolved_limit) :], start=1):
                    lines.append(f"{index}. {str(item.get('question', item)).strip()}")
            else:
                lines.append("当前没有记录到尚未解决的问题。")
            answer = "\n".join(lines)
        result_ids = list(dict.fromkeys(
            str(item.get("memory_id", ""))
            for item in [*interactions, *unresolved]
            if item.get("memory_id")
        ))[:30]
        if not result_ids and (interactions or unresolved):
            result_ids = [
                f"verified:{len(interactions)}",
                f"unresolved:{len(unresolved)}",
            ]
        details = {
            "intent": "paper_learning_memory",
            "verified_interaction_count": len(interactions),
            "explored_sections": sections,
            "figure_count": len(figures),
            "table_count": len(tables),
            "unresolved_count": len(unresolved),
            "memory_version": record.get("version", 1) if isinstance(record, dict) else 1,
            "result_ids": result_ids,
        }
        return answer, details

    @staticmethod
    def _bounded_paper_memory(memory: dict[str, Any]) -> dict[str, Any]:
        """Expose only navigation fields to resolution; never old answers as evidence."""
        if not isinstance(memory, dict):
            return {}
        return {
            "last_question": str(memory.get("last_question", ""))[:500],
            "last_resolved_question": str(memory.get("last_resolved_question", ""))[:500],
            "explored_sections": list(memory.get("explored_sections", []))[-12:],
            "figure_ids": list(memory.get("figure_ids", []))[-12:],
            "table_ids": list(memory.get("table_ids", []))[-12:],
            "unresolved_questions": list(memory.get("unresolved_questions", []))[-5:],
        }

    @staticmethod
    def _rule_fallback(
        question: str, prior: list[dict], language: str
    ) -> dict[str, Any]:
        last_user = next(
            (item["content"] for item in reversed(prior) if item["role"] == "user"), ""
        )
        if not last_user:
            return {
                "needs_clarification": True,
                "clarification_question": (
                    "请说明你指的是哪张图、哪张表或哪个方法。"
                    if language == "zh-CN"
                    else "Which figure, table, or method do you mean?"
                ),
                "resolver_used": False,
            }
        return {
            "standalone_question": f"基于上一问题“{last_user}”，追问：{question}",
            "resolver_used": False,
        }
