import json

from app.services.query_resolver import QueryResolver


class _Settings:
    deepseek_key = "test"


class _Reading:
    settings = _Settings()

    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, 0

    def _request(self, *_args, **_kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return json.dumps(self.payload)


def _conversation(messages, paper_memory=None):
    return {
        "summary": "",
        "memory": {},
        "messages": messages,
        "paper_memory": {"version": 1, "memory": paper_memory or {}},
    }


def test_first_turn_skips_model() -> None:
    reading = _Reading()
    result = QueryResolver(reading).resolve(
        "Table 2 的结果是什么？",
        _conversation([
            {"role": "user", "content": "Table 2 的结果是什么？", "citations": []}
        ]),
    )
    assert result["standalone_question"] == "Table 2 的结果是什么？"
    assert result["resolver_used"] is False
    assert reading.calls == 0


def test_conversation_history_question_is_a_control_intent() -> None:
    reading = _Reading()
    conversation = _conversation([
        {
            "id": "m1",
            "role": "user",
            "content": "论文方法是什么？",
            "citations": [],
            "turn_index": 1,
        },
        {
            "id": "m2",
            "role": "assistant",
            "content": "方法回答。",
            "citations": [],
            "turn_index": 1,
        },
        {
            "id": "m3",
            "role": "user",
            "content": "我之前询问过哪些问题",
            "citations": [],
            "turn_index": 2,
        },
    ])

    result = QueryResolver(reading).resolve("我之前询问过哪些问题", conversation)
    answer, details = QueryResolver.answer_conversation_history(
        conversation, current_message_id="m3"
    )

    assert result["intent"] == "conversation_history"
    assert result["resolver_used"] is False
    assert reading.calls == 0
    assert "论文方法是什么" in answer
    assert "我之前询问过哪些问题" not in answer
    assert details["total_user_questions"] == 1


def test_followup_is_resolved_but_history_is_not_evidence() -> None:
    reading = _Reading({
        "standalone_question": "Table 2 中 CNN 的 F1 比 LSTM 高多少？",
        "referenced_entities": ["CNN", "LSTM", "F1", "Table 2"],
        "referenced_evidence_ids": ["table-2"],
        "needs_clarification": False,
        "clarification_question": "",
        "language": "zh-CN",
    })
    conversation = _conversation([
        {"role": "user", "content": "比较 Table 2 中 CNN 和 LSTM。", "citations": []},
        {
            "role": "assistant",
            "content": "CNN 更好。",
            "citations": [{"evidence_id": "table-2"}],
        },
        {"role": "user", "content": "它的 F1 高了多少？", "citations": []},
    ])
    result = QueryResolver(reading).resolve("它的 F1 高了多少？", conversation)
    assert result["resolver_used"] is True
    assert result["standalone_question"].startswith("Table 2")
    assert result["referenced_evidence_ids"] == ["table-2"]
    assert "CNN 更好" not in result["standalone_question"]


def test_resolver_failure_has_bounded_rule_fallback() -> None:
    reading = _Reading(error=TimeoutError())
    conversation = _conversation([
        {"role": "user", "content": "Figure 2 展示什么？", "citations": []},
        {"role": "assistant", "content": "流程图。", "citations": []},
        {"role": "user", "content": "其中橙色框是什么？", "citations": []},
    ])
    result = QueryResolver(reading).resolve("其中橙色框是什么？", conversation)
    assert result["resolver_fallback"] is True
    assert "Figure 2" in result["standalone_question"]


def test_paper_learning_progress_is_a_control_intent() -> None:
    reading = _Reading()
    conversation = _conversation([], {
        "interactions": [{"question": "方法是什么？"}],
        "explored_sections": ["Method"],
        "figure_ids": ["fig-2"],
        "table_ids": [],
        "unresolved_questions": [{"question": "局限是什么？"}],
    })
    result = QueryResolver(reading).resolve("我对这篇论文了解多少", conversation)
    answer, details = QueryResolver.answer_paper_learning_memory(
        conversation["paper_memory"]
    )
    assert result["intent"] == "paper_learning_memory"
    assert reading.calls == 0
    assert "1 次有证据支持" in answer
    assert "Method" in answer
    assert details["unresolved_count"] == 1


def test_composite_learning_summary_is_a_control_intent() -> None:
    reading = _Reading()
    question = "请总结我之前在这篇论文中重点研究过的内容，并列出尚未解决的问题。"
    result = QueryResolver(reading).resolve(question, _conversation([]))

    assert result["intent"] == "paper_learning_memory"
    assert result["resolver_used"] is False
    assert reading.calls == 0


def test_paper_open_questions_remain_a_paper_question() -> None:
    reading = _Reading()
    result = QueryResolver(reading).resolve(
        "这篇论文尚未解决的问题是什么？", _conversation([])
    )

    assert result["intent"] == "paper_question"


def test_memory_control_failures_are_not_relisted_as_unresolved() -> None:
    answer, details = QueryResolver.answer_paper_learning_memory({
        "version": 2,
        "memory": {
            "interactions": [
                {
                    "memory_id": "memory-method",
                    "question": "方法是什么？",
                    "sections": ["Method"],
                },
                {
                    "memory_id": "memory-control",
                    "question": "我对这篇论文了解多少",
                    "sections": [],
                },
            ],
            "explored_sections": ["Method"],
            "figure_ids": [],
            "table_ids": [],
            "unresolved_questions": [
                {
                    "question": "请总结我之前在这篇论文中重点研究过的内容，并列出尚未解决的问题。"
                },
                {"question": "消融实验还有什么未解释？"},
            ],
        },
    })

    assert "请总结我之前" not in answer
    assert "我对这篇论文了解多少" not in answer
    assert "消融实验还有什么未解释" in answer
    assert details["verified_interaction_count"] == 1
    assert details["unresolved_count"] == 1
    assert details["result_ids"] == ["memory-method"]


def test_continue_previous_topic_uses_paper_memory_only_for_query_rewrite() -> None:
    reading = _Reading()
    result = QueryResolver(reading).resolve(
        "继续上次的问题",
        _conversation([], {"last_resolved_question": "Table 2 中谁的 F1 最高？"}),
    )
    assert result["resolver_used"] is True
    assert result["memory_source"] == "paper_learning_memory"
    assert "Table 2" in result["standalone_question"]
