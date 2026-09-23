import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from app.domain import Paper
from app.repository import PaperRepository
from app.services.conversation import ConversationService
from app.services.query_resolver import QueryResolver
from app.services.run_manager import RunManager
import app.main as main


class _Resolver:
    def resolve(self, question, _conversation):
        return {
            "standalone_question": f"Standalone: {question}",
            "resolver_used": True,
            "needs_clarification": False,
            "referenced_evidence_ids": [],
            "context_message_count": 2,
        }


def _paper(tmp_path: Path) -> Paper:
    return Paper(
        "paper-1", "paper.pdf", "Paper", "", str(tmp_path / "paper.pdf"),
        "test", [], [],
    )


def test_conversation_run_persists_verified_answer_and_trace(
    monkeypatch, tmp_path: Path
) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    repository.save(_paper(tmp_path), {})
    service = ConversationService(repository)
    conversation = service.create("paper-1")
    _message, run, _created = service.submit(
        "paper-1", conversation["id"], "它高了多少？", "browser-0001"
    )
    manager = RunManager(buffer_size=50, heartbeat_seconds=0.01)
    captured_spans = []

    @contextmanager
    def capture_span(name, **attributes):
        captured_spans.append((name, attributes))
        yield SimpleNamespace(record_exception=lambda _error: None)

    monkeypatch.setattr(main, "repository", repository)
    monkeypatch.setattr(main, "conversation_service", service)
    monkeypatch.setattr(main, "query_resolver", _Resolver())
    monkeypatch.setattr(main, "run_manager", manager)
    monkeypatch.setattr(main, "span", capture_span)
    monkeypatch.setattr(main, "ask", lambda _paper_id, request: {
        "answer": "高 3.8 个百分点。",
        "status": "ok",
        "answerable": True,
        "evidence_sufficient": True,
        "citations": [{"evidence_id": "table-2", "type": "table", "page": 4}],
        "trace": {
            "steps": [{"step": 1, "tool": "read_table", "result_ids": ["table-2"]}]
        },
        "route": "langgraph_agentic_rag",
        "mode": "multimodal_multi_agent_rag",
    })

    asyncio.run(main._execute_conversation_run("paper-1", conversation["id"], run["id"]))
    persisted = repository.get_run("paper-1", conversation["id"], run["id"])
    timeline = repository.get_conversation("paper-1", conversation["id"])["messages"]
    assert persisted["status"] == "completed"
    assert timeline[-1]["content"] == "高 3.8 个百分点。"
    assert timeline[-1]["resolved_question"].startswith("Standalone")
    assert timeline[-1]["citations"][0]["evidence_id"] == "table-2"
    assert timeline[-1]["metadata"]["trace"]["conversation_id"] == conversation["id"]
    learning = repository.get_paper_learning_memory("paper-1")["memory"]
    assert learning["interactions"][0]["conversation_id"] == conversation["id"]
    assert learning["table_ids"] == ["table-2"]
    root_name, root_attributes = captured_spans[0]
    assert root_name == "paperlens.conversation_run"
    assert root_attributes["run_id"] == run["id"]
    assert root_attributes["conversation_id"] == conversation["id"]
    assert root_attributes["paper_id"] == "paper-1"
    assert root_attributes["question_preview"] == "它高了多少？"
    assert root_attributes["question_length"] == len("它高了多少？")


def test_new_conversation_does_not_receive_other_conversation_history(tmp_path: Path) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    repository.save(_paper(tmp_path), {})
    repository.create_conversation("conv-a", "paper-1")
    repository.create_conversation("conv-b", "paper-1")
    repository.create_message_and_run(
        paper_id="paper-1",
        conversation_id="conv-a",
        question="Secret context",
        client_message_id="browser-a",
        message_id="message-a",
        run_id="run-a",
    )
    assert repository.get_conversation("paper-1", "conv-b")["messages"] == []


def test_conversation_history_query_bypasses_paper_rag(
    monkeypatch, tmp_path: Path
) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    repository.save(_paper(tmp_path), {})
    service = ConversationService(repository)
    conversation = service.create("paper-1")
    _first_message, first_run, _created = service.submit(
        "paper-1", conversation["id"], "论文的主要方法是什么？", "browser-0001"
    )
    repository.finish_conversation_run(
        paper_id="paper-1",
        conversation_id=conversation["id"],
        run_id=first_run["id"],
        status="completed",
        answer="方法回答。",
    )
    _message, history_run, _created = service.submit(
        "paper-1", conversation["id"], "我之前询问过哪些问题", "browser-0002"
    )
    manager = RunManager(buffer_size=50, heartbeat_seconds=0.01)
    reading = SimpleNamespace(settings=SimpleNamespace(deepseek_key=""))
    monkeypatch.setattr(main, "repository", repository)
    monkeypatch.setattr(main, "conversation_service", service)
    monkeypatch.setattr(main, "query_resolver", QueryResolver(reading))
    monkeypatch.setattr(main, "run_manager", manager)
    monkeypatch.setattr(
        main,
        "ask",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("paper RAG must not run for conversation history")
        ),
    )

    asyncio.run(
        main._execute_conversation_run("paper-1", conversation["id"], history_run["id"])
    )

    persisted = repository.get_run("paper-1", conversation["id"], history_run["id"])
    timeline = repository.get_conversation("paper-1", conversation["id"])["messages"]
    assert persisted["status"] == "completed"
    assert "论文的主要方法是什么" in timeline[-1]["content"]
    assert "我之前询问过哪些问题" not in timeline[-1]["content"]
    assert timeline[-1]["metadata"]["response_source"] == "conversation_history"
    assert timeline[-1]["metadata"]["trace"]["steps"][0]["tool"] == "read_conversation_history"


def test_paper_learning_query_bypasses_rag_across_conversations(
    monkeypatch, tmp_path: Path
) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    repository.save(_paper(tmp_path), {})
    repository.update_paper_learning_memory("paper-1", {
        "interactions": [
            {
                "conversation_id": "old",
                "question": "Method?",
                "resolved_question": "Method?",
                "sections": ["Method"],
            }
        ],
        "explored_sections": ["Method"],
        "figure_ids": [],
        "table_ids": [],
        "unresolved_questions": [],
    })
    service = ConversationService(repository)
    conversation = service.create("paper-1")
    _message, run, _created = service.submit(
        "paper-1", conversation["id"], "我对这篇论文了解多少", "browser-memory"
    )
    manager = RunManager(buffer_size=50, heartbeat_seconds=0.01)
    reading = SimpleNamespace(settings=SimpleNamespace(deepseek_key=""))
    monkeypatch.setattr(main, "repository", repository)
    monkeypatch.setattr(main, "conversation_service", service)
    monkeypatch.setattr(main, "query_resolver", QueryResolver(reading))
    monkeypatch.setattr(main, "run_manager", manager)
    monkeypatch.setattr(
        main,
        "ask",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("paper RAG must not run for learning-memory query")
        ),
    )
    asyncio.run(main._execute_conversation_run("paper-1", conversation["id"], run["id"]))
    persisted = repository.get_run("paper-1", conversation["id"], run["id"])
    timeline = repository.get_conversation("paper-1", conversation["id"])["messages"]
    assert persisted["status"] == "completed"
    assert "1 次有证据支持" in timeline[-1]["content"]
    assert timeline[-1]["metadata"]["response_source"] == "paper_learning_memory"


def test_composite_paper_learning_query_bypasses_rag(
    monkeypatch, tmp_path: Path
) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    repository.save(_paper(tmp_path), {})
    repository.update_paper_learning_memory("paper-1", {
        "interactions": [
            {
                "question": "方法的主要步骤是什么？",
                "resolved_question": "方法的主要步骤是什么？",
                "sections": ["Method"],
            }
        ],
        "explored_sections": ["Method"],
        "figure_ids": [],
        "table_ids": [],
        "unresolved_questions": [{"question": "消融实验还缺少什么解释？"}],
    })
    service = ConversationService(repository)
    conversation = service.create("paper-1")
    question = "请总结我之前在这篇论文中重点研究过的内容，并列出尚未解决的问题。"
    _message, run, _created = service.submit(
        "paper-1", conversation["id"], question, "browser-memory-summary"
    )
    manager = RunManager(buffer_size=50, heartbeat_seconds=0.01)
    reading = SimpleNamespace(settings=SimpleNamespace(deepseek_key=""))
    monkeypatch.setattr(main, "repository", repository)
    monkeypatch.setattr(main, "conversation_service", service)
    monkeypatch.setattr(main, "query_resolver", QueryResolver(reading))
    monkeypatch.setattr(main, "run_manager", manager)
    monkeypatch.setattr(
        main,
        "ask",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("paper RAG must not run for learning-memory query")
        ),
    )

    asyncio.run(main._execute_conversation_run("paper-1", conversation["id"], run["id"]))

    persisted = repository.get_run("paper-1", conversation["id"], run["id"])
    timeline = repository.get_conversation("paper-1", conversation["id"])["messages"]
    answer = timeline[-1]["content"]
    assert persisted["status"] == "completed"
    assert "重点研究过的内容" in answer
    assert "方法的主要步骤是什么" in answer
    assert "尚未解决的问题" in answer
    assert "消融实验还缺少什么解释" in answer
    assert timeline[-1]["metadata"]["response_source"] == "paper_learning_memory"


def test_worker_thread_bridges_live_tool_events_to_sse_manager(
    monkeypatch, tmp_path: Path
) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    repository.save(_paper(tmp_path), {})
    service = ConversationService(repository)
    conversation = service.create("paper-1")
    _message, run, _created = service.submit(
        "paper-1", conversation["id"], "What is the method?", "browser-live"
    )
    manager = RunManager(buffer_size=50, heartbeat_seconds=0.01)

    def fake_ask(_paper_id, _request):
        callback = main._run_event_callback.get()
        assert callback is not None
        callback("tool.started", {"tool": "search_text"})
        callback("tool.completed", {"tool": "search_text", "result_ids": ["s1"]})
        return {
            "answer": "Method answer.",
            "status": "ok",
            "answerable": True,
            "evidence_sufficient": True,
            "citations": [],
            "trace": {"steps": []},
            "route": "standard_rag",
            "mode": "standard_rag",
        }

    monkeypatch.setattr(main, "repository", repository)
    monkeypatch.setattr(main, "conversation_service", service)
    monkeypatch.setattr(main, "query_resolver", _Resolver())
    monkeypatch.setattr(main, "run_manager", manager)
    monkeypatch.setattr(main, "ask", fake_ask)
    asyncio.run(main._execute_conversation_run("paper-1", conversation["id"], run["id"]))
    names = [item.event for item in manager._channels[run["id"]].events]
    assert names.index("tool.started") < names.index("tool.completed")
    assert names.index("tool.completed") < names.index("answer.started")
