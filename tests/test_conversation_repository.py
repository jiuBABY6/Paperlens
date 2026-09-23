from pathlib import Path

import pytest

from app.domain import Paper
from app.repository import PaperRepository


def _repository(tmp_path: Path) -> PaperRepository:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    repository.save(
        Paper("paper-1", "paper.pdf", "Paper", "", str(tmp_path / "paper.pdf"),
              "test", [], []),
        {"source": "test"},
    )
    repository.save(
        Paper("paper-2", "other.pdf", "Other", "", str(tmp_path / "other.pdf"),
              "test", [], []),
        {"source": "test"},
    )
    return repository


def test_conversation_crud_is_paper_scoped(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    created = repository.create_conversation("conv-1", "paper-1", "First")
    assert created["title"] == "First"
    assert repository.get_conversation("paper-2", "conv-1") is None
    assert [item["id"] for item in repository.list_conversations("paper-1")] == ["conv-1"]

    renamed = repository.update_conversation("paper-1", "conv-1", title="Renamed")
    assert renamed and renamed["title"] == "Renamed"
    assert repository.archive_conversation("paper-1", "conv-1") is True
    assert repository.list_conversations("paper-1") == []
    assert repository.list_conversations("paper-1", include_archived=True)[0]["archived_at"]


def test_message_submission_is_idempotent_and_turns_are_stable(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.create_conversation("conv-1", "paper-1")
    first_message, first_run, created = repository.create_message_and_run(
        paper_id="paper-1", conversation_id="conv-1", question="What is it?",
        client_message_id="client-0001", message_id="message-1", run_id="run-1",
    )
    assert created is True
    duplicate_message, duplicate_run, created = repository.create_message_and_run(
        paper_id="paper-1", conversation_id="conv-1", question="Changed retry text",
        client_message_id="client-0001", message_id="message-x", run_id="run-x",
    )
    assert created is False
    assert duplicate_message["id"] == first_message["id"]
    assert duplicate_run["id"] == first_run["id"]

    with pytest.raises(RuntimeError, match="active_run"):
        repository.create_message_and_run(
            paper_id="paper-1", conversation_id="conv-1", question="Second",
            client_message_id="client-0002", message_id="message-2", run_id="run-2",
        )

    repository.mark_run_running("paper-1", "conv-1", "run-1")
    repository.finish_conversation_run(
        paper_id="paper-1", conversation_id="conv-1", run_id="run-1",
        status="completed", answer="Answer", resolved_question="Standalone",
        assistant_message_id="assistant-1",
    )
    conversation = repository.get_conversation("paper-1", "conv-1")
    assert conversation
    assert [(m["turn_index"], m["message_index"], m["role"]) for m in conversation["messages"]] == [
        (1, 1, "user"), (1, 2, "assistant")
    ]


def test_startup_interrupts_abandoned_runs(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.create_conversation("conv-1", "paper-1")
    repository.create_message_and_run(
        paper_id="paper-1", conversation_id="conv-1", question="Question",
        client_message_id="client-0001", message_id="message-1", run_id="run-1",
    )
    restarted = PaperRepository(tmp_path / "paperlens.sqlite3")
    run = restarted.get_run("paper-1", "conv-1", "run-1")
    assert run and run["status"] == "interrupted"
    assert run["error"]["code"] == "SERVER_RESTARTED"


def test_paper_learning_memory_is_cross_conversation_but_paper_scoped(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    saved = repository.update_paper_learning_memory("paper-1", {
        "interactions": [{"conversation_id": "conv-a", "question": "Method?"}],
        "explored_sections": ["Method"],
    })
    assert saved["memory"]["explored_sections"] == ["Method"]
    assert repository.get_paper_learning_memory("paper-1")["memory"]["interactions"]
    assert repository.get_paper_learning_memory("paper-2")["memory"] == {}
    assert repository.clear_paper_learning_memory("paper-1") is True
    assert repository.get_paper_learning_memory("paper-1")["memory"] == {}
