from dataclasses import replace
from pathlib import Path

from app.config import settings
from app.domain import Paper
from app.repository import PaperRepository
from app.services.long_term_memory import LongTermMemoryService
from fastapi.testclient import TestClient
import app.main as main


def _service(tmp_path: Path, *, version: int = 1):
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    paper = Paper(
        "paper-1", "paper.pdf", "Paper", "", str(tmp_path / "paper.pdf"),
        "test", [], [], analysis_version=version,
    )
    repository.save(paper, {})
    local_settings = replace(
        settings, memory_retention_days=30, memory_max_items_per_paper=10
    )
    return repository, LongTermMemoryService(repository, local_settings)


def _record(service, run_id="run-1", *, status="completed", answerable=True):
    return service.record_run(
        paper_id="paper-1",
        conversation_id="conversation-1",
        run={"id": run_id, "original_question": "What is the main method?"},
        resolved_question="What is the paper's main method?",
        status=status,
        result={
            "answerable": answerable,
            "citations": ([{
                "evidence_id": f"evidence-{run_id}", "type": "text", "section": "Method"
            }] if answerable else []),
        },
    )


def test_only_grounded_completed_turn_becomes_active_memory(tmp_path: Path) -> None:
    repository, service = _service(tmp_path)
    active = _record(service)
    unresolved = _record(service, "run-2", status="partial", answerable=False)

    assert active["status"] == "active"
    assert active["confidence"] == 1.0
    assert active["confidence"] > unresolved["confidence"]
    assert unresolved["status"] == "unresolved"
    aggregate = repository.get_paper_learning_memory("paper-1")["memory"]
    assert [item["source_run_id"] for item in aggregate["interactions"]] == ["run-1"]
    assert aggregate["unresolved_questions"][0]["memory_id"] == unresolved["id"]


def test_control_queries_are_excluded_from_learning_aggregate(tmp_path: Path) -> None:
    repository, service = _service(tmp_path)
    _record(service)
    service.record_run(
        paper_id="paper-1",
        conversation_id="conversation-old",
        run={"id": "run-control", "original_question": "我之前询问过哪些问题"},
        resolved_question="我之前询问过哪些问题",
        status="partial",
        result={"answerable": False, "citations": []},
    )

    aggregate = repository.get_paper_learning_memory("paper-1")["memory"]
    assert len(aggregate["interactions"]) == 1
    assert aggregate["unresolved_questions"] == []
    assert aggregate["last_question"] == "What is the main method?"


def test_memory_write_is_idempotent_and_user_can_edit_or_forget(tmp_path: Path) -> None:
    repository, service = _service(tmp_path)
    first = _record(service)
    second = _record(service)
    assert first["id"] == second["id"]
    assert len(repository.list_memory_items("paper-1", include_inactive=True)) == 1

    edited = service.update_item(
        "paper-1", first["id"], pinned=True, user_note="Focus on this method."
    )
    assert edited and edited["pinned"] is True
    assert edited["user_note"] == "Focus on this method."
    unpinned = service.update_item("paper-1", first["id"], pinned=False)
    assert unpinned and unpinned["pinned"] is False
    assert unpinned["expires_at"] is not None
    forgotten = service.update_item("paper-1", first["id"], status="forgotten")
    assert forgotten and forgotten["status"] == "forgotten"
    assert service.list_items("paper-1") == []


def test_reanalysis_invalidates_even_pinned_memory(tmp_path: Path) -> None:
    _repository, service = _service(tmp_path)
    item = _record(service)
    service.update_item("paper-1", item["id"], pinned=True)

    assert service.invalidate_for_version("paper-1", 2) == 1
    stale = service.list_items("paper-1", include_inactive=True)[0]
    assert stale["status"] == "stale"
    assert stale["pinned"] is True


def test_expiration_archives_unpinned_but_not_pinned_memory(tmp_path: Path) -> None:
    repository, service = _service(tmp_path)
    first = _record(service, "run-1")
    second = _record(service, "run-2")
    service.update_item("paper-1", second["id"], pinned=True)
    with repository._connect() as db:
        db.execute(
            "UPDATE paper_memory_items SET expires_at = '2000-01-01 00:00:00' "
            "WHERE paper_id = 'paper-1'"
        )

    assert service.apply_retention() == 1
    by_id = {
        item["id"]: item for item in service.list_items("paper-1", include_inactive=True)
    }
    assert by_id[first["id"]]["status"] == "archived"
    assert by_id[second["id"]]["status"] == "active"


def test_legacy_aggregate_is_preserved_until_items_are_backfilled(tmp_path: Path) -> None:
    repository, service = _service(tmp_path)
    repository.update_paper_learning_memory("paper-1", {
        "interactions": [{"question": "Legacy question"}],
        "explored_sections": ["Introduction"],
    })

    value = service.refresh_aggregate("paper-1")
    assert value["memory"]["interactions"][0]["question"] == "Legacy question"


def test_memory_item_api_lists_edits_and_forgets(monkeypatch, tmp_path: Path) -> None:
    repository, service = _service(tmp_path)
    item = _record(service)
    monkeypatch.setattr(main, "repository", repository)
    client = TestClient(main.app)

    listed = client.get("/api/papers/paper-1/learning-memory/items")
    assert listed.status_code == 200
    assert listed.json()["items"][0]["id"] == item["id"]

    edited = client.patch(
        f"/api/papers/paper-1/learning-memory/items/{item['id']}",
        json={"pinned": True, "user_note": "Important contribution"},
    )
    assert edited.status_code == 200
    assert edited.json()["pinned"] is True
    assert edited.json()["user_note"] == "Important contribution"

    forgotten = client.patch(
        f"/api/papers/paper-1/learning-memory/items/{item['id']}",
        json={"status": "forgotten"},
    )
    assert forgotten.status_code == 200
    assert client.get(
        "/api/papers/paper-1/learning-memory/items"
    ).json()["items"] == []
