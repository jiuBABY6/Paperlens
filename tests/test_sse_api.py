import asyncio
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.domain import Paper
from app.repository import PaperRepository
from app.services.conversation import ConversationService
from app.services.run_manager import RunManager
import app.main as main


def test_verified_answer_is_published_as_observable_sse_chunks(monkeypatch) -> None:
    class _CaptureManager:
        def __init__(self):
            self.events = []

        async def publish(self, run_id, event, data):
            self.events.append((run_id, event, data))

    manager = _CaptureManager()
    delays = []

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(main, "run_manager", manager)
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(sse_answer_chunk_chars=4, sse_answer_chunk_delay_seconds=0.03),
    )
    monkeypatch.setattr(main.asyncio, "sleep", record_sleep)

    asyncio.run(main._publish_answer("run-1", "abcdefghij"))

    deltas = [data["delta"] for _, event, data in manager.events if event == "answer.delta"]
    assert deltas == ["abcd", "efgh", "ij"]
    assert delays == [0.03, 0.03]
    assert manager.events[0][1] == "answer.started"
    assert manager.events[-1][1] == "answer.completed"


def test_terminal_run_can_be_recovered_as_sse(monkeypatch, tmp_path: Path) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    repository.save(
        Paper("paper-1", "paper.pdf", "Paper", "", str(tmp_path / "paper.pdf"),
              "test", [], []), {},
    )
    service = ConversationService(repository)
    conversation = service.create("paper-1")
    _message, run, _created = service.submit("paper-1", conversation["id"], "Question", "browser-0001")
    repository.mark_run_running("paper-1", conversation["id"], run["id"])
    repository.finish_conversation_run(
        paper_id="paper-1", conversation_id=conversation["id"], run_id=run["id"],
        status="completed", answer="Answer",
    )
    monkeypatch.setattr(main, "repository", repository)
    monkeypatch.setattr(main, "run_manager", RunManager(heartbeat_seconds=0.01))
    client = TestClient(main.app)
    response = client.get(
        f"/api/papers/paper-1/conversations/{conversation['id']}/runs/{run['id']}/events"
    )
    assert response.status_code == 200
    assert "event: run.completed" in response.text
    assert '"recovered": true' in response.text


def test_sse_scope_rejects_wrong_paper(monkeypatch, tmp_path: Path) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    repository.save(Paper("paper-1", "a.pdf", "A", "", "a.pdf", "test", [], []), {})
    repository.save(Paper("paper-2", "b.pdf", "B", "", "b.pdf", "test", [], []), {})
    service = ConversationService(repository)
    conversation = service.create("paper-1")
    _message, run, _created = service.submit("paper-1", conversation["id"], "Question", "browser-0001")
    monkeypatch.setattr(main, "repository", repository)
    client = TestClient(main.app)
    response = client.get(
        f"/api/papers/paper-2/conversations/{conversation['id']}/runs/{run['id']}/events"
    )
    assert response.status_code == 404
