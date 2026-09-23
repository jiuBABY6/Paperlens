from pathlib import Path

from fastapi.testclient import TestClient

from app.domain import Paper
from app.repository import PaperRepository
from app.services.conversation import ConversationService
import app.main as main


class _NoopRunManager:
    def start(self, _run_id, _worker):
        return None


def test_conversation_api_crud_and_idempotent_submit(monkeypatch, tmp_path: Path) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    repository.save(
        Paper("paper-1", "paper.pdf", "Paper", "", str(tmp_path / "paper.pdf"),
              "test", [], []),
        {},
    )
    monkeypatch.setattr(main, "repository", repository)
    monkeypatch.setattr(main, "conversation_service", ConversationService(repository))
    monkeypatch.setattr(main, "run_manager", _NoopRunManager())
    client = TestClient(main.app)

    response = client.post("/api/papers/paper-1/conversations", json={"title": "Test"})
    assert response.status_code == 201
    conversation_id = response.json()["id"]
    assert client.get("/api/papers/paper-1/conversations").json()["items"][0]["id"] == conversation_id

    payload = {"question": "What is the method?", "client_message_id": "browser-0001"}
    first = client.post(
        f"/api/papers/paper-1/conversations/{conversation_id}/messages", json=payload
    )
    second = client.post(
        f"/api/papers/paper-1/conversations/{conversation_id}/messages", json=payload
    )
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["run_id"] == second.json()["run_id"]
    assert second.json()["created"] is False

    renamed = client.patch(
        f"/api/papers/paper-1/conversations/{conversation_id}", json={"title": "Renamed"}
    )
    assert renamed.json()["title"] == "Renamed"
    assert client.get(
        f"/api/papers/paper-2/conversations/{conversation_id}"
    ).status_code == 404

    repository.update_paper_learning_memory("paper-1", {
        "interactions": [{"question": "What is the method?"}],
    })
    learning = client.get("/api/papers/paper-1/learning-memory")
    assert learning.status_code == 200
    assert len(learning.json()["memory"]["interactions"]) == 1
    assert client.delete("/api/papers/paper-1/learning-memory").status_code == 204
    assert client.get("/api/papers/paper-1/learning-memory").json()["memory"] == {}
