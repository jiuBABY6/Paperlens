"""上传 API 的内容去重与文件校验测试。"""

from dataclasses import replace

from fastapi.testclient import TestClient

import app.main as main_module
from app.config import settings
from app.domain import Chunk, Paper
from app.repository import PaperRepository
from app.services.retrieval import SearchResult


class FakeParser:
    def __init__(self, should_fail: bool = False) -> None:
        self.calls = 0
        self.should_fail = should_fail

    def parse(self, paper_id, filename, pdf_path, _paper_dir):
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("simulated parser failure")
        chunk = Chunk(f"{paper_id}-chunk", paper_id, 1, "Abstract", "Evidence text", (0, 0, 1, 1))
        return Paper(paper_id, filename, "Title", "Abstract", str(pdf_path), "fake", [chunk], [])


class FakeRetriever:
    def index(self, _paper):
        return True

    def delete_index(self, _paper_id):
        return None

    def search_with_trace(self, chunks, _query, **_kwargs):
        return [SearchResult(chunks[0], 1.0)], {
            "requested_strategy": "hybrid-rerank",
            "effective_strategy": "hybrid-rerank",
            "degraded": False,
            "warnings": [],
        }

    def close(self):
        return None


class FakeReading:
    def create_card(self, _paper):
        return {"source": "fake", "generation_note": "test"}

    def plan_query(self, question):
        return {"semantic_query": question, "lexical_query": "evidence", "translated": True}

    def answer_with_evidence(self, _question, _chunks, _sentences):
        return {
            "answer": "有证据的回答。",
            "answerable": True,
            "claims": [{"text": "有证据的回答。", "sentence_ids": [], "citations": []}],
            "citations": [],
            "refusal_reason": "",
            "status": "ok",
        }


def configure_test_app(monkeypatch, tmp_path, should_fail: bool = False):
    local_settings = replace(settings, data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    for directory in (local_settings.data_dir, local_settings.upload_dir, local_settings.incoming_dir):
        directory.mkdir(parents=True, exist_ok=True)
    fake_parser = FakeParser(should_fail)
    monkeypatch.setattr(main_module, "settings", local_settings)
    monkeypatch.setattr(main_module, "repository", PaperRepository(local_settings.database_path))
    monkeypatch.setattr(main_module, "parser", fake_parser)
    monkeypatch.setattr(main_module, "retriever", FakeRetriever())
    monkeypatch.setattr(main_module, "reading", FakeReading())
    return fake_parser


def test_same_pdf_is_processed_once(monkeypatch, tmp_path) -> None:
    fake_parser = configure_test_app(monkeypatch, tmp_path)
    payload = b"%PDF-1.7\nidentical test document"
    client = TestClient(main_module.app)

    first = client.post("/api/papers", files={"file": ("paper.pdf", payload, "application/pdf")})
    second = client.post("/api/papers", files={"file": ("renamed.pdf", payload, "application/pdf")})

    assert first.status_code == 200
    assert first.json()["reused"] is False
    assert second.status_code == 200
    assert second.json()["reused"] is True
    assert second.json()["id"] == first.json()["id"]
    assert fake_parser.calls == 1
    assert len(list((tmp_path / "data" / "uploads").glob("*/source.pdf"))) == 1


def test_non_pdf_content_is_rejected_and_cleaned(monkeypatch, tmp_path) -> None:
    configure_test_app(monkeypatch, tmp_path)
    client = TestClient(main_module.app)

    response = client.post(
        "/api/papers",
        files={"file": ("paper.pdf", b"not a PDF", "application/pdf")},
    )

    assert response.status_code == 400
    assert not list((tmp_path / "data" / "incoming").iterdir())


def test_processing_failure_releases_database_and_files(monkeypatch, tmp_path) -> None:
    configure_test_app(monkeypatch, tmp_path, should_fail=True)
    client = TestClient(main_module.app)

    response = client.post(
        "/api/papers",
        files={"file": ("paper.pdf", b"%PDF-1.7\nfailing document", "application/pdf")},
    )

    assert response.status_code == 422
    assert not list((tmp_path / "data" / "incoming").iterdir())
    assert not list((tmp_path / "data" / "uploads").iterdir())
    with main_module.repository._connect() as database:
        assert database.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
        assert database.execute("SELECT COUNT(*) FROM document_registry").fetchone()[0] == 0


def test_reanalysis_increments_version_without_copying_pdf(monkeypatch, tmp_path) -> None:
    fake_parser = configure_test_app(monkeypatch, tmp_path)
    client = TestClient(main_module.app)
    payload = b"%PDF-1.7\nversioned test document"
    created = client.post("/api/papers", files={"file": ("paper.pdf", payload, "application/pdf")})

    response = client.post(f"/api/papers/{created.json()['id']}/reanalyze")

    assert response.status_code == 200
    assert response.json()["analysis_version"] == 2
    assert fake_parser.calls == 2
    assert len(list((tmp_path / "data" / "uploads").glob("*/source.pdf"))) == 1


def test_ask_returns_structured_answer_and_retrieval_trace(monkeypatch, tmp_path) -> None:
    configure_test_app(monkeypatch, tmp_path)
    client = TestClient(main_module.app)
    created = client.post(
        "/api/papers",
        files={"file": ("paper.pdf", b"%PDF-1.7\nquestion document", "application/pdf")},
    )

    response = client.post(
        f"/api/papers/{created.json()['id']}/ask",
        json={"question": "论文方法是什么？"},
    )

    assert response.status_code == 200
    assert response.json()["answerable"] is True
    assert response.json()["query_plan"]["translated"] is True
    assert response.json()["retrieval"]["effective_strategy"] == "hybrid-rerank"
    assert response.json()["latency_ms"] >= 0
