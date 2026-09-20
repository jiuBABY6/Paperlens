"""文件级幂等和重新分析版本的回归测试。"""

from pathlib import Path

from app.domain import Chunk, Figure, Paper, Sentence
from app.repository import PaperRepository


def test_same_hash_is_reserved_only_once(tmp_path: Path) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    digest = "a" * 64

    assert repository.reserve_upload("paper-1", "first.pdf", tmp_path / "first.pdf", digest) is None
    assert repository.reserve_upload("paper-2", "renamed.pdf", tmp_path / "second.pdf", digest) == "paper-1"
    assert repository.get_status("paper-1") == "processing"
    assert repository.get_status("paper-2") is None


def test_different_content_with_same_filename_is_not_duplicate(tmp_path: Path) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")

    assert repository.reserve_upload("paper-1", "paper.pdf", tmp_path / "one.pdf", "a" * 64) is None
    assert repository.reserve_upload("paper-2", "paper.pdf", tmp_path / "two.pdf", "b" * 64) is None


def test_failed_first_upload_releases_hash_for_retry(tmp_path: Path) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    digest = "c" * 64

    assert repository.reserve_upload("failed", "paper.pdf", tmp_path / "failed.pdf", digest) is None
    repository.release_upload("failed", digest)
    assert repository.reserve_upload("retry", "paper.pdf", tmp_path / "retry.pdf", digest) is None


def test_legacy_duplicates_choose_sentence_complete_canonical_record(tmp_path: Path) -> None:
    database_path = tmp_path / "paperlens.sqlite3"
    repository = PaperRepository(database_path)
    old_pdf = tmp_path / "old.pdf"
    new_pdf = tmp_path / "new.pdf"
    old_pdf.write_bytes(b"%PDF-1.7\nsame")
    new_pdf.write_bytes(b"%PDF-1.7\nsame")
    old_chunk = Chunk("old-chunk", "old", 1, "Method", "text", None)
    new_chunk = Chunk("new-chunk", "new", 1, "Method", "text", None, sentence_ids=["new-sentence"])
    repository.save(Paper("old", "paper.pdf", "", "", str(old_pdf), "old", [old_chunk], []), None)
    repository.save(Paper(
        "new", "paper.pdf", "", "", str(new_pdf), "new", [new_chunk], [],
        [Sentence("new-sentence", "new", "new-chunk", 1, "Method", "text", None)],
    ), None)

    PaperRepository(database_path)

    with repository._connect() as database:
        canonical = database.execute("SELECT paper_id FROM document_registry").fetchone()[0]
    assert canonical == "new"


def test_saving_reanalysis_invalidates_figure_query_cache(tmp_path: Path) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    figure = Figure("paper-fig_001", "paper", 1, "picture", "Figure 1", None, None)
    paper = Paper("paper", "paper.pdf", "", "", str(tmp_path / "paper.pdf"), "test", [], [figure])
    repository.save(paper, None)
    repository.save_figure_query_cache("paper", figure.id, "question", {"answerable": True})
    assert repository.get_figure_query_cache("paper", figure.id, "question") is not None

    paper.analysis_version = 2
    repository.save(paper, None)

    assert repository.get_figure_query_cache("paper", figure.id, "question") is None


def test_figure_query_cache_is_versioned_and_can_be_deleted_per_figure(tmp_path: Path) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    figure = Figure("paper-fig_001", "paper", 1, "picture", "Figure 1", None, None)
    other = Figure("paper-fig_002", "paper", 2, "picture", "Figure 2", None, None)
    paper = Paper(
        "paper", "paper.pdf", "", "", str(tmp_path / "paper.pdf"),
        "test", [], [figure, other],
    )
    repository.save(paper, None)
    repository.save_figure_query_cache(
        "paper", figure.id, "question", {"value": "v1"}, cache_version=1
    )
    repository.save_figure_query_cache(
        "paper", other.id, "question", {"value": "other"}, cache_version=1
    )

    assert repository.get_figure_query_cache(
        "paper", figure.id, "question", cache_version=2
    ) is None
    assert repository.delete_figure_query_cache("paper", figure.id) == 1
    assert repository.get_figure_query_cache(
        "paper", other.id, "question", cache_version=1
    ) == {"value": "other"}
