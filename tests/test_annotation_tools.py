"""人工标注辅助工具测试。"""

import json

from app.domain import Chunk, Paper, Sentence
from app.repository import PaperRepository
from scripts.annotate import annotation_records
from scripts.validate_dataset import validate_dataset


def sample_paper() -> Paper:
    chunks = [
        Chunk("chunk-method", "paper-1", 2, "Method", "The model uses an attention encoder.", (0, 10, 100, 30), sentence_ids=["sentence-method"]),
        Chunk("chunk-result", "paper-1", 5, "Results", "Accuracy improves by 3.2 percent.", (0, 40, 100, 60), sentence_ids=["sentence-result"]),
    ]
    sentences = [
        Sentence("sentence-method", "paper-1", "chunk-method", 2, "Method", "The model uses an attention encoder.", (0, 10, 100, 30)),
        Sentence("sentence-result", "paper-1", "chunk-result", 5, "Results", "Accuracy improves by 3.2 percent.", (0, 40, 100, 60)),
    ]
    return Paper("paper-1", "paper.pdf", "Paper", "", "paper.pdf", "test", chunks, [], sentences)


def test_annotation_records_filter_page_keyword_and_unit() -> None:
    paper = sample_paper()

    sentences = annotation_records(paper, page=5, keyword="accuracy")
    chunks = annotation_records(paper, unit="chunk", section="method")

    assert [item["sentence_id"] for item in sentences] == ["sentence-result"]
    assert [item["chunk_id"] for item in chunks] == ["chunk-method"]


def test_valid_dataset_passes_all_evidence_checks(tmp_path) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    repository.save(sample_paper(), None)
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(json.dumps({
        "case_id": "paper1-method-01",
        "paper_id": "paper-1",
        "question": "模型使用了什么编码器？",
        "expected_answer": "模型使用注意力编码器。",
        "answerable": True,
        "expected_chunk_ids": ["chunk-method"],
        "expected_sentence_ids": ["sentence-method"],
        "expected_pages": [2],
        "gold_quotes": ["uses an attention encoder"],
        "tags": ["method"],
        "split": "dev",
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    report = validate_dataset(dataset, repository)

    assert report["valid"] is True
    assert report["errors"] == []


def test_invalid_dataset_reports_unknown_ids_pages_and_quotes(tmp_path) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    repository.save(sample_paper(), None)
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(json.dumps({
        "case_id": "broken",
        "paper_id": "paper-1",
        "question": "错误标注？",
        "expected_answer": "错误。",
        "answerable": True,
        "expected_chunk_ids": ["missing-chunk"],
        "expected_sentence_ids": ["sentence-method"],
        "expected_pages": [99],
        "gold_quotes": ["text that does not exist"],
        "tags": [],
        "split": "test",
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    report = validate_dataset(dataset, repository)

    assert report["valid"] is False
    assert any("Chunk 不属于" in item for item in report["errors"])
    assert any("页码" in item for item in report["errors"])
    assert any("gold_quote" in item for item in report["errors"])


def test_unanswerable_case_requires_empty_gold_evidence(tmp_path) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    repository.save(sample_paper(), None)
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(json.dumps({
        "case_id": "paper1-unanswerable-01",
        "paper_id": "paper-1",
        "question": "论文是否测试了不存在的数据集？",
        "expected_answer": "论文没有提供相关信息。",
        "answerable": False,
        "expected_chunk_ids": ["chunk-method"],
        "expected_sentence_ids": [],
        "expected_pages": [],
        "gold_quotes": [],
        "tags": ["unanswerable"],
        "split": "dev",
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    report = validate_dataset(dataset, repository)

    assert report["valid"] is False
    assert any("Gold Evidence 必须为空" in item for item in report["errors"])


def test_same_paper_cannot_appear_in_dev_and_test(tmp_path) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    repository.save(sample_paper(), None)
    base = {
        "paper_id": "paper-1",
        "expected_answer": "模型使用注意力编码器。",
        "answerable": True,
        "expected_chunk_ids": ["chunk-method"],
        "expected_sentence_ids": ["sentence-method"],
        "expected_pages": [2],
        "gold_quotes": ["uses an attention encoder"],
        "tags": ["method"],
    }
    rows = [
        {**base, "case_id": "dev-case", "question": "开发集问题？", "split": "dev"},
        {**base, "case_id": "test-case", "question": "测试集问题？", "split": "test"},
    ]
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    report = validate_dataset(dataset, repository)

    assert report["valid"] is False
    assert any("同时出现在 dev/test" in item for item in report["errors"])


def test_alternative_chunk_groups_must_match_chunk_union(tmp_path) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    repository.save(sample_paper(), None)
    row = {
        "case_id": "alternative-evidence",
        "paper_id": "paper-1",
        "question": "模型使用了什么编码器？",
        "expected_answer": "模型使用注意力编码器。",
        "answerable": True,
        "expected_chunk_ids": ["chunk-method"],
        "expected_chunk_groups": [["chunk-result"]],
        "expected_sentence_ids": ["sentence-method"],
        "expected_pages": [2],
        "gold_quotes": ["uses an attention encoder"],
        "tags": ["method"],
        "split": "dev",
    }
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")

    report = validate_dataset(dataset, repository)

    assert report["valid"] is False
    assert any("必须等于 expected_chunk_groups 的并集" in item for item in report["errors"])


def test_alternative_sentence_groups_must_match_sentence_union(tmp_path) -> None:
    repository = PaperRepository(tmp_path / "paperlens.sqlite3")
    repository.save(sample_paper(), None)
    row = {
        "case_id": "alternative-sentences",
        "paper_id": "paper-1",
        "question": "模型使用了什么编码器？",
        "expected_answer": "模型使用注意力编码器。",
        "answerable": True,
        "expected_chunk_ids": ["chunk-method", "chunk-result"],
        "expected_sentence_ids": ["sentence-method"],
        "expected_sentence_groups": [["sentence-result"]],
        "expected_pages": [2],
        "gold_quotes": ["uses an attention encoder"],
        "tags": ["method"],
        "split": "dev",
    }
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")

    report = validate_dataset(dataset, repository)

    assert report["valid"] is False
    assert any("必须等于 expected_sentence_groups 的并集" in item for item in report["errors"])
