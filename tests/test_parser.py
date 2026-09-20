from app.domain import Figure
from app.services.parser import PaperParser


def test_limitations_heading_is_recognized() -> None:
    """Limitations 必须被识别为独立章节，防止混入方法概览。"""
    parser = PaperParser()
    assert parser._heading("6 Limitations") == "Limitations"
    assert parser._heading("Limitations and Future Work") == "Limitations"


def test_docling_table_content_becomes_retrievable_evidence() -> None:
    parser = PaperParser()
    figure = Figure(
        "figure-1", "paper", 7, "table", "Table 1: Results",
        (10, 20, 300, 400), None, "| Model | F1 |\n|---|---|\n| Ours | 72.5 |",
    )

    chunks, sentences = parser._table_evidence([figure], "paper")

    assert len(chunks) == 1
    assert chunks[0].section.startswith("Table:")
    assert "72.5" in chunks[0].text
    assert chunks[0].sentence_ids == [sentences[0].id]
    assert sentences[0].bbox == figure.bbox
