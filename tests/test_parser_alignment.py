"""Docling 结构与 PyMuPDF 坐标段落对齐测试。"""

from app.domain import Chunk, Figure, Paper, Sentence
from app.services.parser import PaperParser


def test_docling_section_is_propagated_to_located_chunk_and_sentence() -> None:
    located = Chunk(
        "chunk-1", "paper", 3, "Front Matter",
        "We introduce an evidence grounded retrieval method.",
        (10, 20, 300, 80), sentence_ids=["sentence-1"],
    )
    sentence = Sentence(
        "sentence-1", "paper", "chunk-1", 3, "Front Matter",
        "We introduce an evidence grounded retrieval method.",
        (10, 20, 300, 80),
    )
    paper = Paper("paper", "paper.pdf", "", "", "paper.pdf", "test", [located], [], [sentence])
    structured = Chunk(
        "docling-1", "paper", 3, "Method",
        "We introduce an evidence grounded retrieval method.",
        (10, 20, 300, 80),
    )

    PaperParser()._align_docling_sections(paper, [structured])

    assert located.section == "Method"
    assert sentence.section == "Method"


def test_figure_context_excludes_inside_text_and_uses_valid_nearby_section() -> None:
    parser = PaperParser()
    figure = Figure(
        "paper-fig_001", "paper", 2, "picture", "Figure 1: Architecture",
        (100, 100, 400, 400), "figure.png", section="but then just gave up. LAUGH",
    )
    sentences = [
        Sentence("inside", "paper", "inside", 2, "but then just gave up. LAUGH", "dialogue inside the figure", (120, 140, 350, 180)),
        Sentence("method-1", "paper", "m1", 2, "2.1 Convolutional Neural Network", "The model has three input features.", (100, 60, 400, 80)),
        Sentence("method-2", "paper", "m2", 2, "2.1 Convolutional Neural Network", "Each feature is encoded by a CNN.", (100, 420, 400, 440)),
        Sentence("method-3", "paper", "m3", 2, "2.1 Convolutional Neural Network", "The outputs form high-level vectors.", (100, 450, 400, 470)),
    ]

    parser._enrich_figures([figure], sentences)

    assert figure.section == "2.1 Convolutional Neural Network"
    assert "inside" not in figure.related_sentence_ids
    assert "dialogue inside the figure" not in figure.nearby_text


def test_table_section_uses_results_caption_or_overlapping_table_text() -> None:
    parser = PaperParser()
    results_table = Figure(
        "table-1", "paper", 4, "table", "Table 1: Results, percentage.",
        (100, 50, 500, 140), "table-1.png", "| A | F1 |", section="3.1 Corpus",
    )
    comparison_table = Figure(
        "table-2", "paper", 4, "table", "Table 2: CNN versus LSTM.",
        (80, 170, 290, 215), "table-2.png", "| CNN | 62.9 |", section="3.1 Corpus",
    )
    sentences = [
        Sentence("t1", "paper", "c1", 4, "2.2 Model", "table one cells", (120, 60, 480, 130)),
        Sentence("t2", "paper", "c2", 4, "2.2 Model", "table two cells", (90, 175, 280, 210)),
        Sentence("r1", "paper", "c3", 4, "3.3 Results and discussion", "Results are shown in Table 1.", (310, 300, 540, 330)),
    ]

    parser._enrich_figures([results_table, comparison_table], sentences)

    assert results_table.section == "3.3 Results and discussion"
    assert comparison_table.section == "2.2 Model"
