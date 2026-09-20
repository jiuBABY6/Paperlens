"""论文精读领域对象与 API 输出对象。"""

from dataclasses import dataclass, field
from typing import Any


EVIDENCE_TYPES = ("text", "figure", "table")


def markdown_table_data(markdown: str) -> tuple[list[str], list[dict[str, str]]]:
    """把 Docling Markdown 表格解析成列名和行对象，解析失败时安全返回空结构。"""
    lines = [line.strip() for line in markdown.splitlines() if "|" in line]
    matrix = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    matrix = [row for row in matrix if row and not all(set(cell) <= {"-", ":", " "} for cell in row)]
    if len(matrix) < 2:
        return [], []
    columns = matrix[0]
    rows = [
        {columns[index] if columns[index] else f"column_{index + 1}": value for index, value in enumerate(row[:len(columns)])}
        for row in matrix[1:]
        if len(row) >= len(columns)
    ]
    return columns, rows


@dataclass
class EvidenceObject:
    """跨文本、图片和表格的统一可追溯证据。"""

    evidence_id: str
    type: str
    page: int
    bbox: tuple[float, float, float, float] | None
    content: str | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in EVIDENCE_TYPES:
            raise ValueError(f"不支持的 evidence type: {self.type}")

    def to_payload(self, score: float | None = None) -> dict[str, Any]:
        """输出 API、Agent Tool 与 PDF.js 共用的契约。"""
        value = {
            "evidence_id": self.evidence_id,
            "id": self.evidence_id,  # 向后兼容旧前端。
            "type": self.type,
            "page": self.page,
            "bbox": self.bbox,
            "content": self.content,
            "text": self.content or self.metadata.get("caption", ""),
            "section": self.section,
            "metadata": self.metadata,
        }
        if score is not None:
            value["score"] = round(float(score), 4)
        return value


@dataclass
class Chunk:
    """可定位到原 PDF 页面区域的证据块。"""

    id: str
    paper_id: str
    page: int
    section: str
    text: str
    bbox: tuple[float, float, float, float] | None
    char_start: int | None = None
    char_end: int | None = None
    sentence_ids: list[str] = field(default_factory=list)


@dataclass
class Sentence:
    """可被 LLM 直接选择并精确高亮的最小原文证据单元。"""

    id: str
    paper_id: str
    chunk_id: str
    page: int
    section: str
    text: str
    bbox: tuple[float, float, float, float] | None
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class Figure:
    """论文中可展示的图或表区域。"""

    id: str
    paper_id: str
    page: int
    kind: str
    caption: str
    bbox: tuple[float, float, float, float] | None
    image_path: str | None
    content: str = ""
    section: str = ""
    nearby_text: str = ""
    related_sentence_ids: list[str] = field(default_factory=list)
    description: dict[str, Any] = field(default_factory=dict)

    @property
    def evidence_type(self) -> str:
        return "table" if self.kind == "table" else "figure"

    def as_evidence(self) -> EvidenceObject:
        """将已有 Figure/Table 适配为统一 EvidenceObject。"""
        metadata: dict[str, Any] = {
            "caption": self.caption,
            "image_path": self.image_path,
            "nearby_text": self.nearby_text,
            "related_sentence_ids": self.related_sentence_ids,
        }
        if self.evidence_type == "figure":
            metadata.update({
                "description": self.description,
                "summary": self.description.get("summary", ""),
            })
        else:
            metadata["markdown"] = self.content
            columns, rows = markdown_table_data(self.content)
            metadata["columns"] = columns
            metadata["rows"] = rows
        return EvidenceObject(
            self.id, self.evidence_type, self.page, self.bbox,
            self.content or None, self.section or None, metadata,
        )


@dataclass
class Paper:
    """持久化论文及其解析后的内容。"""

    id: str
    filename: str
    title: str
    abstract: str
    pdf_path: str
    parser_name: str
    chunks: list[Chunk]
    figures: list[Figure]
    sentences: list[Sentence] = field(default_factory=list)
    content_sha256: str = ""
    status: str = "completed"
    analysis_version: int = 1
    vector_indexed: bool = False
    index_version: int = 0
    index_error: str = ""


def citation_payload(chunk: Chunk, score: float | None = None) -> dict[str, Any]:
    """将证据块转换为前端可直接使用的引用数据。"""
    return {
        "id": chunk.id,
        "evidence_id": chunk.id,
        "type": "text",
        "page": chunk.page,
        "section": chunk.section,
        "text": chunk.text,
        "content": chunk.text,
        "bbox": chunk.bbox,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "score": round(score, 4) if score is not None else None,
        "metadata": {"sentence_ids": chunk.sentence_ids, "chunk_level": True},
    }


def sentence_payload(sentence: Sentence) -> dict[str, Any]:
    """将句子证据转换为卡片和 PDF.js 可直接消费的定位数据。"""
    return {
        "id": sentence.id,
        "evidence_id": sentence.id,
        "type": "text",
        "page": sentence.page,
        "section": sentence.section,
        "text": sentence.text,
        "bbox": sentence.bbox,
        "bboxes": sentence.bboxes or ([sentence.bbox] if sentence.bbox else []),
        "content": sentence.text,
        "metadata": {"chunk_id": sentence.chunk_id},
    }


def sentence_evidence(sentence: Sentence) -> EvidenceObject:
    """保持 sentence_id 不变地适配旧文本证据。"""
    return EvidenceObject(
        sentence.id,
        "text",
        sentence.page,
        sentence.bbox,
        sentence.text,
        sentence.section,
        {"chunk_id": sentence.chunk_id, "bboxes": sentence.bboxes},
    )
