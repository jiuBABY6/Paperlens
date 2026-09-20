"""论文解析：优先使用 Docling 原生结构，失败时降级为 PyMuPDF。"""

from difflib import SequenceMatcher
from collections import Counter
from pathlib import Path
import re

from app.domain import Chunk, Figure, Paper, Sentence


HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?"
    r"(abstract|introduction|related work|method(?:ology)?|approach|"
    r"experiments?|results?|discussion|conclusion|limitations?|future work|references)\b",
    re.I,
)
CAPTION = re.compile(r"^(?:figure|fig\.?|table)\s*\d+[:.]?\s*(.+)", re.I)
GENERIC_HEADING = re.compile(r"^(?:[A-Z]|\d+(?:\.\d+)*)\s+[A-Z][A-Za-z][^.!?]{0,110}$")
TEXT_LABELS = {"text", "list_item", "formula", "caption", "footnote"}


class PaperParser:
    """把 PDF 转换为可追溯的段落和图表对象。"""

    def __init__(self, backend: str = "docling", models_dir: Path | None = None) -> None:
        """保存解析后端和 Docling 本地模型目录。"""
        self.backend = backend.lower()
        self.models_dir = models_dir

    def parse(self, paper_id: str, filename: str, pdf_path: Path, output_dir: Path) -> Paper:
        """按配置解析论文；Docling 出错时保证仍可返回带页码的降级结果。"""
        if self.backend == "pymupdf":
            return self._parse_with_pymupdf(paper_id, filename, pdf_path, output_dir)
        if self.backend != "docling":
            raise ValueError("PARSER_BACKEND 仅支持 docling 或 pymupdf")
        try:
            return self._parse_with_docling(paper_id, filename, pdf_path, output_dir)
        except Exception as error:
            paper = self._parse_with_pymupdf(paper_id, filename, pdf_path, output_dir)
            paper.parser_name = f"pymupdf（Docling 降级：{type(error).__name__}）"
            return paper

    def _parse_with_docling(self, paper_id: str, filename: str, pdf_path: Path, output_dir: Path) -> Paper:
        """从 Docling 的原生节点、provenance 和图表对象构建论文数据。"""
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions(
            artifacts_path=self.models_dir,
            do_table_structure=True,
            do_ocr=True,
            generate_picture_images=True,
            generate_table_images=True,
        )
        # Windows 中文系统下 torch.compile 可能以 GBK 读取 PyTorch 内核文件而失败。
        options.layout_options.engine_options.compile_model = False
        converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)
        })
        document = converter.convert(str(pdf_path)).document
        # 句子级证据由 PyMuPDF 生成，确保每条 evidence 都有可渲染的 PDF bbox。
        text_paper = self._parse_with_pymupdf(paper_id, filename, pdf_path, output_dir, extract_figures=False)
        structural_chunks = self._docling_chunks(document, paper_id)
        self._align_docling_sections(text_paper, structural_chunks)
        chunks = text_paper.chunks
        figures = self._docling_figures(document, paper_id, output_dir)
        self._enrich_figures(figures, text_paper.sentences)
        table_chunks, table_sentences = self._table_evidence(figures, paper_id)
        chunks.extend(table_chunks)
        text_paper.sentences.extend(table_sentences)
        if not chunks:
            raise RuntimeError("Docling 未提取到可用文本节点")
        title = self._docling_title(structural_chunks or chunks, filename)
        abstract = self._abstract_from_chunks(structural_chunks or chunks)
        return Paper(paper_id, filename, title, abstract, str(pdf_path), "docling-structure+pymupdf-coordinates", chunks, figures, text_paper.sentences)

    def _table_evidence(
        self,
        figures: list[Figure],
        paper_id: str,
        max_chars: int = 1400,
    ) -> tuple[list[Chunk], list[Sentence]]:
        """把 Docling 表格 Markdown 转成可检索、可定位到表格区域的证据块。"""
        chunks: list[Chunk] = []
        sentences: list[Sentence] = []
        table_index = 0
        for figure in figures:
            if figure.kind != "table" or not figure.content.strip():
                continue
            table_index += 1
            prefix = f"{figure.caption}\n"
            parts: list[str] = []
            buffer = prefix
            for line in figure.content.splitlines():
                clean = line.strip()
                if not clean:
                    continue
                addition = clean + "\n"
                if len(buffer) + len(addition) > max_chars and buffer.strip() != prefix.strip():
                    parts.append(buffer.strip())
                    buffer = prefix + addition
                else:
                    buffer += addition
            if buffer.strip() and buffer.strip() != prefix.strip():
                parts.append(buffer.strip())
            for part_index, text in enumerate(parts, start=1):
                chunk_id = f"{paper_id}-table{table_index}-p{part_index}"
                sentence_id = f"{chunk_id}-s0"
                section = f"Table: {figure.caption}"
                sentences.append(Sentence(
                    sentence_id,
                    paper_id,
                    chunk_id,
                    figure.page,
                    section,
                    text,
                    figure.bbox,
                    [figure.bbox] if figure.bbox else [],
                ))
                chunks.append(Chunk(
                    chunk_id,
                    paper_id,
                    figure.page,
                    section,
                    text,
                    figure.bbox,
                    sentence_ids=[sentence_id],
                ))
        return chunks, sentences

    def _align_docling_sections(self, paper: Paper, structural_chunks: list[Chunk]) -> None:
        """用同页文本和 bbox 将 Docling 章节标签映射到 PyMuPDF 坐标段落。"""
        candidates_by_page: dict[int, list[Chunk]] = {}
        for item in structural_chunks:
            candidates_by_page.setdefault(item.page, []).append(item)
        sentence_map = {sentence.id: sentence for sentence in paper.sentences}
        for chunk in paper.chunks:
            candidates = candidates_by_page.get(chunk.page, [])
            if not candidates:
                continue
            score, match = max(
                ((self._alignment_score(chunk, candidate), candidate) for candidate in candidates),
                key=lambda value: value[0],
            )
            if score < 0.35:
                continue
            chunk.section = match.section
            for sentence_id in chunk.sentence_ids:
                if sentence_id in sentence_map:
                    sentence_map[sentence_id].section = match.section

    def _alignment_score(self, located: Chunk, structured: Chunk) -> float:
        """组合文本相似度与区域重叠度，降低双栏页面错配概率。"""
        left = re.sub(r"\W+", "", located.text.lower())[:1200]
        right = re.sub(r"\W+", "", structured.text.lower())[:1200]
        text_score = SequenceMatcher(None, left, right).ratio() if left and right else 0.0
        bbox_score = self._bbox_overlap(located.bbox, structured.bbox)
        return 0.75 * text_score + 0.25 * bbox_score

    def _bbox_overlap(self, first, second) -> float:
        """计算两个 PDF 矩形的交并比。"""
        if not first or not second:
            return 0.0
        x0, y0 = max(first[0], second[0]), max(first[1], second[1])
        x1, y1 = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    def _docling_chunks(self, document, paper_id: str) -> list[Chunk]:
        """将 Docling 文本节点转成段落，并保留章节层级与原生 bbox。"""
        chunks: list[Chunk] = []
        section = "Front Matter"
        for index, (item, _level) in enumerate(document.iterate_items()):
            label = self._label(item)
            raw_text = getattr(item, "text", "")
            whole_text = re.sub(r"\s+", " ", raw_text).strip()
            if label == "section_header" and whole_text:
                section = whole_text
            if label not in TEXT_LABELS and label != "section_header":
                continue
            provenance = getattr(item, "prov", [])
            if not provenance:
                continue
            for prov_index, prov in enumerate(provenance):
                # 一个 Docling 节点可能跨列或跨页。必须按 charspan 切片，不能将完整节点
                # 重复绑定到每个 bbox，否则卡片结论会和高亮区域错位。
                char_start, char_end = getattr(prov, "charspan", (0, len(raw_text)))
                text = re.sub(r"\s+", " ", raw_text[char_start:char_end]).strip()
                if not text or (label != "section_header" and len(text) < 20):
                    continue
                page = int(prov.page_no)
                bbox = self._docling_bbox(document, page, prov.bbox)
                suffix = f"{index}-{prov_index}"
                chunks.append(Chunk(
                    f"{paper_id}-d{suffix}", paper_id, page, section, text, bbox,
                    int(char_start), int(char_end),
                ))
        return chunks

    def _docling_figures(self, document, paper_id: str, output_dir: Path) -> list[Figure]:
        """导出 Docling 图片和表格渲染图，并保留其标题与 bbox。"""
        figure_dir = output_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        figures: list[Figure] = []
        section_map = self._docling_item_sections(document)
        for kind, items in (("picture", document.pictures), ("table", document.tables)):
            for index, item in enumerate(items, start=1):
                provenance = getattr(item, "prov", [])
                if not provenance:
                    continue
                prov = provenance[0]
                page = int(prov.page_no)
                bbox = self._docling_bbox(document, page, prov.bbox)
                caption = item.caption_text(document).strip() or f"{kind.title()} {index}"
                image_path = self._save_docling_image(item, document, figure_dir, kind, index)
                prefix = "fig" if kind == "picture" else "table"
                figures.append(Figure(
                    f"{paper_id}-{prefix}_{index:03d}", paper_id, page, kind, caption, bbox, image_path,
                    self._table_content(item, document) if kind == "table" else "",
                    section=section_map.get(self._docling_item_key(item), ""),
                ))
        return figures

    def _docling_item_sections(self, document) -> dict[str, str]:
        """按 Docling 文档阅读顺序把 Figure/Table 绑定到最近的有效章节标题。"""
        sections: dict[str, str] = {}
        section = "Front Matter"
        for item, _level in document.iterate_items():
            label = self._label(item)
            text = re.sub(r"\s+", " ", str(getattr(item, "text", "") or "")).strip()
            if label == "section_header" and self._valid_section(text):
                section = text
            if label in {"picture", "table"}:
                sections[self._docling_item_key(item)] = section
        return sections

    def _docling_item_key(self, item) -> str:
        """优先使用跨容器稳定的 self_ref，兼容 Docling 返回对象副本。"""
        reference = getattr(item, "self_ref", None)
        return str(reference) if reference is not None else f"object:{id(item)}"

    def _enrich_figures(self, figures: list[Figure], sentences: list[Sentence]) -> None:
        """关联 Figure 外部的邻近正文，并避免图中文字被误判成章节标题。"""
        by_page: dict[int, list[Sentence]] = {}
        for sentence in sentences:
            by_page.setdefault(sentence.page, []).append(sentence)
        for figure in figures:
            page_candidates = by_page.get(figure.page, [])
            if figure.kind == "table":
                inferred_section = self._infer_table_section(figure, page_candidates)
                if inferred_section:
                    figure.section = inferred_section
            candidates = [
                item for item in page_candidates
                if not CAPTION.match(item.text.strip())
                and self._bbox_coverage(item.bbox, figure.bbox) < 0.25
            ]
            candidates = candidates or page_candidates
            if not candidates:
                continue
            ranked = sorted(
                candidates,
                key=lambda item: (
                    -self._context_overlap(figure.caption, item.text),
                    self._vertical_distance(item.bbox, figure.bbox),
                ),
            )[:8]
            if not self._valid_section(figure.section):
                nearby_sections = [item.section for item in ranked if self._valid_section(item.section)]
                page_sections = [item.section for item in page_candidates if self._valid_section(item.section)]
                choices = nearby_sections or page_sections
                figure.section = Counter(choices).most_common(1)[0][0] if choices else "Front Matter"
            same_section = [item for item in ranked if item.section == figure.section]
            if len(same_section) >= 3:
                ranked = same_section[:8]
            ranked.sort(key=lambda item: (item.bbox[1] if item.bbox else 0, item.id))
            figure.related_sentence_ids = [item.id for item in ranked]
            figure.nearby_text = " ".join(item.text for item in ranked)[:4000]

    def _infer_table_section(self, table: Figure, page_sentences: list[Sentence]) -> str | None:
        """Infer a table section despite multi-column floating-object ordering."""
        valid = [item for item in page_sentences if self._valid_section(item.section)]
        if not valid:
            return None
        # A results-labelled floating table can precede its owning heading in PDF
        # reading order, so use the explicit caption signal when it is available.
        if re.search(r"\bresults?\b", table.caption, re.I):
            result_sections = [
                item.section for item in valid
                if re.search(r"\bresults?\b", item.section, re.I)
            ]
            if result_sections:
                return Counter(result_sections).most_common(1)[0][0]
        # Otherwise, the section carried by text geometrically inside the table
        # is more reliable than Docling's global traversal order on two columns.
        inside_sections = [
            item.section for item in valid
            if not CAPTION.match(item.text.strip())
            and self._bbox_coverage(item.bbox, table.bbox) >= 0.5
        ]
        return Counter(inside_sections).most_common(1)[0][0] if inside_sections else None

    def _valid_section(self, section: str) -> bool:
        """拒绝被 Docling/PyMuPDF 误标为章节的图内对白和普通句子。"""
        value = re.sub(r"\s+", " ", section or "").strip()
        if not value:
            return False
        if value == "Front Matter":
            return True
        if len(value) > 160 or re.search(r"[!?]|\.(?:\s|$)", value):
            return False
        if HEADING.match(value):
            return True
        return bool(GENERIC_HEADING.match(value))

    def _bbox_coverage(self, inner, outer) -> float:
        """计算句子区域有多少比例落在 Figure/Table 内部。"""
        if not inner or not outer:
            return 0.0
        x0, y0 = max(inner[0], outer[0]), max(inner[1], outer[1])
        x1, y1 = min(inner[2], outer[2]), min(inner[3], outer[3])
        intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
        return intersection / area if area else 0.0

    def _vertical_distance(self, first, second) -> float:
        """计算正文与图表边缘的纵向距离；重叠区域距离为零。"""
        if not first or not second:
            return float("inf")
        if first[3] < second[1]:
            return second[1] - first[3]
        if first[1] > second[3]:
            return first[1] - second[3]
        return 0.0

    def _context_overlap(self, caption: str, text: str) -> int:
        """用图注中的科研术语提升真正相关正文，空间距离仅作为次级排序。"""
        stop = {
            "figure", "fig", "table", "framework", "diagram", "results", "result",
            "the", "and", "are", "for", "with", "from", "this", "that", "each",
        }
        caption_terms = {
            item for item in re.findall(r"[a-z][a-z0-9_-]{2,}|\d+(?:\.\d+)?", caption.lower())
            if item not in stop
        }
        text_terms = set(re.findall(r"[a-z][a-z0-9_-]{2,}|\d+(?:\.\d+)?", text.lower()))
        return len(caption_terms & text_terms)

    def _save_docling_image(self, item, document, output_dir: Path, kind: str, index: int) -> str | None:
        """将 Docling 生成的 PIL 图片写入论文私有目录。"""
        image = item.get_image(document)
        if image is None:
            return None
        path = output_dir / f"{kind}-{index}.png"
        image.save(path, format="PNG")
        return str(path)

    def _table_content(self, item, document) -> str:
        """导出 Docling 表格的 Markdown，供后续图表精读与问答引用。"""
        try:
            return item.export_to_markdown(document)[:6000]
        except Exception:
            return ""

    def _docling_bbox(self, document, page_number: int, bbox) -> tuple[float, float, float, float]:
        """将 Docling 的左下角坐标转换为 PDF.js 使用的左上角坐标。"""
        page = document.pages[page_number]
        height = float(page.size.height)
        return (
            round(float(bbox.l), 1),
            round(height - float(bbox.t), 1),
            round(float(bbox.r), 1),
            round(height - float(bbox.b), 1),
        )

    def _label(self, item) -> str:
        """兼容不同 Docling 版本的标签枚举表示。"""
        label = getattr(item, "label", "")
        return str(getattr(label, "value", label)).lower()

    def _docling_title(self, chunks: list[Chunk], filename: str) -> str:
        """优先采用第一页第一个章节标题作为论文标题。"""
        for chunk in chunks:
            if chunk.page == 1 and len(chunk.text) > 12 and "abstract" not in chunk.text.lower():
                return chunk.text
        return Path(filename).stem

    def _abstract_from_chunks(self, chunks: list[Chunk]) -> str:
        """根据 Abstract 标题后的连续证据块提取摘要。"""
        collected: list[str] = []
        in_abstract = False
        for chunk in chunks:
            if HEADING.match(chunk.text) and chunk.text.lower().startswith("abstract"):
                in_abstract = True
                continue
            if in_abstract and HEADING.match(chunk.text):
                break
            if in_abstract:
                collected.append(chunk.text)
        return " ".join(collected)[:1800] or "未识别摘要。"

    def _parse_with_pymupdf(self, paper_id: str, filename: str, pdf_path: Path, output_dir: Path, extract_figures: bool = True) -> Paper:
        """Docling 不可用时，以 PyMuPDF 保障基本文本、页码和图片能力。"""
        import fitz

        document = fitz.open(pdf_path)
        chunks: list[Chunk] = []
        sentences: list[Sentence] = []
        figures: list[Figure] = []
        section = "Front Matter"
        all_text: list[str] = []
        figure_dir = output_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        for page_index, page in enumerate(document, start=1):
            blocks = sorted(page.get_text("blocks"), key=lambda item: (round(item[1] / 10), item[0]))
            page_text = "\n".join(block[4].strip() for block in blocks if block[4].strip())
            all_text.append(page_text)
            words_by_block: dict[int, list[tuple]] = {}
            for word in page.get_text("words"):
                words_by_block.setdefault(int(word[5]), []).append(word)
            for block_index, block in enumerate(blocks):
                text = re.sub(r"\s+", " ", block[4]).strip()
                if len(text) < 25:
                    continue
                heading = self._heading(text) or self._generic_heading(text)
                if heading:
                    section = heading
                chunk_id = f"{paper_id}-p{page_index}-b{block_index}"
                block_sentences = self._sentences_from_words(
                    words_by_block.get(int(block[5]), []), paper_id, chunk_id, page_index, section
                )
                # 无可用词坐标时保留整块作为一个可定位句子，避免证据链断裂。
                if not block_sentences:
                    block_sentences = [Sentence(f"{chunk_id}-s0", paper_id, chunk_id, page_index, section, text, tuple(round(value, 1) for value in block[:4]))]
                sentences.extend(block_sentences)
                chunks.append(Chunk(chunk_id, paper_id, page_index, section, text, tuple(round(value, 1) for value in block[:4]), sentence_ids=[item.id for item in block_sentences]))
            if extract_figures:
                figures.extend(self._extract_images(document, page, page_index, paper_id, figure_dir, blocks))
        title = self._title(chunks, filename)
        abstract = self._abstract("\n".join(all_text), "")
        self._enrich_figures(figures, sentences)
        return Paper(paper_id, filename, title, abstract, str(pdf_path), "pymupdf-sentence", chunks, figures, sentences)

    def _sentences_from_words(self, words, paper_id: str, chunk_id: str, page: int, section: str) -> list[Sentence]:
        """按 PyMuPDF 单词坐标切句，为每一句计算独立 bbox。"""
        if not words:
            return []
        words = sorted(words, key=lambda item: (item[6], item[7], item[0]))
        result: list[Sentence] = []
        buffer: list[tuple] = []
        for word in words:
            buffer.append(word)
            if re.search(r"[.!?。！？][\"'\)\]]*$", word[4]):
                result.append(self._build_sentence(buffer, paper_id, chunk_id, page, section, len(result)))
                buffer = []
        if buffer:
            result.append(self._build_sentence(buffer, paper_id, chunk_id, page, section, len(result)))
        return result

    def _build_sentence(self, words, paper_id: str, chunk_id: str, page: int, section: str, index: int) -> Sentence:
        """按行保存句子矩形，避免跨行句子被一个大框覆盖无关内容。"""
        text = " ".join(word[4] for word in words)
        by_line: dict[int, list[tuple]] = {}
        for word in words:
            by_line.setdefault(int(word[6]), []).append(word)
        bboxes = [
            (round(min(word[0] for word in line), 1), round(min(word[1] for word in line), 1), round(max(word[2] for word in line), 1), round(max(word[3] for word in line), 1))
            for _line, line in sorted(by_line.items())
        ]
        bbox = (round(min(box[0] for box in bboxes), 1), round(min(box[1] for box in bboxes), 1), round(max(box[2] for box in bboxes), 1), round(max(box[3] for box in bboxes), 1))
        return Sentence(f"{chunk_id}-s{index}", paper_id, chunk_id, page, section, text, bbox, bboxes)

    def _extract_images(self, document, page, page_number: int, paper_id: str, output_dir: Path, blocks) -> list[Figure]:
        """降级路径下导出足够大的嵌入位图并关联最近的图注。"""
        figures: list[Figure] = []
        for index, image in enumerate(page.get_images(full=True), start=1):
            data = document.extract_image(image[0])
            if int(data.get("width", 0)) * int(data.get("height", 0)) < 40_000:
                continue
            path = output_dir / f"page-{page_number}-image-{index}.{data.get('ext', 'png')}"
            path.write_bytes(data["image"])
            rects = page.get_image_rects(image[0])
            bbox = tuple(rects[0]) if rects else None
            figures.append(Figure(
                f"{paper_id}-fig_{len(figures) + 1:03d}", paper_id, page_number, "picture",
                self._nearest_caption(blocks, bbox), bbox, str(path)
            ))
        return figures

    def _nearest_caption(self, blocks, bbox) -> str:
        """选择与图片纵向距离最近的图表标题。"""
        captions = [
            (re.sub(r"\s+", " ", block[4]).strip(), block[1]) for block in blocks
            if CAPTION.match(re.sub(r"\s+", " ", block[4]).strip())
        ]
        if not captions:
            return "图表标题未识别"
        if bbox is None:
            return captions[0][0]
        return min(captions, key=lambda item: abs(item[1] - bbox[3]))[0]

    def _heading(self, text: str) -> str | None:
        """识别常见英文学术章节标题，供 PyMuPDF 降级路径使用。"""
        match = HEADING.match(text)
        return match.group(1).title() if match else None

    def _generic_heading(self, text: str) -> str | None:
        """识别附录和编号短标题，防止后续正文错误继承前一章节标签。"""
        return text if GENERIC_HEADING.match(text) else None

    def _title(self, chunks: list[Chunk], filename: str) -> str:
        """从首页较长文本块推断标题。"""
        for chunk in chunks:
            if chunk.page == 1 and 12 < len(chunk.text) < 250 and not self._heading(chunk.text):
                return chunk.text
        return Path(filename).stem

    def _abstract(self, text: str, fallback: str) -> str:
        """从降级文本中截取摘要。"""
        match = re.search(r"\babstract\b\s*[:.]?\s*(.+?)(?=\bintroduction\b|\n\s*1[.\s]|$)", text, re.I | re.S)
        value = re.sub(r"\s+", " ", match.group(1)).strip() if match else fallback
        return value[:1800] or "未识别摘要。"
