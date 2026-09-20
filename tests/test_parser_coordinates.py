"""Docling 坐标转换的回归测试。"""

from types import SimpleNamespace

from app.services.parser import PaperParser


def test_docling_bottom_left_bbox_is_converted_to_top_left() -> None:
    """PDF.js 与 PyMuPDF 都采用左上角坐标，Docling 左下角坐标必须翻转。"""
    document = SimpleNamespace(pages={1: SimpleNamespace(size=SimpleNamespace(height=800))})
    bbox = SimpleNamespace(l=10, t=700, r=210, b=500)
    assert PaperParser()._docling_bbox(document, 1, bbox) == (10.0, 100.0, 210.0, 300.0)
