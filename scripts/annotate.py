"""按页码、章节或关键词查看论文中的 Chunk/Sentence，辅助人工标注评测集。"""

import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.config import settings
from app.repository import PaperRepository


def annotation_records(
    paper,
    unit: str = "sentence",
    page: int | None = None,
    keyword: str = "",
    section: str = "",
    limit: int = 50,
) -> list[dict]:
    """筛选并返回适合复制到评测集的证据记录。"""
    keyword_value = keyword.casefold().strip()
    section_value = section.casefold().strip()
    if unit == "chunk":
        values = [
            {
                "chunk_id": item.id,
                "page": item.page,
                "section": item.section,
                "text": item.text,
                "bbox": item.bbox,
                "sentence_ids": item.sentence_ids,
            }
            for item in paper.chunks
        ]
    else:
        values = [
            {
                "sentence_id": item.id,
                "chunk_id": item.chunk_id,
                "page": item.page,
                "section": item.section,
                "text": item.text,
                "bbox": item.bbox,
                "bboxes": item.bboxes,
            }
            for item in paper.sentences
        ]
    records = []
    for item in values:
        if page is not None and item["page"] != page:
            continue
        if section_value and section_value not in item["section"].casefold():
            continue
        searchable = f"{item['section']} {item['text']}".casefold()
        if keyword_value and keyword_value not in searchable:
            continue
        records.append(item)
    records.sort(key=lambda item: (
        item["page"],
        item.get("bbox")[1] if item.get("bbox") else float("inf"),
        item.get("bbox")[0] if item.get("bbox") else float("inf"),
        item.get("sentence_id") or item.get("chunk_id"),
    ))
    return records[:limit]


def print_readable(records: list[dict], unit: str) -> None:
    """以便于人工核对和复制 ID 的格式输出。"""
    for index, item in enumerate(records, start=1):
        identifier = item["sentence_id"] if unit == "sentence" else item["chunk_id"]
        print(f"\n[{index}] page={item['page']} section={item['section']}")
        print(f"{unit}_id={identifier}")
        if unit == "sentence":
            print(f"chunk_id={item['chunk_id']}")
        else:
            print(f"sentence_ids={json.dumps(item['sentence_ids'], ensure_ascii=False)}")
        print(item["text"])


def main() -> None:
    parser = argparse.ArgumentParser(description="查看论文证据，辅助构建人工评测集")
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--unit", choices=("sentence", "chunk"), default="sentence")
    parser.add_argument("--page", type=int)
    parser.add_argument("--keyword", default="")
    parser.add_argument("--section", default="")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--jsonl", action="store_true", help="输出机器可读 JSONL")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    repository = PaperRepository(settings.database_path)
    result = repository.get(args.paper_id)
    if not result:
        parser.error(f"未找到论文：{args.paper_id}")
    paper, _card = result
    records = annotation_records(
        paper,
        unit=args.unit,
        page=args.page,
        keyword=args.keyword,
        section=args.section,
        limit=max(args.limit, 1),
    )
    if args.jsonl:
        for item in records:
            print(json.dumps(item, ensure_ascii=False))
    else:
        print(f"paper_id={paper.id} title={paper.title} matches={len(records)}")
        print_readable(records, args.unit)


if __name__ == "__main__":
    main()
