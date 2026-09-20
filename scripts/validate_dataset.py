"""校验人工评测集中的论文、证据 ID、页码和原文引用。"""

import argparse
import json
from pathlib import Path
import re
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.config import settings
from app.repository import PaperRepository


def normalize_quote(value: str) -> str:
    """仅归一化空白，不改写标点和单词，保证引用仍是原文。"""
    return re.sub(r"\s+", " ", value).strip().casefold()


def load_dataset(path: Path) -> tuple[list[tuple[int, dict]], list[str]]:
    """逐行读取 JSONL，并保留可定位的行号错误。"""
    rows: list[tuple[int, dict]] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"line {line_number}: JSON 解析失败：{error.msg}")
            continue
        if not isinstance(value, dict):
            errors.append(f"line {line_number}: 每行必须是 JSON 对象")
            continue
        rows.append((line_number, value))
    return rows, errors


def validate_dataset(path: Path, repository: PaperRepository) -> dict:
    """返回包含 errors/warnings 的机器可读校验报告。"""
    rows, errors = load_dataset(path)
    warnings: list[str] = []
    seen_case_ids: set[str] = set()
    seen_questions: set[tuple[str, str]] = set()
    paper_cache: dict[str, object] = {}
    paper_splits: dict[str, set[str]] = {}

    for line_number, row in rows:
        prefix = f"line {line_number}"
        case_id = str(row.get("case_id", "")).strip()
        paper_id = str(row.get("paper_id", "")).strip()
        question = str(row.get("question", "")).strip()
        expected_answer = str(row.get("expected_answer", "")).strip()
        answerable = row.get("answerable")
        split = row.get("split")

        if not case_id:
            errors.append(f"{prefix}: 缺少 case_id")
        elif case_id in seen_case_ids:
            errors.append(f"{prefix}: case_id 重复：{case_id}")
        else:
            seen_case_ids.add(case_id)
        if not paper_id:
            errors.append(f"{prefix}: 缺少 paper_id")
            continue
        if not question:
            errors.append(f"{prefix}: question 不能为空")
        elif (paper_id, question) in seen_questions:
            warnings.append(f"{prefix}: 同一论文的问题重复")
        else:
            seen_questions.add((paper_id, question))
        if not expected_answer:
            errors.append(f"{prefix}: expected_answer 不能为空")
        if not isinstance(answerable, bool):
            errors.append(f"{prefix}: answerable 必须是 true 或 false")
        if split not in ("dev", "test"):
            errors.append(f"{prefix}: split 必须是 dev 或 test")
        elif paper_id:
            paper_splits.setdefault(paper_id, set()).add(split)

        list_fields = ("expected_chunk_ids", "expected_sentence_ids", "expected_pages", "gold_quotes", "tags")
        for field in list_fields:
            if not isinstance(row.get(field), list):
                errors.append(f"{prefix}: {field} 必须是数组")
        for field in (
            "expected_evidence_ids", "expected_modalities", "expected_tools",
            "expected_text_ids", "expected_figure_ids", "expected_table_ids",
        ):
            if field in row and not isinstance(row.get(field), list):
                errors.append(f"{prefix}: {field} 必须是数组")

        result = paper_cache.get(paper_id)
        if result is None:
            result = repository.get(paper_id)
            paper_cache[paper_id] = result
        if not result:
            errors.append(f"{prefix}: paper_id 不存在：{paper_id}")
            continue
        paper, _card = result
        if paper.status != "completed":
            errors.append(f"{prefix}: 论文尚未处理完成：{paper.status}")
            continue

        chunk_ids = row.get("expected_chunk_ids", [])
        sentence_ids = row.get("expected_sentence_ids", [])
        pages = row.get("expected_pages", [])
        quotes = row.get("gold_quotes", [])
        raw_groups = row.get("expected_chunk_groups")
        chunk_groups: list[list[str]] = []
        if raw_groups is not None:
            if not isinstance(raw_groups, list):
                errors.append(f"{prefix}: expected_chunk_groups 必须是二维数组")
            else:
                for group_index, group in enumerate(raw_groups, start=1):
                    if not isinstance(group, list) or not group:
                        errors.append(
                            f"{prefix}: expected_chunk_groups 第 {group_index} 组必须是非空数组"
                        )
                        continue
                    if not all(isinstance(item, str) and item.strip() for item in group):
                        errors.append(
                            f"{prefix}: expected_chunk_groups 第 {group_index} 组包含无效 Chunk ID"
                        )
                        continue
                    chunk_groups.append(group)
                group_union = {item for group in chunk_groups for item in group}
                if group_union != set(chunk_ids):
                    errors.append(
                        f"{prefix}: expected_chunk_ids 必须等于 expected_chunk_groups 的并集"
                    )
        raw_sentence_groups = row.get("expected_sentence_groups")
        sentence_groups: list[list[str]] = []
        if raw_sentence_groups is not None:
            if not isinstance(raw_sentence_groups, list):
                errors.append(f"{prefix}: expected_sentence_groups 必须是二维数组")
            else:
                for group_index, group in enumerate(raw_sentence_groups, start=1):
                    if not isinstance(group, list) or not group:
                        errors.append(
                            f"{prefix}: expected_sentence_groups 第 {group_index} 组必须是非空数组"
                        )
                        continue
                    if not all(isinstance(item, str) and item.strip() for item in group):
                        errors.append(
                            f"{prefix}: expected_sentence_groups 第 {group_index} 组包含无效 Sentence ID"
                        )
                        continue
                    sentence_groups.append(group)
                group_union = {item for group in sentence_groups for item in group}
                if group_union != set(sentence_ids):
                    errors.append(
                        f"{prefix}: expected_sentence_ids 必须等于 expected_sentence_groups 的并集"
                    )
        evidence_ids = row.get("expected_evidence_ids")
        if evidence_ids is None:
            evidence_ids = list(sentence_ids)
        raw_evidence_groups = row.get("expected_evidence_groups")
        evidence_groups: list[list[str]] = []
        if raw_evidence_groups is not None:
            if not isinstance(raw_evidence_groups, list):
                errors.append(f"{prefix}: expected_evidence_groups 必须是二维数组")
            else:
                for group_index, group in enumerate(raw_evidence_groups, start=1):
                    if not isinstance(group, list) or not group:
                        errors.append(
                            f"{prefix}: expected_evidence_groups 第 {group_index} 组必须是非空数组"
                        )
                        continue
                    if not all(isinstance(item, str) and item.strip() for item in group):
                        errors.append(
                            f"{prefix}: expected_evidence_groups 第 {group_index} 组包含无效 Evidence ID"
                        )
                        continue
                    evidence_groups.append(group)
                group_union = {item for group in evidence_groups for item in group}
                if group_union != set(evidence_ids):
                    errors.append(
                        f"{prefix}: expected_evidence_ids 必须等于 expected_evidence_groups 的并集"
                    )
        expected_modalities = row.get("expected_modalities")
        if expected_modalities is None:
            expected_modalities = ["text"] if sentence_ids else []
        if answerable is True and (not evidence_ids or not quotes):
            errors.append(f"{prefix}: 可回答问题必须标注统一 Evidence 和原文引用")
        if answerable is True and "text" in expected_modalities and (
            not chunk_ids or not sentence_ids
        ):
            errors.append(f"{prefix}: Text 问题必须标注 Chunk 和 Sentence")
        if answerable is False and (
            chunk_ids or sentence_ids or pages or quotes or chunk_groups
            or sentence_groups or evidence_ids or evidence_groups
        ):
            errors.append(f"{prefix}: 无答案问题的 Gold Evidence 必须为空")

        chunk_map = {item.id: item for item in paper.chunks}
        sentence_map = {item.id: item for item in paper.sentences}
        figure_map = {item.id: item for item in paper.figures}
        # expected_chunk_ids 保留旧字段名，但多模态评测允许它表示
        # Figure/Table 检索单元；文本 Sentence 仍必须指向真实 Chunk。
        unknown_chunks = [
            item for item in chunk_ids
            if item not in chunk_map and item not in figure_map
        ]
        unknown_sentences = [item for item in sentence_ids if item not in sentence_map]
        if unknown_chunks:
            errors.append(f"{prefix}: Chunk 不属于该论文：{unknown_chunks}")
        if unknown_sentences:
            errors.append(f"{prefix}: Sentence 不属于该论文：{unknown_sentences}")
        unknown_evidence = [
            item for item in evidence_ids
            if item not in sentence_map and item not in chunk_map and item not in figure_map
        ]
        if unknown_evidence:
            errors.append(f"{prefix}: Evidence 不属于该论文：{unknown_evidence}")
        for field, kind in (("expected_figure_ids", "picture"), ("expected_table_ids", "table")):
            invalid = [
                item for item in row.get(field, [])
                if item not in figure_map or figure_map[item].kind != kind
            ]
            if invalid:
                errors.append(f"{prefix}: {field} 不属于该论文或类型错误：{invalid}")
        invalid_modalities = [
            item for item in row.get("expected_modalities", [])
            if item not in ("text", "figure", "table")
        ]
        if invalid_modalities:
            errors.append(f"{prefix}: expected_modalities 包含未知类型：{invalid_modalities}")

        valid_sentences = [sentence_map[item] for item in sentence_ids if item in sentence_map]
        inconsistent = [item.id for item in valid_sentences if item.chunk_id not in chunk_ids]
        if inconsistent:
            errors.append(f"{prefix}: Sentence 对应的 chunk_id 未列入 expected_chunk_ids：{inconsistent}")
        gold_evidence = [
            sentence_map[item] if item in sentence_map
            else chunk_map[item] if item in chunk_map
            else figure_map[item]
            for item in evidence_ids
            if item in sentence_map or item in chunk_map or item in figure_map
        ]
        actual_pages = {item.page for item in gold_evidence}
        if gold_evidence and set(pages) != actual_pages:
            errors.append(
                f"{prefix}: expected_pages={sorted(set(pages))} 与 Evidence 页码={sorted(actual_pages)} 不一致"
            )
        normalized_sources = [normalize_quote(item.text) for item in valid_sentences]
        for evidence_id in evidence_ids:
            item = figure_map.get(evidence_id)
            if item:
                normalized_sources.extend(normalize_quote(value) for value in (
                    item.caption, item.content, item.nearby_text
                ) if value)
        for quote in quotes:
            normalized = normalize_quote(str(quote))
            if not normalized or not any(normalized in text for text in normalized_sources):
                errors.append(f"{prefix}: gold_quote 不是已标注 Evidence 的原文子串：{quote!r}")

    for paper_id, splits in paper_splits.items():
        if len(splits) > 1:
            errors.append(
                f"paper_id {paper_id} 同时出现在 dev/test，必须按整篇论文隔离"
            )

    return {
        "dataset": str(path),
        "valid": not errors,
        "rows": len(rows),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 PaperLens JSONL 人工评测集")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="输出完整 JSON 报告")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not args.dataset.is_file():
        parser.error(f"评测集不存在：{args.dataset}")
    report = validate_dataset(args.dataset, PaperRepository(settings.database_path))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"rows={report['rows']} errors={len(report['errors'])} warnings={len(report['warnings'])}")
        for item in report["errors"]:
            print(f"ERROR: {item}")
        for item in report["warnings"]:
            print(f"WARN: {item}")
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
