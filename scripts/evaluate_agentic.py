"""读取 JSON/JSONL 记录并比较三种 RAG 模式。"""

import argparse
import json
from pathlib import Path

from app.evaluation import evaluate_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    text = args.dataset.read_text(encoding="utf-8")
    if args.dataset.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        records = value if isinstance(value, list) else value.get("records", [])
    print(json.dumps(evaluate_records(records, args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
