"""Evaluate persisted long-term memory against a small human-labelled JSONL set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.config import settings
from app.evaluation import evaluate_memory_cases
from app.repository import PaperRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PaperLens long-term memory")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=settings.database_path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    repository = PaperRepository(args.database)
    report = evaluate_memory_cases(
        cases,
        lambda paper_id: repository.list_memory_items(
            paper_id, include_inactive=True, limit=1000
        ),
    )
    content = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
