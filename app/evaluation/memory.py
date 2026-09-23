"""Deterministic quality metrics for paper-scoped long-term memory."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable


VALID_STATUSES = {"active", "unresolved", "stale", "archived", "forgotten"}


def evaluate_memory_cases(
    cases: list[dict[str, Any]],
    load_items: Callable[[str], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Evaluate memory state, contamination and source traceability.

    Each case declares ``paper_id`` and ``expected_items`` containing
    ``source_run_id`` + ``status``. ``forbidden_active_run_ids`` identifies
    turns that must never become trusted memory.
    """
    records: list[dict[str, Any]] = []
    state_correct = state_total = 0
    forbidden_active = forbidden_total = 0
    traceable = traceable_total = 0
    actual_active = expected_active = matched_active = 0

    for case in cases:
        paper_id = str(case.get("paper_id", "")).strip()
        if not paper_id:
            raise ValueError("memory case requires paper_id")
        expected = {
            str(item["source_run_id"]): str(item["status"])
            for item in case.get("expected_items", [])
            if item.get("source_run_id")
        }
        invalid = sorted(set(expected.values()) - VALID_STATUSES)
        if invalid:
            raise ValueError(f"invalid expected memory status: {invalid[0]}")
        items = load_items(paper_id)
        by_run = {
            str(item.get("source_run_id")): item
            for item in items if item.get("source_run_id")
        }
        mismatches = []
        for run_id, wanted in expected.items():
            actual = by_run.get(run_id, {}).get("status", "missing")
            state_total += 1
            if actual == wanted:
                state_correct += 1
            else:
                mismatches.append({"source_run_id": run_id, "expected": wanted, "actual": actual})

        forbidden = {str(value) for value in case.get("forbidden_active_run_ids", [])}
        active_ids = {
            run_id for run_id, item in by_run.items() if item.get("status") == "active"
        }
        bad_active = sorted(active_ids.intersection(forbidden))
        forbidden_active += len(bad_active)
        forbidden_total += len(forbidden)

        expected_active_ids = {
            run_id for run_id, status in expected.items() if status == "active"
        }
        expected_active += len(expected_active_ids)
        matched_active += len(expected_active_ids.intersection(active_ids))
        actual_active += len(active_ids)

        active_items = [item for item in items if item.get("status") == "active"]
        for item in active_items:
            traceable_total += 1
            if (
                item.get("source_run_id")
                and int(item.get("source_analysis_version", 0)) > 0
                and item.get("evidence_ids")
            ):
                traceable += 1

        records.append({
            "case_id": case.get("case_id", paper_id),
            "paper_id": paper_id,
            "expected_count": len(expected),
            "actual_count": len(items),
            "status_counts": dict(Counter(item.get("status", "unknown") for item in items)),
            "mismatches": mismatches,
            "forbidden_active_run_ids": bad_active,
            "passed": not mismatches and not bad_active,
        })

    return {
        "case_count": len(cases),
        "passed_count": sum(record["passed"] for record in records),
        "state_accuracy": _ratio(state_correct, state_total),
        "active_precision": _ratio(matched_active, actual_active),
        "active_recall": _ratio(matched_active, expected_active),
        "forbidden_activation_rate": _ratio(forbidden_active, forbidden_total),
        "active_traceability_rate": _ratio(traceable, traceable_total),
        "records": records,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None
