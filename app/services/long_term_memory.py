"""Versioned paper-level memory lifecycle built only from persisted, verified runs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
import uuid
from typing import Any

from app.observability import llmops, span
from app.services.query_resolver import is_memory_control_query


class LongTermMemoryService:
    MEMORY_VERSION = 2

    def __init__(self, repository, settings) -> None:
        self.repository = repository
        self.settings = settings

    def record_run(
        self, *, paper_id: str, conversation_id: str, run: dict[str, Any],
        resolved_question: str, status: str, result: dict[str, Any],
    ) -> dict[str, Any]:
        """Create active memory only for grounded answers; partial results stay unresolved."""
        with span("memory.record", operation="record_run", status=status):
            paper_result = self.repository.get(paper_id)
            if not paper_result:
                raise KeyError("paper_not_found")
            paper = paper_result[0]
            citations = list(result.get("citations", []))
            verified = (
                status == "completed"
                and result.get("answerable") is True
                and bool(citations)
            )
            evidence_pairs = list(dict.fromkeys(
                (
                    str(item.get("evidence_id", item.get("id", ""))),
                    str(item.get("type", "text")),
                )
                for item in citations if item.get("evidence_id", item.get("id"))
            ))[:30]
            evidence_ids = [identifier for identifier, _kind in evidence_pairs]
            evidence_types = [kind for _identifier, kind in evidence_pairs]
            sections = list(dict.fromkeys(
                str(item.get("section", "")).strip()
                for item in citations if str(item.get("section", "")).strip()
            ))[:20]
            question = str(run.get("original_question", ""))[:1000]
            resolved = resolved_question[:1000]
            fingerprint = self._fingerprint(resolved, evidence_ids, verified)
            item = self.repository.upsert_memory_item({
                "id": f"memory-{uuid.uuid4()}",
                "paper_id": paper_id,
                "source_conversation_id": conversation_id,
                "source_run_id": run.get("id"),
                "source_analysis_version": paper.analysis_version,
                "fingerprint": fingerprint,
                "question": question,
                "resolved_question": resolved,
                "evidence_ids": evidence_ids,
                "evidence_types": evidence_types,
                "sections": sections,
                "importance": self.score_importance(resolved, evidence_types, sections),
                "confidence": 1.0 if verified else 0.0,
                "status": "active" if verified else "unresolved",
                "memory_version": self.MEMORY_VERSION,
                "expires_at": self._expiry(),
            })
            self._enforce_capacity(paper_id)
            self.refresh_aggregate(paper_id)
            llmops.memory_operations.labels(
                operation="write", status=item["status"]
            ).inc()
            return item

    def backfill(self, paper_id: str) -> dict[str, int]:
        """Idempotently migrate old completed/partial runs into versioned memory items."""
        created = active = unresolved = skipped = 0
        for candidate in self.repository.list_memory_backfill_candidates(paper_id):
            if is_memory_control_query(str(candidate.get("original_question", ""))):
                skipped += 1
                continue
            result = dict(candidate.get("result", {}))
            citations = candidate.get("citations", [])
            if not citations:
                citations = result.get("citations", [])
            result["citations"] = citations
            if "answerable" not in result:
                result["answerable"] = candidate.get("metadata", {}).get("answerable")
            try:
                item = self.record_run(
                    paper_id=paper_id,
                    conversation_id=candidate["conversation_id"],
                    run={
                        "id": candidate["run_id"],
                        "original_question": candidate["original_question"],
                    },
                    resolved_question=candidate["resolved_question"],
                    status=candidate["run_status"],
                    result=result,
                )
            except (KeyError, ValueError):
                skipped += 1
                continue
            created += 1
            if item["status"] == "active":
                active += 1
            else:
                unresolved += 1
        self.refresh_aggregate(paper_id)
        llmops.memory_operations.labels(operation="backfill", status="success").inc()
        return {"processed": created, "active": active, "unresolved": unresolved, "skipped": skipped}

    def invalidate_for_version(self, paper_id: str, analysis_version: int) -> int:
        count = self.repository.mark_memory_stale_for_version(paper_id, analysis_version)
        self.refresh_aggregate(paper_id)
        if count:
            llmops.memory_operations.labels(operation="invalidate", status="stale").inc(count)
        return count

    def apply_retention(self) -> int:
        count = self.repository.expire_memory_items()
        if count:
            llmops.memory_operations.labels(operation="expire", status="archived").inc(count)
        return count

    def list_items(
        self, paper_id: str, *, include_inactive: bool = False
    ) -> list[dict[str, Any]]:
        self.apply_retention()
        return self.repository.list_memory_items(
            paper_id, include_inactive=include_inactive,
            limit=self.settings.memory_max_items_per_paper,
        )

    def update_item(self, paper_id: str, memory_id: str, **changes) -> dict[str, Any] | None:
        if changes.get("pinned") is False:
            # Pinning clears the TTL. Unpinning starts a fresh retention window.
            changes["expires_at"] = self._expiry()
        item = self.repository.update_memory_item(paper_id, memory_id, **changes)
        if item:
            self.refresh_aggregate(paper_id)
            operation = "forget" if item["status"] == "forgotten" else "edit"
            llmops.memory_operations.labels(operation=operation, status=item["status"]).inc()
        return item

    def refresh_aggregate(self, paper_id: str) -> dict[str, Any]:
        items = self.repository.list_memory_items(
            paper_id, include_inactive=True,
            limit=self.settings.memory_max_items_per_paper,
        )
        if not items:
            # Preserve a legacy v1 aggregate until historical runs can be backfilled,
            # and keep an explicitly cleared memory empty instead of recreating a
            # synthetic aggregate row.
            existing = self.repository.get_paper_learning_memory(paper_id)
            self._update_metrics()
            return existing
        eligible_items = [
            item for item in items
            if not is_memory_control_query(str(item.get("question", "")))
        ]
        active = [item for item in eligible_items if item["status"] == "active"]
        unresolved = [item for item in eligible_items if item["status"] == "unresolved"]
        newest = max(
            [*active, *unresolved], key=lambda item: item["updated_at"], default=None
        )
        memory = {
            "memory_version": self.MEMORY_VERSION,
            "interactions": [{
                "memory_id": item["id"],
                "conversation_id": item["source_conversation_id"],
                "source_run_id": item["source_run_id"],
                "question": item["question"],
                "resolved_question": item["resolved_question"],
                "evidence_ids": item["evidence_ids"],
                "evidence_types": item["evidence_types"],
                "sections": item["sections"],
                "importance": item["importance"],
                "confidence": item["confidence"],
                "pinned": item["pinned"],
                "user_note": item["user_note"],
                "source_analysis_version": item["source_analysis_version"],
                "status": item["status"],
            } for item in active],
            "explored_sections": list(dict.fromkeys(
                section for item in active for section in item["sections"]
            ))[-50:],
            "figure_ids": list(dict.fromkeys(
                evidence_id for item in active
                for evidence_id, kind in self._evidence_pairs(item)
                if kind == "figure"
            ))[-50:],
            "table_ids": list(dict.fromkeys(
                evidence_id for item in active
                for evidence_id, kind in self._evidence_pairs(item)
                if kind == "table"
            ))[-50:],
            "unresolved_questions": [{
                "memory_id": item["id"], "question": item["question"],
                "resolved_question": item["resolved_question"],
                "importance": item["importance"],
            } for item in unresolved[-50:]],
            "last_question": newest["question"] if newest else "",
            "last_resolved_question": newest["resolved_question"] if newest else "",
            "status_counts": self._status_counts(eligible_items),
        }
        value = self.repository.update_paper_learning_memory(
            paper_id, memory, version=self.MEMORY_VERSION
        )
        self._update_metrics()
        return value

    @staticmethod
    def score_importance(
        question: str, evidence_types: list[str], sections: list[str]
    ) -> float:
        text = f"{question} {' '.join(sections)}".lower()
        score = 0.45
        if re.search(r"method|methodology|approach|contribution|方法|贡献|创新", text):
            score += 0.25
        if re.search(r"result|experiment|metric|f1|accuracy|实验|结果|指标", text):
            score += 0.2
        if "figure" in evidence_types or "table" in evidence_types:
            score += 0.05
        if re.search(r"acknowledg|致谢|颜色|color", text):
            score -= 0.15
        return round(max(0.0, min(score, 1.0)), 3)

    def _enforce_capacity(self, paper_id: str) -> None:
        items = self.repository.list_memory_items(paper_id, include_inactive=False, limit=1000)
        overflow = len(items) - self.settings.memory_max_items_per_paper
        if overflow <= 0:
            return
        candidates = sorted(
            (item for item in items if not item["pinned"]),
            key=lambda item: (item["importance"], item["updated_at"]),
        )
        for item in candidates[:overflow]:
            self.repository.update_memory_item(
                paper_id, item["id"], status="archived"
            )

    def _expiry(self) -> str:
        value = datetime.now(timezone.utc) + timedelta(
            days=self.settings.memory_retention_days
        )
        return value.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _fingerprint(resolved: str, evidence_ids: list[str], verified: bool) -> str:
        normalized = re.sub(r"\s+", " ", resolved.strip().lower())
        value = json.dumps(
            [normalized, sorted(evidence_ids), "active" if verified else "unresolved"],
            ensure_ascii=False, separators=(",", ":"),
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _evidence_pairs(item: dict[str, Any]) -> list[tuple[str, str]]:
        types = item.get("evidence_types", [])
        if len(types) == 1:
            return [(identifier, types[0]) for identifier in item.get("evidence_ids", [])]
        return [
            (identifier, kind)
            for identifier, kind in zip(item.get("evidence_ids", []), types)
        ]

    @staticmethod
    def _status_counts(items: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        return counts

    def _update_metrics(self) -> None:
        counts = self.repository.memory_status_counts()
        for status in ("active", "unresolved", "stale", "archived", "forgotten"):
            llmops.memory_items.labels(status=status).set(counts.get(status, 0))
