"""单次查询范围内的临时 Evidence Memory。"""

from app.domain import EvidenceObject


class EvidenceMemory:
    def __init__(self) -> None:
        self._items: dict[str, EvidenceObject] = {}
        self._records: list[dict] = []

    def add(self, subtask_id: str, evidence: EvidenceObject, relevance: float, source_tool: str) -> bool:
        duplicate = evidence.evidence_id in self._items
        self._items[evidence.evidence_id] = evidence
        if duplicate:
            record = next(item for item in self._records if item["evidence_id"] == evidence.evidence_id)
            if subtask_id not in record["subtask_ids"]:
                record["subtask_ids"].append(subtask_id)
        else:
            self._records.append({
                "subtask_id": subtask_id,
                "subtask_ids": [subtask_id],
                "evidence_id": evidence.evidence_id,
                "type": evidence.type,
                "relevance": round(float(relevance), 4),
                "source_tool": source_tool,
                "used": False,
            })
        return not duplicate

    def evidence(self) -> list[EvidenceObject]:
        return list(self._items.values())

    def records(self) -> list[dict]:
        return list(self._records)

    def mark_used(self, evidence_ids: list[str]) -> None:
        """把最终答案实际引用的证据同步到可审计 Memory。"""
        used = set(evidence_ids)
        for item in self._records:
            item["used"] = item["evidence_id"] in used

    def sufficiency(self, subtask_ids: list[str]) -> dict:
        covered = {
            subtask_id
            for item in self._records
            for subtask_id in item.get("subtask_ids", [item["subtask_id"]])
        }
        missing = [item for item in subtask_ids if item not in covered]
        return {
            "sufficient": not missing,
            "covered_subtasks": [item for item in subtask_ids if item in covered],
            "missing_subtasks": missing,
            "missing_information": [f"{item} 尚未找到可追溯证据。" for item in missing],
        }
