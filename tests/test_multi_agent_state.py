from app.domain import EvidenceObject
from app.multi_agent.state import evidence_from_state, evidence_to_state, merge_evidence


def test_state_evidence_round_trip_and_stable_deduplication() -> None:
    evidence = EvidenceObject(
        "paper-p1-b1-s0", "text", 1, (1.0, 2.0, 3.0, 4.0),
        "Direct evidence.", "Methods", {"chunk_id": "paper-p1-b1"},
    )
    payload = evidence_to_state(evidence, 0.8)
    merged = merge_evidence(
        [payload],
        [{**payload, "metadata": {"chunk_id": "paper-p1-b1", "verified": True}}],
    )

    assert len(merged) == 1
    assert merged[0]["metadata"]["verified"] is True
    restored = evidence_from_state(merged[0])
    assert restored.evidence_id == evidence.evidence_id
    assert restored.bbox == evidence.bbox

