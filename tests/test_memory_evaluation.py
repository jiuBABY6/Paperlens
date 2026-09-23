from app.evaluation import evaluate_memory_cases


def test_memory_evaluation_reports_state_contamination_and_traceability() -> None:
    items = [{
        "source_run_id": "grounded", "status": "active",
        "source_analysis_version": 2, "evidence_ids": ["evidence-1"],
    }, {
        "source_run_id": "refused", "status": "unresolved",
        "source_analysis_version": 2, "evidence_ids": [],
    }]
    report = evaluate_memory_cases([{
        "case_id": "memory-1", "paper_id": "paper-1",
        "expected_items": [
            {"source_run_id": "grounded", "status": "active"},
            {"source_run_id": "refused", "status": "unresolved"},
        ],
        "forbidden_active_run_ids": ["refused"],
    }], lambda _paper_id: items)

    assert report["state_accuracy"] == 1.0
    assert report["active_precision"] == 1.0
    assert report["active_recall"] == 1.0
    assert report["forbidden_activation_rate"] == 0.0
    assert report["active_traceability_rate"] == 1.0
    assert report["passed_count"] == 1


def test_memory_evaluation_detects_wrongly_activated_refusal() -> None:
    report = evaluate_memory_cases([{
        "paper_id": "paper-1",
        "expected_items": [{"source_run_id": "refused", "status": "unresolved"}],
        "forbidden_active_run_ids": ["refused"],
    }], lambda _paper_id: [{
        "source_run_id": "refused", "status": "active",
        "source_analysis_version": 1, "evidence_ids": ["wrong"],
    }])

    assert report["state_accuracy"] == 0.0
    assert report["forbidden_activation_rate"] == 1.0
    assert report["passed_count"] == 0

