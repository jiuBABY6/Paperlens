from dataclasses import replace

from app.config import settings
from app.multi_agent.agents.critic_agent import EvidenceCriticAgent


def critic():
    return EvidenceCriticAgent(replace(settings, multi_agent_max_retries=1))


def figure_evidence(answerable: bool) -> dict:
    return {
        "evidence_id": "paper-fig_001",
        "type": "figure",
        "content": None,
        "metadata": {"query_analysis": {"answerable": answerable}},
    }


def figure_state(*, retry_count: int, answerable: bool) -> dict:
    return {
        "route": {"required_modalities": ["figure"]},
        "plan": {"sub_tasks": [{"task_id": "task_1", "modality": "figure"}]},
        "evidence": [figure_evidence(answerable)],
        "agent_results": {
            "task_1": {
                "task_id": "task_1",
                "status": "success" if answerable else "partial",
                "retryable": not answerable,
                "observations": [],
            }
        },
        "retry_count": retry_count,
    }


def test_critic_requests_one_bounded_targeted_retry() -> None:
    first = critic().review(figure_state(retry_count=0, answerable=False))
    exhausted = critic().review(figure_state(retry_count=1, answerable=False))

    assert first["decision"] == "retry"
    assert first["retry_targets"] == ["task_1"]
    assert exhausted["decision"] == "partial"
    assert exhausted["retry_targets"] == []


def test_critic_approves_complete_evidence_and_refuses_empty_pool() -> None:
    approved = critic().review(figure_state(retry_count=0, answerable=True))
    refused = critic().review({
        "route": {"required_modalities": ["text"]},
        "plan": {"sub_tasks": []},
        "evidence": [],
        "agent_results": {},
        "retry_count": 0,
    })

    assert approved["decision"] == "approved"
    assert refused["decision"] == "refuse"


def test_critic_marks_numeric_conflicts_partial() -> None:
    result = critic().review({
        "route": {"required_modalities": ["text"]},
        "plan": {"sub_tasks": []},
        "evidence": [{
            "evidence_id": "text-1", "type": "text", "content": "Results."
        }],
        "agent_results": {
            "a": {"observations": [{"metric": "F1", "value": "81.0"}]},
            "b": {"observations": [{"metric": "F1", "value": "75.0"}]},
        },
        "retry_count": 0,
    })

    assert result["decision"] == "partial"
    assert result["conflicts"][0]["type"] == "numeric_conflict"

