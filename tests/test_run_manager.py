import asyncio

from app.services.run_manager import RunManager


def test_events_are_ordered_replayable_and_terminal() -> None:
    async def scenario():
        manager = RunManager(buffer_size=10, heartbeat_seconds=0.01)
        await manager.publish("run-1", "run.started", {"value": 1})
        await manager.publish("run-1", "answer.delta", {"delta": "中"})
        await manager.publish("run-1", "run.completed", {"status": "completed"})
        events = [value async for value in manager.events("run-1")]
        replay = [value async for value in manager.events("run-1", last_event_id=1)]
        return events, replay

    events, replay = asyncio.run(scenario())
    assert [value.splitlines()[0] for value in events] == ["id: 1", "id: 2", "id: 3"]
    assert len(replay) == 2
    assert "event: answer.delta" in replay[0]


def test_heartbeat_is_valid_sse_comment() -> None:
    async def scenario():
        manager = RunManager(heartbeat_seconds=0.01)
        manager.register("run-1")
        stream = manager.events("run-1")
        value = await anext(stream)
        await stream.aclose()
        return value

    assert asyncio.run(scenario()) == ": heartbeat\n\n"
