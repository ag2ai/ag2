# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime

import pytest

from ag2 import Agent
from ag2.events import TaskCompleted, TaskStarted, ToolCallEvent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import EV_CHANNEL_CLOSED, Hub, Resume
from ag2.network.task_mirror import TaskMirror
from ag2.stream import MemoryStream
from ag2.task import TaskState
from ag2.testing import TestConfig

from ._helpers import _MockClock


class Expiry:
    def __init__(self, hub: Hub, clock: _MockClock) -> None:
        self.hub = hub
        self.clock = clock

    async def expire(self, event: TaskStarted) -> None:
        self.clock.advance(120)
        await self.hub.expire_due()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["ask", "run"])
@pytest.mark.parametrize("expire", [False, True])
async def test_model_run_mirrors_capability_and_preserves_first_terminal(entry: str, expire: bool) -> None:
    clock = _MockClock(datetime.now(UTC).isoformat())
    hub = await Hub.open(MemoryKnowledgeStore(), clock=clock, ttl_sweep_interval=0, expectation_sweep_interval=0)
    try:
        agent = Agent("analyst", config=TestConfig("done"))
        client = await hub.register(agent, resume=Resume(claimed_capabilities=["analysis"]))
        stream = MemoryStream()
        mirror = TaskMirror(hub=hub, owner_id=client.agent_id)
        subscriptions = mirror.attach(stream)
        try:
            if expire:
                stream.where(TaskStarted).subscribe(Expiry(hub, clock).expire)
            if entry == "ask":
                await agent.ask("analyse", stream=stream, capability="analysis", ttl_seconds=60)
            else:
                async with agent.run("analyse", stream=stream, capability="analysis", ttl_seconds=60) as run:
                    await run.result()
        finally:
            mirror.detach(stream, subscriptions)
        events = await stream.history.get_events()
        [started] = [event for event in events if isinstance(event, TaskStarted)]
        [completed] = [event for event in events if isinstance(event, TaskCompleted)]
        assert started.task_id == completed.task_id
        metadata = await hub.get_task(started.task_id)
        assert metadata.spec.capability == "analysis"
        assert metadata.expires_at == started.expires_at
        assert metadata.state == (TaskState.EXPIRED if expire else TaskState.COMPLETED)
        resume = await hub.get_resume(client.agent_id)
        stat = resume.observed["analysis"]
        assert stat.n == 1
        assert stat.expired == int(expire)
        assert stat.completed == int(not expire)
    finally:
        await hub.close()


@pytest.mark.asyncio
async def test_delegation_inside_a_network_turn_is_visible_on_the_hub() -> None:
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
    try:
        worker = Agent("worker", config=TestConfig("summary"))
        bob = Agent(
            "bob",
            config=TestConfig(
                ToolCallEvent(name="task_worker", arguments='{"objective": "summarise the report"}'),
                "here is the summary",
            ),
            tools=[worker.as_tool(description="Summarise documents")],
        )
        alice = await hub.register(Agent("alice"))
        bob_client = await hub.register(bob)

        channel = await alice.open(type="consulting", target="bob")
        await channel.send("Please summarise the report", audience=[bob_client.agent_id])
        await alice.wait_for_channel_event(
            channel_id=channel.channel_id,
            predicate=lambda envelope: envelope.event_type == EV_CHANNEL_CLOSED,
            timeout=5.0,
        )

        tasks = await hub.list_tasks(agent_id=bob_client.agent_id)
        assert [(task.spec.title, task.state) for task in tasks] == [("summarise the report", TaskState.COMPLETED)]
    finally:
        await hub.close()
