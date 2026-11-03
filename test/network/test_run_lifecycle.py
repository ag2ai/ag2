# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import pytest

from ag2 import Agent
from ag2.events import TaskCompleted, TaskStarted
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import Hub, Resume
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
    clock = _MockClock(datetime.now(timezone.utc).isoformat())
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
