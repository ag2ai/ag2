# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic observer lifecycle tests.

Locks in the ``ObserverStarted`` / ``ObserverCompleted`` contract around an
agent turn without hitting a real provider:

* one ``Started`` and one ``Completed`` per registered observer;
* ``Started`` fires before the turn body, ``Completed`` after;
* ``Completed`` is emitted while observers are still registered, so an
  observer subscribed to ``ObserverCompleted`` receives its own event.
"""

import pytest

from ag2 import Agent, observer
from ag2.events import ModelResponse, ObserverCompleted, ObserverStarted
from ag2.stream import MemoryStream
from ag2.testing import TestConfig

pytestmark = pytest.mark.asyncio


async def test_lifecycle_events_emitted_per_observer() -> None:
    started: list[ObserverStarted] = []
    completed: list[ObserverCompleted] = []

    stream = MemoryStream()
    stream.where(ObserverStarted).subscribe(lambda e: started.append(e))
    stream.where(ObserverCompleted).subscribe(lambda e: completed.append(e))

    obs_a = observer(ModelResponse, lambda e: None)
    obs_b = observer(ModelResponse, lambda e: None)

    agent = Agent("lifecycle", config=TestConfig("ok"), observers=[obs_a, obs_b])
    await agent.ask("hi", stream=stream)

    assert len(started) == 2
    assert len(completed) == 2


async def test_started_before_turn_and_completed_after() -> None:
    order: list[str] = []

    stream = MemoryStream()
    stream.where(ObserverStarted).subscribe(lambda e: order.append("started"))
    stream.where(ModelResponse).subscribe(lambda e: order.append("response"))
    stream.where(ObserverCompleted).subscribe(lambda e: order.append("completed"))

    obs = observer(ModelResponse, lambda e: None)

    agent = Agent("lifecycle", config=TestConfig("ok"), observers=[obs])
    await agent.ask("hi", stream=stream)

    assert order == ["started", "response", "completed"]


async def test_observer_receives_its_own_completed() -> None:
    """``Completed`` is emitted before observers unregister, so an observer
    subscribed to ``ObserverCompleted`` still sees it."""
    seen: list[ObserverCompleted] = []

    obs = observer(ObserverCompleted, lambda e: seen.append(e))

    agent = Agent("lifecycle", config=TestConfig("ok"), observers=[obs])
    await agent.ask("hi")

    assert len(seen) == 1


async def test_no_lifecycle_events_without_observers() -> None:
    started: list[ObserverStarted] = []
    completed: list[ObserverCompleted] = []

    stream = MemoryStream()
    stream.where(ObserverStarted).subscribe(lambda e: started.append(e))
    stream.where(ObserverCompleted).subscribe(lambda e: completed.append(e))

    agent = Agent("lifecycle", config=TestConfig("ok"))
    await agent.ask("hi", stream=stream)

    assert started == []
    assert completed == []
