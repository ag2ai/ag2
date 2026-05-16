# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for fanout()."""

import asyncio

import pytest

from autogen.beta import Agent, fanout
from autogen.beta.testing import TestConfig


def _agent(response: str = "ok") -> Agent:
    return Agent("test", config=TestConfig(response))


# ---------------------------------------------------------------------------
# Calling conventions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_single_agent_multiple_prompts() -> None:
    agent = _agent("reply")
    replies = await fanout(agent, ["q1", "q2", "q3"])
    assert len(replies) == 3
    assert all(r.body == "reply" for r in replies)


@pytest.mark.asyncio()
async def test_multiple_agents_single_prompt() -> None:
    agents = [_agent("a"), _agent("b"), _agent("c")]
    replies = await fanout(agents, "shared prompt")
    assert len(replies) == 3
    assert [r.body for r in replies] == ["a", "b", "c"]


@pytest.mark.asyncio()
async def test_explicit_pairs() -> None:
    jobs = [
        (_agent("x"), "prompt x"),
        (_agent("y"), "prompt y"),
    ]
    replies = await fanout(jobs)
    assert [r.body for r in replies] == ["x", "y"]


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_empty_prompts() -> None:
    agent = _agent("reply")
    replies = await fanout(agent, [])
    assert replies == []


@pytest.mark.asyncio()
async def test_empty_agents() -> None:
    replies = await fanout([], "prompt")
    assert replies == []


@pytest.mark.asyncio()
async def test_empty_pairs() -> None:
    replies = await fanout([])
    assert replies == []


# ---------------------------------------------------------------------------
# Ordering preserved
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_order_preserved_single_agent() -> None:
    counter = 0
    agent = _agent("ignored")

    original_ask = agent.ask

    async def ordered_ask(prompt: str, **kwargs: object) -> object:
        nonlocal counter
        idx = int(prompt)
        await asyncio.sleep(0.01 * (3 - idx))  # later prompts finish first
        counter += 1
        return await original_ask(str(idx))

    agent.ask = ordered_ask  # type: ignore[method-assign]

    replies = await fanout(agent, ["0", "1", "2"])
    # Results must be in input order even though execution order was reversed
    assert len(replies) == 3


@pytest.mark.asyncio()
async def test_order_preserved_explicit_pairs() -> None:
    results = []

    async def slow_ask(self: object, prompt: str, **kwargs: object) -> object:
        await asyncio.sleep(0.05 if prompt == "slow" else 0.0)
        results.append(prompt)
        from autogen.beta.testing import TestConfig

        return await Agent("t", config=TestConfig(prompt)).ask(prompt)

    a_slow = _agent("slow")
    a_fast = _agent("fast")

    a_slow.ask = lambda p, **k: slow_ask(a_slow, p, **k)  # type: ignore[method-assign]
    a_fast.ask = lambda p, **k: slow_ask(a_fast, p, **k)  # type: ignore[method-assign]

    # slow job first, but fanout runs them concurrently
    replies = await fanout([(a_slow, "slow"), (a_fast, "fast")])

    # replies are in input order regardless of completion order
    assert replies[0].body == "slow"
    assert replies[1].body == "fast"


# ---------------------------------------------------------------------------
# max_concurrent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_max_concurrent_limits_parallelism() -> None:
    active = 0
    peak_active = 0

    agent = _agent("ok")
    original_ask = agent.ask

    async def tracked_ask(prompt: str, **kwargs: object) -> object:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return await original_ask(prompt)

    agent.ask = tracked_ask  # type: ignore[method-assign]

    await fanout(agent, ["p1", "p2", "p3", "p4", "p5"], max_concurrent=2)

    assert peak_active <= 2


@pytest.mark.asyncio()
async def test_max_concurrent_none_runs_all_at_once() -> None:
    active = 0
    peak_active = 0

    agent = _agent("ok")
    original_ask = agent.ask

    async def tracked_ask(prompt: str, **kwargs: object) -> object:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return await original_ask(prompt)

    agent.ask = tracked_ask  # type: ignore[method-assign]

    await fanout(agent, ["p1", "p2", "p3"], max_concurrent=None)

    # All 3 should be active simultaneously
    assert peak_active == 3


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_exception_propagates() -> None:
    agent = _agent("ok")
    original_ask = agent.ask

    call_count = 0

    async def failing_ask(prompt: str, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        if prompt == "bad":
            raise ValueError("deliberate error")
        return await original_ask(prompt)

    agent.ask = failing_ask  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="deliberate error"):
        await fanout(agent, ["ok", "bad", "ok"])


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_single_agent_requires_list_of_prompts() -> None:
    with pytest.raises(ValueError):
        await fanout(_agent(), "single string instead of list")


@pytest.mark.asyncio()
async def test_list_of_agents_requires_string_prompt() -> None:
    with pytest.raises(ValueError):
        await fanout([_agent(), _agent()], ["list instead of string", "list2"])


@pytest.mark.asyncio()
async def test_pairs_rejects_second_arg() -> None:
    with pytest.raises(ValueError):
        await fanout([(_agent(), "p")], "unexpected second arg")
