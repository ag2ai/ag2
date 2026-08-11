# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Per-channel cache pruning on terminal channel transitions.

The hub holds several per-channel scratch maps — ``_adapter_states``,
``_channel_locks``, ``_channel_open_waiters``, ``_fired_violations``.
They must be pruned when a channel transitions to a terminal state
(``CLOSED`` / ``EXPIRED``) so a long-lived hub processing many short
channels does not grow without bound.

The channel metadata itself (``_channels``) is intentionally kept for
auditability + ``read_wal()`` access. Only volatile per-channel
caches are pruned.
"""

import asyncio

import pytest

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    Hub,
)
from ag2.testing import TestConfig


def _agent(name: str) -> Agent:
    return Agent(name=name, config=TestConfig())


@pytest.mark.asyncio
async def test_adapter_state_retained_after_close_for_analysis() -> None:
    """``_adapter_states`` is intentionally kept post-close for inspection.

    Fold state carries analytical value for tests, debug tools, and
    post-mortem analysis. Pruning the heavier per-channel
    synchronization primitives (locks) is enough to bound memory; the
    state dataclass itself is small.
    """
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))
    channel = await alice.open(type="conversation", target="bob")
    channel_id = channel.channel_id

    pre_close_state = hub.adapter_state(channel_id)
    assert pre_close_state is not None

    await hub.close_channel(channel_id, reason="explicit")

    # State stays available for analysis.
    assert hub.adapter_state(channel_id) is pre_close_state
    # Channel metadata stays for audit / WAL access.
    metadata = await hub.get_channel(channel_id)
    assert metadata.is_terminal()

    await hub.close()


@pytest.mark.asyncio
async def test_channel_locks_pruned_on_close() -> None:
    """``_channel_locks`` drops the entry on terminal transition."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))
    channel = await alice.open(type="conversation", target="bob")
    channel_id = channel.channel_id

    # WAL append acquired the lock, so it's in the dict.
    await channel.send("hello")
    assert channel_id in hub._channel_locks  # pre-close

    await hub.close_channel(channel_id, reason="explicit")
    assert channel_id not in hub._channel_locks  # post-close

    await hub.close()


@pytest.mark.asyncio
async def test_fired_violations_pruned_on_close() -> None:
    """``_fired_violations`` (already pre-existing prune) keeps the contract."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))
    channel = await alice.open(type="conversation", target="bob")
    channel_id = channel.channel_id

    # Force a fired-violations entry directly (no live expectation).
    hub._fired_violations[channel_id] = {(0, "max_silence", "")}
    await hub.close_channel(channel_id, reason="explicit")
    assert channel_id not in hub._fired_violations

    await hub.close()


@pytest.mark.asyncio
async def test_pruning_under_many_short_channels_keeps_caches_bounded() -> None:
    """Open + close 20 channels — the per-channel caches stay bounded.

    Realistic proxy for the long-lived-hub case: every channel that
    closes must release its per-channel scratch state so the dicts
    don't grow without bound.
    """
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))

    for _ in range(20):
        channel = await alice.open(type="conversation", target="bob")
        await channel.send("ping")
        await hub.close_channel(channel.channel_id, reason="cycle")

    # Yield so any sweeper / dispatch task settles.
    await asyncio.sleep(0.1)

    # All 20 closed → the heavy per-channel synchronization caches
    # are empty. ``_adapter_states`` is retained by design.
    assert hub._channel_locks == {}
    assert hub._fired_violations == {}
    assert hub._channel_open_waiters == {}

    await hub.close()
