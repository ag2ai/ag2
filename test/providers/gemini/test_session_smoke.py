# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Cross-cutting channel smoke tests against real Gemini.

These complement the per-adapter smoke tests by exercising scenarios
mocked tests can't reproduce reliably:

* A real-LLM exchange persisted to ``DiskKnowledgeStore`` survives a
  hub restart — passports, WAL, audit log, and capability index all
  rebuild cleanly via ``Hub.hydrate()``.
* Two consulting channels in flight from the same initiator stay
  isolated — replies from each respondent land in their own WAL with
  no cross-contamination.
* A channel closed by one party while the other party's LLM is
  mid-turn does not corrupt state — the late reply attempt fails
  cleanly and the WAL stays consistent.

Each test runs a small handful of ``gemini-3-flash-preview`` turns.
"""

import asyncio
import os
import tempfile

import pytest

from ag2 import Agent
from ag2.config import GeminiConfig
from ag2.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_CLOSED,
    EV_TEXT,
    Hub,
    Resume,
)
from ag2.network.adapters.consulting import CONSULTING_TYPE
from ag2.network.channel import ChannelState
from ag2.network.hub import (
    AUDIT_KIND_AGENT_REGISTERED,
    AUDIT_KIND_CHANNEL_CLOSED,
    AUDIT_KIND_CHANNEL_CREATED,
    AuditLog,
)


@pytest.fixture()
def gemini_config() -> GeminiConfig:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")
    return GeminiConfig(model="gemini-3-flash-preview", api_key=api_key, temperature=0)


def _agent(name: str, prompt: str, config: GeminiConfig) -> Agent:
    return Agent(name=name, prompt=prompt, config=config)


async def _wait_text_count(hub: Hub, channel_id: str, expected: int, *, timeout: float = 30.0) -> list:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        wal = await hub.read_wal(channel_id)
        if sum(1 for e in wal if e.event_type == EV_TEXT) >= expected:
            return wal
        await asyncio.sleep(0.1)
    raise asyncio.TimeoutError(f"channel {channel_id!r} did not reach {expected} EV_TEXT envelopes")


async def _wait_for_terminal(channel, *, timeout: float = 10.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        info = await channel.info()
        if info.is_terminal():
            return
        await asyncio.sleep(0.05)
    raise asyncio.TimeoutError("channel did not reach terminal state in time")


@pytest.mark.gemini
@pytest.mark.asyncio
async def test_persisted_consulting_survives_hub_restart(gemini_config: GeminiConfig) -> None:
    """A consulting exchange persisted on disk hydrates cleanly after
    the hub closes — passports, WAL, capability index, and audit log
    all reconstruct from disk."""
    with tempfile.TemporaryDirectory() as tmp:
        store = DiskKnowledgeStore(tmp)
        hub1 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

        alice = await hub1.register(
            _agent("alice", "You are alice, asking a math question.", gemini_config),
        )
        bob = await hub1.register(
            _agent(
                "bob",
                "You are bob, a math tutor. Reply with ONLY the numeric answer.",
                gemini_config,
            ),
            resume=Resume(claimed_capabilities=["math"]),
        )

        channel = await alice.open(type=CONSULTING_TYPE, target="bob")
        channel_id = channel.channel_id
        await channel.send("What is 7 * 8?", audience=[bob.agent_id])

        await _wait_text_count(hub1, channel_id, expected=2)
        await _wait_for_terminal(channel)

        # Snapshot live state for comparison after rehydrate.
        live_wal = await hub1.read_wal(channel_id)
        live_text = [e.event_data["text"] for e in live_wal if e.event_type == EV_TEXT]
        assert any("56" in t for t in live_text), f"expected '56' in bob's reply, got: {live_text!r}"

        await hub1.close()

        # Reopen — fresh in-memory state, must rebuild from disk.
        hub2 = await Hub.open(DiskKnowledgeStore(tmp), ttl_sweep_interval=0, expectation_sweep_interval=0)

        # Identities recovered.
        alice_p = await hub2.get_agent("alice")
        bob_p = await hub2.get_agent("bob")
        assert alice_p.agent_id == alice.agent_id
        assert bob_p.agent_id == bob.agent_id

        # Capability index restored (bob claims math).
        math_agents = await hub2.list_agents(capability="math")
        assert {p.name for p in math_agents} == {"bob"}

        # WAL bytes survive verbatim.
        rehydrated_wal = await hub2.read_wal(channel_id)
        assert len(rehydrated_wal) == len(live_wal)
        rehydrated_text = [e.event_data["text"] for e in rehydrated_wal if e.event_type == EV_TEXT]
        assert rehydrated_text == live_text

        # Closed channel is in _channels but NOT _active_channels.
        assert channel_id in hub2._channels
        assert channel_id not in hub2._active_channels
        assert hub2._channels[channel_id].state == ChannelState.CLOSED

        # Audit log reconstructed.
        audit = AuditLog(hub2._store)
        records = await audit.read_all()
        kinds = {r["kind"] for r in records}
        assert AUDIT_KIND_AGENT_REGISTERED in kinds
        assert AUDIT_KIND_CHANNEL_CREATED in kinds
        assert AUDIT_KIND_CHANNEL_CLOSED in kinds

        await hub2.close()


@pytest.mark.gemini
@pytest.mark.asyncio
async def test_concurrent_consultings_isolated(gemini_config: GeminiConfig) -> None:
    """Two consulting channels opened concurrently from one initiator
    stay isolated — each respondent's reply lands in its own WAL, no
    envelope leak across channels."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(
        _agent("alice", "You are alice.", gemini_config),
    )
    bob = await hub.register(
        _agent(
            "bob",
            "Reply with ONLY the numeric answer. No words, no punctuation, just the number.",
            gemini_config,
        ),
    )
    carol = await hub.register(
        _agent(
            "carol",
            "Reply with ONLY the numeric answer. No words, no punctuation, just the number.",
            gemini_config,
        ),
    )

    s_bob = await alice.open(type=CONSULTING_TYPE, target="bob")
    s_carol = await alice.open(type=CONSULTING_TYPE, target="carol")
    assert s_bob.channel_id != s_carol.channel_id

    # Fire both prompts simultaneously.
    await asyncio.gather(
        s_bob.send("What is 5 * 5?", audience=[bob.agent_id]),
        s_carol.send("What is 6 * 6?", audience=[carol.agent_id]),
    )

    await _wait_text_count(hub, s_bob.channel_id, expected=2)
    await _wait_text_count(hub, s_carol.channel_id, expected=2)

    bob_wal = await hub.read_wal(s_bob.channel_id)
    carol_wal = await hub.read_wal(s_carol.channel_id)

    bob_texts = [e.event_data["text"] for e in bob_wal if e.event_type == EV_TEXT]
    carol_texts = [e.event_data["text"] for e in carol_wal if e.event_type == EV_TEXT]

    # Bob's WAL contains the 5*5 prompt + a reply containing 25; no 6*6 / 36 leakage.
    assert "What is 5 * 5?" in bob_texts
    assert any("25" in t for t in bob_texts[1:])
    assert "What is 6 * 6?" not in bob_texts
    assert all("36" not in t for t in bob_texts[1:])

    # Carol's WAL is the mirror.
    assert "What is 6 * 6?" in carol_texts
    assert any("36" in t for t in carol_texts[1:])
    assert "What is 5 * 5?" not in carol_texts
    assert all("25" not in t for t in carol_texts[1:])

    # Each consulting channel auto-closes independently.
    await _wait_for_terminal(s_bob)
    await _wait_for_terminal(s_carol)

    await hub.close()


@pytest.mark.gemini
@pytest.mark.asyncio
async def test_close_during_llm_turn_rejects_late_reply(gemini_config: GeminiConfig) -> None:
    """Closing a channel while the respondent's LLM is mid-turn must
    NOT corrupt the WAL — the late reply attempt fails cleanly inside
    bob's notify handler (caught by the per-frame error path) and only
    alice's prompt + the close envelope remain."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(
        _agent("alice", "You are alice.", gemini_config),
    )
    bob = await hub.register(
        _agent("bob", "You are bob, a thoughtful assistant. Reply briefly.", gemini_config),
    )

    channel = await alice.open(type=CONSULTING_TYPE, target="bob")
    sid = channel.channel_id

    await channel.send("Tell me a story about a clever fox.", audience=[bob.agent_id])

    # Bob's notify handler has begun engaging his LLM (real call takes
    # >100ms). Close the channel out from under him.
    await asyncio.sleep(0.05)
    await channel.close(reason="abort_during_turn")

    # Wait long enough for bob's LLM to complete and his late reply
    # attempt to fail through post_envelope.
    await asyncio.sleep(8.0)

    info = await channel.info()
    assert info.is_terminal()
    assert info.close_reason == "abort_during_turn"

    wal = await hub.read_wal(sid)
    text_envelopes = [e for e in wal if e.event_type == EV_TEXT]
    assert len(text_envelopes) == 1, (
        f"expected only alice's prompt in WAL after early close, got {len(text_envelopes)}: "
        f"{[e.event_data.get('text') for e in text_envelopes]}"
    )
    assert text_envelopes[0].sender_id == alice.agent_id

    closed_envelopes = [e for e in wal if e.event_type == EV_CHANNEL_CLOSED]
    assert len(closed_envelopes) == 1
    assert closed_envelopes[0].event_data.get("reason") == "abort_during_turn"

    await hub.close()
