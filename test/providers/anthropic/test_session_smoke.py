# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end channel smoke tests against real Claude.

The default notify handler engages an agent's LLM via
``_process_text`` when a substantive envelope lands. These tests
drive the consulting / conversation / discussion adapters through
that handler with real ``claude-haiku-4-5`` agents — registration,
invite handshake, dispatch, fold, auto-close, and round-robin
rotation are all exercised against real LLM latency and reply
shapes.

Each test sends a few hundred Haiku tokens; the suite is ~$0.02 to
run end-to-end.
"""

import asyncio
import os

import pytest

from ag2 import Agent
from ag2.config import AnthropicConfig
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_TEXT,
    Hub,
    Resume,
)
from ag2.network.adapters.consulting import CONSULTING_TYPE
from ag2.network.adapters.conversation import CONVERSATION_TYPE
from ag2.network.adapters.discussion import DISCUSSION_TYPE


@pytest.fixture()
def anthropic_config() -> AnthropicConfig:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return AnthropicConfig(model="claude-haiku-4-5", api_key=api_key, temperature=0)


def _agent(name: str, prompt: str, config: AnthropicConfig) -> Agent:
    return Agent(name=name, prompt=prompt, config=config)


async def _wait_text_count(hub: Hub, channel_id: str, expected: int, *, timeout: float = 30.0) -> list:
    """Poll WAL until ``expected`` ``EV_TEXT`` envelopes appear."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        wal = await hub.read_wal(channel_id)
        if sum(1 for e in wal if e.event_type == EV_TEXT) >= expected:
            return wal
        await asyncio.sleep(0.1)
    raise asyncio.TimeoutError(f"channel {channel_id!r} did not reach {expected} EV_TEXT envelopes")


@pytest.mark.anthropic
@pytest.mark.asyncio
async def test_consulting_two_agents_real_llm(anthropic_config: AnthropicConfig) -> None:
    """alice asks a math question, bob's real LLM answers, consulting auto-closes.

    Validates: invite handshake, dispatch to respondent, default
    handler engaging the LLM, reply auto-posted via ``Channel.send``,
    consulting adapter's ``on_accepted`` returning ``CLOSED``.
    """
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(
        _agent("alice", "You are alice, a curious user.", anthropic_config),
    )
    bob = await hub.register(
        _agent(
            "bob",
            "You are bob, a math tutor. Reply with ONLY the numeric answer, nothing else.",
            anthropic_config,
        ),
        resume=Resume(claimed_capabilities=["math"]),
    )

    channel = await alice.open(type=CONSULTING_TYPE, target="bob")
    await channel.send("What is 12 * 17?", audience=[bob.agent_id])

    # Wait for bob's reply to land (1 prompt + 1 reply = 2 EV_TEXT envelopes).
    wal = await _wait_text_count(hub, channel.channel_id, expected=2)
    text_envelopes = [e for e in wal if e.event_type == EV_TEXT]
    assert len(text_envelopes) == 2
    bob_reply = text_envelopes[1].event_data["text"]
    assert "204" in bob_reply, f"expected '204' in bob's reply, got: {bob_reply!r}"

    # Wait for adapter to auto-close after the reply.
    deadline = asyncio.get_event_loop().time() + 5.0
    while asyncio.get_event_loop().time() < deadline:
        info = await channel.info()
        if info.is_terminal():
            break
        await asyncio.sleep(0.05)
    info = await channel.info()
    assert info.is_terminal()
    assert info.close_reason == "consulting_complete"

    # Audit log records CHANNEL_CREATED + CHANNEL_CLOSED for this channel.
    from ag2.network.hub import (
        AUDIT_KIND_CHANNEL_CLOSED,
        AUDIT_KIND_CHANNEL_CREATED,
        AuditLog,
    )

    audit = AuditLog(hub._store)
    records = await audit.read_all()
    channel_records = [r for r in records if r.get("channel_id") == channel.channel_id]
    kinds = {r["kind"] for r in channel_records}
    assert AUDIT_KIND_CHANNEL_CREATED in kinds
    assert AUDIT_KIND_CHANNEL_CLOSED in kinds

    await hub.close()


@pytest.mark.anthropic
@pytest.mark.asyncio
async def test_conversation_bidirectional_real_llm(anthropic_config: AnthropicConfig) -> None:
    """Bidirectional conversation — both default handlers engage their
    LLMs in alternation until we explicitly close.

    No automatic stopping condition — the test bounds the exchange to
    ~3 turns by polling and closing once the floor reaches a target.
    """
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(
        _agent(
            "alice",
            "You are alice, a brief and polite chat partner. Always reply in one short sentence.",
            anthropic_config,
        ),
    )
    bob = await hub.register(
        _agent(
            "bob",
            "You are bob, a brief and polite chat partner. Always reply in one short sentence.",
            anthropic_config,
        ),
    )

    channel = await alice.open(type=CONVERSATION_TYPE, target="bob")
    await channel.send("Hello bob, what is your favorite color?", audience=[bob.agent_id])

    # Bob replies (turn 2). Alice's default handler engages on bob's
    # reply and responds (turn 3). Stop there.
    wal = await _wait_text_count(hub, channel.channel_id, expected=3, timeout=45.0)
    text_envelopes = [e for e in wal if e.event_type == EV_TEXT]
    assert len(text_envelopes) >= 3

    senders = [e.sender_id for e in text_envelopes[:3]]
    assert senders == [alice.agent_id, bob.agent_id, alice.agent_id]

    await channel.close(reason="test-bound")
    info = await channel.info()
    assert info.is_terminal()

    await hub.close()


@pytest.mark.anthropic
@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_discussion_round_robin_three_real_agents(anthropic_config: AnthropicConfig) -> None:
    """3-way discussion. Each agent's default handler engages only
    when the adapter's ``expected_next_speaker`` matches them (via the
    ``can_send`` probe), so the rotation is naturally synchronous.
    """
    _opinion_prompt = (
        "You are {name}, a participant in a round-robin discussion about "
        "whether remote work is good for software teams. When it is your "
        "turn, reply with exactly one short opinion (one sentence) as "
        "plain text. Do not ask questions and do not call any tools — just "
        "state your opinion."
    )
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(
        _agent("alice", _opinion_prompt.format(name="alice"), anthropic_config),
    )
    bob = await hub.register(
        _agent("bob", _opinion_prompt.format(name="bob"), anthropic_config),
    )
    carol = await hub.register(
        _agent("carol", _opinion_prompt.format(name="carol"), anthropic_config),
    )

    channel = await alice.open(
        type=DISCUSSION_TYPE,
        target=["bob", "carol"],
        knobs={"ordering": "round_robin"},
    )
    await channel.send("Let's debate: is remote work good for software teams?")

    # Expect alice (kickoff) → bob → carol → alice (cycle wraps).
    # Stop at 4 EV_TEXT envelopes. Poll budget sits under the 90s
    # per-test timeout marker so a stall surfaces the assertion below
    # rather than a signal kill.
    wal = await _wait_text_count(hub, channel.channel_id, expected=4, timeout=75.0)
    text_envelopes = [e for e in wal if e.event_type == EV_TEXT]
    senders = [e.sender_id for e in text_envelopes[:4]]
    assert senders == [alice.agent_id, bob.agent_id, carol.agent_id, alice.agent_id], (
        f"unexpected speaker order: {senders}"
    )

    await channel.close()
    await hub.close()
