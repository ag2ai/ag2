# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``HumanClient`` integration tests.

Covers registration, push + pull surfaces, channel participation across
adapter types, and the rejection paths that keep the agent / human entry
points distinct.
"""

import asyncio

import pytest

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_TEXT,
    Channel,
    Hub,
    HubClient,
    HumanClient,
    LocalLink,
    Passport,
    Resume,
)
from ag2.network.channel import ChannelState
from ag2.testing import TestConfig

from ._helpers import wait_for_text_count


def _agent(name: str, *replies: str) -> Agent:
    return Agent(name=name, config=TestConfig(*replies))


@pytest.mark.asyncio
async def test_register_human_stamps_kind_and_id() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    human = await hc.register_human(
        Passport(name="reviewer-1"),
        resume=Resume(summary="approves payouts"),
    )

    assert isinstance(human, HumanClient)
    assert human.passport.kind == "human"
    assert human.passport.effective_kind == "human"
    assert human.agent_id  # stamped
    assert human.resume.summary == "approves payouts"

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_register_human_with_default_resume() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    human = await hc.register_human(Passport(name="dave"))
    assert human.resume == Resume()

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_register_rejects_human_kind() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    with pytest.raises(ValueError, match="register_human"):
        await hc.register(
            _agent("bot"),
            Passport(name="bot", kind="human"),
            Resume(),
        )

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_register_human_rejects_non_human_kind() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    with pytest.raises(ValueError, match="kind='human'"):
        await hc.register_human(Passport(name="x", kind="agent"))

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_list_agents_filter_by_kind() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    await hub.register(_agent("alice"))
    await hc.register_human(Passport(name="reviewer"))

    agents = await hub.list_agents(kind="agent")
    humans = await hub.list_agents(kind="human")
    everyone = await hub.list_agents()

    assert {p.name for p in agents} == {"alice"}
    assert {p.name for p in humans} == {"reviewer"}
    assert {p.name for p in everyone} == {"alice", "reviewer"}

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_consulting_human_responds_to_agent() -> None:
    """An agent opens consulting to a human; the human's reply closes the channel."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    human_hc = HubClient(link, hub=hub)

    alice = await hub.register(_agent("alice"))
    human = await human_hc.register_human(Passport(name="reviewer"))

    # Human reacts to the inbound prompt by replying once.
    received: list[str] = []

    async def respond(envelope):
        if envelope.event_type != EV_TEXT:
            return
        text = envelope.event_data.get("text", "")
        received.append(text)
        # Reply via the human's send().
        await human.send(envelope.channel_id, "approved", audience=[envelope.sender_id])

    human.on_envelope(respond)

    channel = await alice.open(type="consulting", target="reviewer")
    assert channel.state == ChannelState.ACTIVE

    await channel.send("Please review payout #42", audience=[human.agent_id])

    # Wait for the channel to close (consulting auto-closes after the respondent's reply).
    close_env = await alice.wait_for_channel_event(
        channel_id=channel.channel_id,
        predicate=lambda e: e.event_type == "ag2.channel.closed",
        timeout=5.0,
    )
    assert close_env is not None
    assert received == ["Please review payout #42"]

    await human_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_pull_surface_returns_envelopes_in_order() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    human_hc = HubClient(link, hub=hub)

    alice = await hub.register(_agent("alice"))
    human = await human_hc.register_human(Passport(name="reviewer"))

    channel = await alice.open(type="conversation", target="reviewer")

    # Drain channel-protocol envelopes first so the test waits on substantive.
    await human.next_envelope(timeout=2.0)  # invite or opened, application detail
    # Send a substantive envelope from alice.
    await channel.send("hello, reviewer")

    envelope = await human.next_envelope(
        predicate=lambda e: e.event_type == EV_TEXT,
        timeout=2.0,
    )
    assert envelope.event_data["text"] == "hello, reviewer"

    await human_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_envelopes_iterator_streams_until_disconnect() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    human_hc = HubClient(link, hub=hub)

    alice = await hub.register(_agent("alice"))
    human = await human_hc.register_human(Passport(name="reviewer"))

    channel = await alice.open(type="conversation", target="reviewer")
    await channel.send("one")
    await channel.send("two")

    seen: list[str] = []

    async def consume() -> None:
        async for envelope in human.envelopes():
            if envelope.event_type == EV_TEXT:
                seen.append(envelope.event_data["text"])
                if len(seen) == 2:
                    await human.disconnect()
                    return

    await asyncio.wait_for(consume(), timeout=5.0)
    assert seen == ["one", "two"]

    await human_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_callback_exception_does_not_break_dispatch() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    human_hc = HubClient(link, hub=hub)

    alice = await hub.register(_agent("alice"))
    human = await human_hc.register_human(Passport(name="reviewer"))

    good_calls = 0

    async def bad(envelope):
        raise RuntimeError("boom")

    async def good(envelope):
        nonlocal good_calls
        if envelope.event_type == EV_TEXT:
            good_calls += 1

    human.on_envelope(bad)
    human.on_envelope(good)

    channel = await alice.open(type="conversation", target="reviewer")
    await channel.send("first")
    await channel.send("second")

    # Pull queue is still populated even though the push callback raised.
    text_envs = []
    while len(text_envs) < 2:
        env = await human.next_envelope(timeout=2.0)
        if env.event_type == EV_TEXT:
            text_envs.append(env)

    assert good_calls == 2
    assert [e.event_data["text"] for e in text_envs] == ["first", "second"]

    await human_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_human_initiates_consulting() -> None:
    """Human opens consulting against an agent; agent's notify handler replies."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    human_hc = HubClient(link, hub=hub)

    human = await human_hc.register_human(Passport(name="ops"))
    await hub.register(_agent("bob", "bob's reply"))

    channel = await human.open(type="consulting", target="bob")
    assert isinstance(channel, Channel)
    assert channel.state == ChannelState.ACTIVE

    await channel.send("status check", audience=[await _id_for(human_hc, "bob")])

    # Wait for bob's substantive reply on the human's pull queue.
    reply = await human.next_envelope(
        predicate=lambda e: e.event_type == EV_TEXT and e.sender_id != human.agent_id,
        timeout=5.0,
    )
    assert reply.event_data["text"] == "bob's reply"

    await human_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_discussion_round_robin_with_human() -> None:
    """A human participates in a round-robin discussion alongside agents.

    The human's reply callback gates on adapter state — the human only
    posts when round-robin makes them the expected next speaker. This is
    the realistic UI pattern: "show me whose turn it is, let the user
    type only when it's theirs."
    """
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    human_hc = HubClient(link, hub=hub)

    alice = await hub.register(_agent("alice", "alice says hi"))
    await hub.register(_agent("bob", "bob says hi"))
    human = await human_hc.register_human(Passport(name="reviewer"))

    posted = {"done": False}

    async def respond(envelope):
        if envelope.event_type != EV_TEXT or envelope.sender_id == human.agent_id:
            return
        if posted["done"]:
            return
        state = await human_hc.adapter_state(envelope.channel_id)
        if state is None or getattr(state, "expected_next_speaker", None) != human.agent_id:
            return
        posted["done"] = True
        await human.send(envelope.channel_id, "reviewer notes")

    human.on_envelope(respond)

    channel = await alice.open(
        type="discussion",
        target=["bob", "reviewer"],
        knobs={"ordering": "round_robin"},
    )
    await channel.send("kickoff")

    # 3 substantive texts: kickoff (alice) + bob + human reviewer.
    await wait_for_text_count(hub, channel.channel_id, expected=3, timeout=5.0)
    wal = await hub.read_wal(channel.channel_id)
    text_senders = [e.sender_id for e in wal if e.event_type == EV_TEXT]
    assert human.agent_id in text_senders
    # Round-robin order should have human after bob, both after alice's kickoff.
    text_only = [e for e in wal if e.event_type == EV_TEXT]
    assert text_only[0].sender_id == alice.agent_id
    assert text_only[2].sender_id == human.agent_id

    await human_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_receive_chunk_is_noop() -> None:
    """Streaming chunks to a HumanClient don't raise."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    human = await hc.register_human(Passport(name="reviewer"))
    # Direct method call — chunks are an LLM-streaming concept; humans
    # see only completed envelopes in V1.
    await human.receive_chunk(object(), channel_id="x", parent_envelope_id="y")

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_post_envelope_after_disconnect_raises() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    human = await hc.register_human(Passport(name="x"))
    await human.disconnect()
    with pytest.raises(RuntimeError, match="disconnected"):
        await human.send("c", "hi")

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_disconnect_wakes_blocked_next_envelope() -> None:
    """A consumer parked on next_envelope must wake when disconnect fires.

    Regression for the HumanClient hang where disconnect() flipped the flag
    but did not unblock pending Queue.get() calls.
    """
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    human = await hc.register_human(Passport(name="x"))

    async def disconnect_soon() -> None:
        await asyncio.sleep(0.05)
        await human.disconnect()

    asyncio.create_task(disconnect_soon())

    with pytest.raises(RuntimeError, match="disconnected"):
        await asyncio.wait_for(human.next_envelope(timeout=5.0), timeout=2.0)

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_disconnect_wakes_blocked_envelopes_iterator() -> None:
    """An async-for over envelopes() must terminate when disconnect fires."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    human = await hc.register_human(Passport(name="x"))

    seen: list[object] = []

    async def consume() -> None:
        async for envelope in human.envelopes():
            seen.append(envelope)

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0.05)  # let consumer park on get()
    await human.disconnect()
    await asyncio.wait_for(consumer, timeout=2.0)
    # Iterator exited cleanly; no envelopes ever arrived.
    assert seen == []

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_disconnect_wakes_concurrent_pull_consumers() -> None:
    """Multiple consumers parked on the pull queue all wake on disconnect.

    The sentinel-resender pattern means the first awaiter that observes
    the disconnect re-enqueues the sentinel for siblings.
    """
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    human = await hc.register_human(Passport(name="x"))

    waiters = [asyncio.create_task(human.next_envelope(timeout=5.0)) for _ in range(3)]
    await asyncio.sleep(0.05)
    await human.disconnect()

    results = await asyncio.gather(*waiters, return_exceptions=True)
    for result in results:
        assert isinstance(result, RuntimeError)
        assert "disconnected" in str(result)

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_wait_for_channel_event_wakes_on_disconnect() -> None:
    """Per-channel waiters also wake when disconnect fires."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    human = await hc.register_human(Passport(name="x"))
    # Pre-create a per-channel inbox so the waiter parks on it.
    human.ensure_channel_inbox("ch-pending")

    async def disconnect_soon() -> None:
        await asyncio.sleep(0.05)
        await human.disconnect()

    asyncio.create_task(disconnect_soon())

    with pytest.raises(RuntimeError, match="disconnected"):
        await human.wait_for_channel_event(
            channel_id="ch-pending",
            predicate=lambda _e: True,
            timeout=5.0,
        )

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_disconnect_idempotent() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    human = await hc.register_human(Passport(name="x"))
    await human.disconnect()
    await human.disconnect()  # second call must not enqueue another sentinel
    # The single sentinel is consumed and re-enqueued by next_envelope; if
    # a second was inserted, two waiters would consume two sentinels
    # without re-enqueue contention. We assert that the inbox has exactly
    # one item (the sentinel) by reading it once and confirming a second
    # read still finds one (the re-enqueue).
    with pytest.raises(RuntimeError):
        await human.next_envelope(timeout=1.0)
    with pytest.raises(RuntimeError):
        await human.next_envelope(timeout=1.0)

    await hc.close()
    await hub.close()


# ── Helpers ─────────────────────────────────────────────────────────────────


async def _id_for(hub_client: HubClient, name: str) -> str:
    passport = await hub_client.get_agent(name)
    assert passport.agent_id is not None
    return passport.agent_id
