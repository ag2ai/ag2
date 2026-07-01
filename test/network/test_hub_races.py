# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Hub-mechanic edge cases — registration races, endpoint binding,
channel-id state, concurrent dispatch.

These tests target hub-level invariants exercised through the public
``HubClient`` / ``AgentClient`` surface and through ``Hub`` directly
where the test scenario is hub-only (e.g. raw envelope post from a
nonexistent sender). They run in milliseconds against
``MemoryKnowledgeStore`` + ``LocalLink``.
"""

import asyncio

import pytest

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_TEXT,
    AccessDeniedError,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    NotFoundError,
    Passport,
    ProtocolError,
    Resume,
)
from ag2.network.channel import ChannelState

from ._helpers import ScriptedConfig


def _agent(name: str) -> Agent:
    # ScriptedConfig() with no replies returns "" — the default notify
    # handler treats an empty body as "don't send," so no reply
    # cascades. The default handler still auto-acks invites, which is
    # what these tests need (the alternative — overriding on_envelope
    # to silence the agent — would break invite acks and hang
    # channel creation at invite_ack_timeout).
    return Agent(name=name, config=ScriptedConfig())


# ── Registration races ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_register_same_name_serializes() -> None:
    """Two concurrent registers with the same name: one wins, one raises.

    The hub's ``_registration_lock`` serialises identity mutations.
    Without it, both could stamp the same name → divergent agent_ids
    → orphaned identity files.
    """
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    async def reg() -> tuple[bool, str]:
        try:
            client = await hc.register(_agent("dup"), Passport(name="dup"), Resume())
            return True, client.agent_id
        except ProtocolError as exc:
            return False, str(exc)

    results = await asyncio.gather(reg(), reg(), reg(), return_exceptions=False)
    successes = [r for r in results if r[0]]
    failures = [r for r in results if not r[0]]
    assert len(successes) == 1, f"expected exactly 1 winner, got {len(successes)}"
    assert len(failures) == 2
    for ok, msg in failures:
        assert "already registered" in msg

    # On-disk state matches: only one agent_id, one passport file.
    listed = await hub.list_agents()
    assert {p.name for p in listed} == {"dup"}
    assert listed[0].agent_id == successes[0][1]

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_reregister_after_unregister_assigns_new_id() -> None:
    """unregister → register same name produces a different agent_id."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    first = await hc.register(_agent("reuse"), Passport(name="reuse"), Resume())
    first_id = first.agent_id
    await first.unregister()

    second = await hc.register(_agent("reuse"), Passport(name="reuse"), Resume())
    assert second.agent_id != first_id

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_register_collision_does_not_orphan_files() -> None:
    """A failed re-registration must not write any new identity files."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    await hc.register(_agent("first"), Passport(name="first"), Resume())
    with pytest.raises(ProtocolError, match="already registered"):
        await hc.register(_agent("first"), Passport(name="first"), Resume())

    listed = await hub.list_agents()
    assert len(listed) == 1
    assert listed[0].name == "first"

    await hc.close()
    await hub.close()


# ── Unregister mid-flight ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unregister_mid_channel_preserves_wal() -> None:
    """Unregistering a participant after channel activation does NOT erase
    the channel's WAL — the hub keeps channel/task state for audit even
    when an agent leaves."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc_alice = HubClient(link, hub=hub)
    hc_bob = HubClient(link, hub=hub)

    alice = await hc_alice.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await hc_bob.register(_agent("bob"), Passport(name="bob"), Resume())
    bob_id = bob.agent_id

    channel = await alice.open(type="conversation", target="bob")
    await channel.send("hello", audience=[bob_id])

    pre_wal = await hub.read_wal(channel.channel_id)
    assert any(e.event_type == EV_TEXT for e in pre_wal)

    await bob.unregister()

    # WAL still intact — an unregistered agent doesn't garbage-collect
    # channel history.
    post_wal = await hub.read_wal(channel.channel_id)
    assert post_wal == pre_wal

    # Bob is gone from registry.
    with pytest.raises(NotFoundError):
        await hub.get_agent(bob_id)

    # Alice can still send (no-one to deliver to, but the WAL appends).
    await channel.send("anyone there?", audience=[bob_id])
    final_wal = await hub.read_wal(channel.channel_id)
    assert sum(1 for e in final_wal if e.event_type == EV_TEXT) == 2

    await hc_alice.close()
    await hc_bob.close()
    await hub.close()


@pytest.mark.asyncio
async def test_post_from_unregistered_agent_id_raises() -> None:
    """Hub-side check fires even when sender_id was once valid."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    alice_id = alice.agent_id
    await alice.unregister()

    # Construct an envelope with the dead agent_id and post via the hub
    # directly (bypassing the AgentClient's local "disconnected" guard).
    envelope = Envelope(
        channel_id="any",
        sender_id=alice_id,
        audience=None,
        event_type=EV_TEXT,
        event_data={"text": "ghost"},
    )
    with pytest.raises(NotFoundError, match="sender not registered"):
        await hub.post_envelope(envelope)

    await hc.close()
    await hub.close()


# ── Endpoint binding ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bind_unattached_endpoint_raises() -> None:
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    with pytest.raises(NotFoundError, match="endpoint not attached"):
        hub.bind_endpoint("never-attached-endpoint-id", alice.agent_id)

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_bind_endpoint_to_unregistered_agent_raises() -> None:
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    # Trigger endpoint creation by registering one real agent — the link
    # opens its endpoint pair on first register.
    hc = HubClient(link, hub=hub)
    await hc.register(_agent("real"), Passport(name="real"), Resume())

    # Pick the existing endpoint id and try to bind a nonexistent agent.
    endpoint_id = next(iter(hub._endpoints_by_id.keys()))
    with pytest.raises(NotFoundError, match="agent not registered"):
        hub.bind_endpoint(endpoint_id, "nonexistent-agent-id")

    await hc.close()
    await hub.close()


# ── Concurrent post_envelope ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_posts_same_channel_serialize() -> None:
    """Per-channel WAL lock serialises append + fold + on_accepted.

    Conversation adapter accepts unbounded sends from either side, so
    five parallel posts from alice all land in WAL with no corruption,
    no gaps, no duplicate envelope_ids.
    """
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await hc.register(_agent("bob"), Passport(name="bob"), Resume())

    channel = await alice.open(type="conversation", target="bob")

    async def post(i: int) -> str:
        return await channel.send(f"msg-{i}", audience=[bob.agent_id])

    ids = await asyncio.gather(*[post(i) for i in range(5)])
    assert len(set(ids)) == 5, "envelope_ids must be unique"

    wal = await hub.read_wal(channel.channel_id)
    text_envelopes = [e for e in wal if e.event_type == EV_TEXT]
    assert len(text_envelopes) == 5
    bodies = sorted(e.event_data["text"] for e in text_envelopes)
    assert bodies == [f"msg-{i}" for i in range(5)]

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_concurrent_posts_different_channels_independent() -> None:
    """Different channels hold different locks — no cross-contamination."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await hc.register(_agent("bob"), Passport(name="bob"), Resume())
    carol = await hc.register(_agent("carol"), Passport(name="carol"), Resume())

    s1 = await alice.open(type="conversation", target="bob")
    s2 = await alice.open(type="conversation", target="carol")

    await asyncio.gather(
        s1.send("to bob 1", audience=[bob.agent_id]),
        s2.send("to carol 1", audience=[carol.agent_id]),
        s1.send("to bob 2", audience=[bob.agent_id]),
        s2.send("to carol 2", audience=[carol.agent_id]),
    )

    wal1 = [e for e in await hub.read_wal(s1.channel_id) if e.event_type == EV_TEXT]
    wal2 = [e for e in await hub.read_wal(s2.channel_id) if e.event_type == EV_TEXT]
    assert len(wal1) == 2 and len(wal2) == 2
    assert {e.event_data["text"] for e in wal1} == {"to bob 1", "to bob 2"}
    assert {e.event_data["text"] for e in wal2} == {"to carol 1", "to carol 2"}

    await hc.close()
    await hub.close()


# ── Channel-state edges ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_post_to_terminal_channel_raises() -> None:
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await hc.register(_agent("bob"), Passport(name="bob"), Resume())

    channel = await alice.open(type="conversation", target="bob")
    await channel.close(reason="test-close")
    refreshed = await channel.info()
    assert refreshed.state == ChannelState.CLOSED

    with pytest.raises(ProtocolError, match="closed|expired"):
        await channel.send("after close", audience=[bob.agent_id])

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_post_to_unknown_channel_raises() -> None:
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())

    envelope = Envelope(
        channel_id="no-such-channel",
        sender_id=alice.agent_id,
        audience=None,
        event_type=EV_TEXT,
        event_data={"text": "hi"},
    )
    with pytest.raises(NotFoundError, match="channel not found"):
        await alice.send_envelope(envelope)

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_create_channel_with_unknown_participant_raises() -> None:
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())

    with pytest.raises(NotFoundError):
        await hub.create_channel(
            creator_id=alice.agent_id,
            manifest_type="conversation",
            participants=["nonexistent-uuid"],
        )

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_create_channel_duplicate_participant_raises() -> None:
    """Listing the same participant twice trips a ProtocolError before
    any persistence."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await hc.register(_agent("bob"), Passport(name="bob"), Resume())

    with pytest.raises(ProtocolError, match="listed twice"):
        await hub.create_channel(
            creator_id=alice.agent_id,
            manifest_type="discussion",
            participants=[bob.agent_id, bob.agent_id],
        )

    # No channel leaked into the registry.
    assert await hub.list_channels() == []

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_create_channel_empty_participants_raises() -> None:
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())

    with pytest.raises(ProtocolError, match="at least one"):
        await hub.create_channel(
            creator_id=alice.agent_id,
            manifest_type="conversation",
            participants=[],
        )

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_create_channel_unknown_manifest_raises() -> None:
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await hc.register(_agent("bob"), Passport(name="bob"), Resume())

    with pytest.raises(NotFoundError, match="no adapter"):
        await hub.create_channel(
            creator_id=alice.agent_id,
            manifest_type="nonexistent_protocol",
            participants=[bob.agent_id],
        )

    await hc.close()
    await hub.close()


# ── Lifecycle ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hub_close_idempotent() -> None:
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    await hub.close()
    await hub.close()  # second call must not raise


@pytest.mark.asyncio
async def test_hub_close_with_active_channels_clean_shutdown() -> None:
    """Closing a hub while channels are open should not leak endpoint tasks."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await hc.register(_agent("bob"), Passport(name="bob"), Resume())

    await alice.open(type="conversation", target="bob")
    assert hub._endpoint_tasks, "expected at least one endpoint task"

    await hc.close()
    await hub.close()

    # All endpoint tasks gone after close.
    assert all(t.done() for t in hub._endpoint_tasks)


# ── Outbound-access edge ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_outbound_access_check_runs_before_channel_check() -> None:
    """If a sender has no permission to reach a recipient, hub raises
    AccessDeniedError before checking channel existence — confirms the
    check ordering at the top of post_envelope."""
    from ag2.network import AccessBlock, Rule

    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(
        _agent("alice"),
        Passport(name="alice"),
        Resume(),
        rule=Rule(access=AccessBlock(outbound_to=["only-carol"])),
    )
    bob = await hc.register(_agent("bob"), Passport(name="bob"), Resume())

    envelope = Envelope(
        channel_id="totally-fake-channel",  # would normally raise NotFoundError
        sender_id=alice.agent_id,
        audience=[bob.agent_id],
        event_type=EV_TEXT,
        event_data={"text": "hi"},
    )
    with pytest.raises(AccessDeniedError):
        await alice.send_envelope(envelope)

    await hc.close()
    await hub.close()
