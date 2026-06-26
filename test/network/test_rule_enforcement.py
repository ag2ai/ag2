# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Rule enforcement — access blocks, concurrency caps, TTL parsing.

Validates every enforced field on ``Rule``:

* ``access.outbound_to`` / ``access.inbound_from`` — glob over peer name
* ``limits.delegation_depth`` — max envelope ``depth`` (0 disables)
* ``limits.max_concurrent_channels`` — capped on creator (0 disables)
* ``limits.max_concurrent_tasks`` — capped on owner via ``observe_task``
* ``limits.inbox.max_pending`` — backpressure on dispatch (substantive only)
* ``limits.channel_ttl_default`` — drives ``ChannelMetadata.expires_at``

The per-minute throttle in ``RateBlock`` is stored but not enforced
by the in-process hub (see ``RateBlock`` docstring), so it isn't
tested here.
"""

import asyncio
from datetime import datetime

import pytest

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_TEXT,
    AccessBlock,
    AccessDeniedError,
    Envelope,
    Hub,
    HubClient,
    InboxFull,
    LimitsBlock,
    LocalLink,
    Passport,
    Resume,
    Rule,
)
from ag2.network.rule import InboxBlock, parse_duration
from ag2.task import TaskMetadata, TaskSpec, TaskState

from ._helpers import ScriptedConfig


def _agent(name: str) -> Agent:
    return Agent(name=name, config=ScriptedConfig())


# ── parse_duration ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value,expected",
    [
        ("30s", 30),
        ("15m", 900),
        ("2h", 7200),
        ("4h", 14400),
        ("1d", 86400),
        ("0s", 0),
        (0, 0),
        (3600, 3600),
        ("", 0),
        ("60", 60),
        ("3600", 3600),
    ],
)
def test_parse_duration_valid(value, expected) -> None:
    assert parse_duration(value) == expected


@pytest.mark.parametrize("value", ["abc", "5x", "10y", "fast"])
def test_parse_duration_invalid_raises(value) -> None:
    with pytest.raises(ValueError):
        parse_duration(value)


# ── Access: outbound_to + inbound_from ──────────────────────────────────────


@pytest.mark.asyncio
async def test_outbound_to_glob_allows_pattern() -> None:
    """outbound_to supports glob via fnmatch — `bot-*` matches `bot-bob`.

    The sender's own access rule must NOT block protocol broadcasts
    that include the sender in their audience (``EV_CHANNEL_OPENED`` /
    ``EV_CHANNEL_CLOSED``). The hub skips the outbound check when
    ``recipient_id == sender_id``, so a creator with restrictive
    ``outbound_to`` (not including themselves) can still create
    channels and receive their own channel-state notifications.
    """
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(
        _agent("alice"),
        rule=Rule(access=AccessBlock(outbound_to=["bot-*"])),  # alice not in own whitelist
    )
    bob = await hub.register(_agent("bot-bob"))
    eve = await hub.register(_agent("user-eve"))

    # Reaching bot-bob is allowed.
    channel_ok = await alice.open(type="conversation", target="bot-bob")
    await channel_ok.send("greetings", audience=[bob.agent_id])

    # Reaching user-eve is denied.
    envelope = Envelope(
        channel_id=channel_ok.channel_id,
        sender_id=alice.agent_id,
        audience=[eve.agent_id],
        event_type=EV_TEXT,
        event_data={"text": "x"},
    )
    with pytest.raises(AccessDeniedError):
        await alice.send_envelope(envelope)

    await hub.close()


@pytest.mark.asyncio
async def test_outbound_to_self_send_always_allowed() -> None:
    """A sender posting to their own agent_id passes the outbound check
    even when their ``outbound_to`` whitelist excludes their own name."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(
        _agent("alice"),
        Passport(name="alice"),
        Resume(),
        rule=Rule(access=AccessBlock(outbound_to=["nobody-*"])),
    )
    bob = await hc.register(_agent("bob"), Passport(name="bob"), Resume())

    # Alice can self-only-broadcast (audience contains alice + bob).
    # Without the self-skip in the outbound check, this would raise
    # AccessDeniedError before reaching bob.
    # Use a channel manifest that allows arbitrary text — conversation works.
    # First open with bob (bob is in audience for invite — but alice's
    # outbound_to=["nobody-*"] doesn't match "bob" → invite denied).
    # So instead, register bob under a name matching the whitelist:
    await hc.unregister_agent(bob.agent_id)
    await hc.register(_agent("nobody-bob"), Passport(name="nobody-bob"), Resume())

    channel = await alice.open(type="conversation", target="nobody-bob")
    # Audience includes alice — without the self-skip fix, opening would fail.
    assert channel.metadata.state.value == "active"

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_inbound_from_blocks_dispatch_not_post() -> None:
    """inbound_from is a recipient-side filter — the sender's post
    succeeds (WAL appends) but no NotifyFrame reaches the blocked-from
    recipient.

    Setup uses ``set_rule`` mid-channel because pre-channel
    ``inbound_from`` blocks are now caught at ``create_channel`` time
    by the fail-fast check. This test specifically validates the
    in-flight dispatch filter.
    """
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    received_by_bob: list[Envelope] = []

    # Capture wrapper — preserves the default handler behaviour
    # (auto-ack invites, etc.) but lets us observe what arrived.
    original_handler = bob._on_envelope
    assert original_handler is not None

    async def capture(env: Envelope) -> None:
        received_by_bob.append(env)
        await original_handler(env)

    bob.on_envelope(capture)

    channel = await alice.open(type="conversation", target="bob")
    assert sum(1 for e in received_by_bob if e.event_type == EV_TEXT) == 0

    # Tighten bob's inbound filter — alice no longer matches.
    await bob.set_rule(Rule(access=AccessBlock(inbound_from=["nobody-*"])))

    await channel.send("filtered text", audience=[bob.agent_id])
    await asyncio.sleep(0.05)

    # WAL has the text; bob's handler did NOT see it.
    wal = await hub.read_wal(channel.channel_id)
    assert any(e.event_type == EV_TEXT and e.event_data["text"] == "filtered text" for e in wal)

    post_text_received = [e for e in received_by_bob if e.event_data.get("text") == "filtered text"]
    assert post_text_received == []

    await hub.close()


@pytest.mark.asyncio
async def test_inbound_from_blocks_create_channel_fast_fails() -> None:
    """Creating a channel with an invitee whose ``inbound_from`` blocks
    the creator must raise ``AccessDeniedError`` immediately, not hang
    until ``invite_ack_timeout``.

    Without the pre-flight check in ``create_channel``, the dispatch
    path silently filters the invite, the recipient never sees it,
    never acks, and the creator times out with a generic ``ProtocolError``.
    """
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    await hub.register(
        _agent("bob"),
        rule=Rule(access=AccessBlock(inbound_from=["carol-*"])),
    )

    with pytest.raises(AccessDeniedError, match="does not accept inbound"):
        await alice.open(type="conversation", target="bob")

    # No channel leaked into the registry.
    assert await hub.list_channels() == []

    await hub.close()


# ── delegation_depth ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delegation_depth_at_cap_accepted_above_rejected() -> None:
    """depth == cap is accepted (`>` check); depth == cap+1 is rejected."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(
        _agent("alice"),
        rule=Rule(limits=LimitsBlock(delegation_depth=3)),
    )
    bob = await hub.register(_agent("bob"))

    channel = await alice.open(type="conversation", target="bob")

    # depth=3 (== cap) → ok
    await channel.send("at-cap", audience=[bob.agent_id], depth=3)
    # depth=4 (> cap) → denied
    with pytest.raises(AccessDeniedError, match="delegation_depth"):
        await channel.send("over-cap", audience=[bob.agent_id], depth=4)

    await hub.close()


@pytest.mark.asyncio
async def test_delegation_depth_zero_disables_cap() -> None:
    """delegation_depth=0 means unlimited."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(
        _agent("alice"),
        rule=Rule(limits=LimitsBlock(delegation_depth=0)),
    )
    bob = await hub.register(_agent("bob"))

    channel = await alice.open(type="conversation", target="bob")
    # Even an absurd depth should pass.
    await channel.send("any-depth", audience=[bob.agent_id], depth=99999)

    await hub.close()


# ── max_concurrent_channels ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_concurrent_channels_cap_blocks_new_creates() -> None:
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(
        _agent("alice"),
        rule=Rule(limits=LimitsBlock(max_concurrent_channels=2)),
    )
    await hub.register(_agent("bob"))
    await hub.register(_agent("carol"))
    await hub.register(_agent("dave"))

    # 2 concurrent channels ok.
    s1 = await alice.open(type="conversation", target="bob")
    await alice.open(type="conversation", target="carol")

    # Third attempt → AccessDeniedError before any persistence.
    with pytest.raises(AccessDeniedError, match="max_concurrent_channels"):
        await alice.open(type="conversation", target="dave")

    # Closing one frees the slot.
    await s1.close()
    await alice.open(type="conversation", target="dave")  # ok now

    await hub.close()


@pytest.mark.asyncio
async def test_max_concurrent_channels_zero_disables() -> None:
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(
        _agent("alice"),
        rule=Rule(limits=LimitsBlock(max_concurrent_channels=0)),
    )
    for name in ("bob", "carol", "dave", "erin"):
        await hub.register(_agent(name))

    # Open 4 — no cap.
    for name in ("bob", "carol", "dave", "erin"):
        await alice.open(type="conversation", target=name)

    channels = await hub.list_channels(agent_id=alice.agent_id)
    assert len(channels) == 4

    await hub.close()


# ── max_concurrent_tasks (via observe_task) ─────────────────────────────────


@pytest.mark.asyncio
async def test_max_concurrent_tasks_blocks_observe() -> None:
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(
        _agent("alice"),
        rule=Rule(limits=LimitsBlock(max_concurrent_tasks=2)),
    )

    # Two non-terminal tasks ok.
    for i in range(2):
        await hub.observe_task(
            TaskMetadata(
                task_id=f"task-{i}",
                owner_id=alice.agent_id,
                spec=TaskSpec(title=f"t{i}"),
                state=TaskState.RUNNING,
            )
        )

    # Third → denied.
    with pytest.raises(AccessDeniedError, match="max_concurrent_tasks"):
        await hub.observe_task(
            TaskMetadata(
                task_id="task-2",
                owner_id=alice.agent_id,
                spec=TaskSpec(title="t2"),
                state=TaskState.RUNNING,
            )
        )

    # Terminating one frees the slot.
    await hub.update_task("task-0", state=TaskState.COMPLETED)
    await hub.observe_task(
        TaskMetadata(
            task_id="task-2",
            owner_id=alice.agent_id,
            spec=TaskSpec(title="t2"),
            state=TaskState.RUNNING,
        )
    )

    await hub.close()


# ── inbox.max_pending ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inbox_max_pending_rejects_when_full() -> None:
    """Substantive sends to a recipient at inbox capacity raise InboxFull.

    The counter increments on dispatch; bob never replies (ScriptedConfig
    returns ""), so it never decrements. Once at cap, alice's next send
    is rejected before WAL append.
    """
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(
        _agent("bob"),
        rule=Rule(limits=LimitsBlock(inbox=InboxBlock(max_pending=2))),
    )

    channel = await alice.open(type="conversation", target="bob")
    await asyncio.sleep(0.05)  # let channel-protocol dispatch settle

    # Cap=2 means the counter must be < 2 to allow a new send.
    # Protocol envelopes don't increment, so two substantive sends
    # bring bob's count to 2 → third raises.
    await channel.send("msg-1", audience=[bob.agent_id])
    await channel.send("msg-2", audience=[bob.agent_id])
    with pytest.raises(InboxFull):
        await channel.send("msg-3", audience=[bob.agent_id])

    await hub.close()


@pytest.mark.asyncio
async def test_inbox_protocol_events_bypass_capacity() -> None:
    """Protocol envelopes (invite/ack/open/close) must always reach the
    recipient, regardless of inbox capacity. Otherwise a blocked invite
    ack would deadlock the channel machine."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    # Cap of 1 — easy to overflow with substantive but invites must
    # still be dispatched.
    await hub.register(
        _agent("bob"),
        rule=Rule(limits=LimitsBlock(inbox=InboxBlock(max_pending=1))),
    )

    # If invites failed the inbox cap, this would hang at invite ack
    # timeout. Success here confirms the protocol-event bypass.
    channel = await alice.open(type="conversation", target="bob")
    assert channel.metadata.state.value == "active"

    await hub.close()


# ── channel_ttl_default ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_channel_ttl_default_drives_expires_at() -> None:
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(
        _agent("alice"),
        rule=Rule(limits=LimitsBlock(channel_ttl_default="1h")),
    )
    await hub.register(_agent("bob"))

    channel = await alice.open(type="conversation", target="bob")
    assert channel.metadata.expires_at is not None
    created = datetime.fromisoformat(channel.metadata.created_at)
    expires = datetime.fromisoformat(channel.metadata.expires_at)
    delta = (expires - created).total_seconds()
    # 1h ± 1s
    assert 3599 <= delta <= 3601

    await hub.close()


@pytest.mark.asyncio
async def test_channel_ttl_per_channel_override_wins() -> None:
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(
        _agent("alice"),
        rule=Rule(limits=LimitsBlock(channel_ttl_default="1h")),
    )
    await hub.register(_agent("bob"))

    channel = await alice.open(type="conversation", target="bob", ttl="30m")
    expires = datetime.fromisoformat(channel.metadata.expires_at)
    created = datetime.fromisoformat(channel.metadata.created_at)
    delta = (expires - created).total_seconds()
    assert 1799 <= delta <= 1801

    await hub.close()


@pytest.mark.asyncio
async def test_channel_ttl_zero_no_expiry() -> None:
    """ttl=0 → no expires_at stamped."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))

    channel = await alice.open(type="conversation", target="bob", ttl=0)
    assert channel.metadata.expires_at is None

    await hub.close()
