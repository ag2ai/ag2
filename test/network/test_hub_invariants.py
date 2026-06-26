# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Hub correctness invariants — registration, concurrency, dispatch, projection.

Covers:

* ``Hub.unregister`` deletes on-disk identity files so ``hydrate()``
  does not re-load unregistered agents.
* Concurrency caps + inbox limits enforced
  (``max_concurrent_channels`` / ``max_concurrent_tasks`` /
  ``InboxBlock.max_pending``).
* ``delegate`` pre-creates the inbox before sending so a fast reply
  cannot be dropped (race fix).
* ``Hub.register`` rejects a duplicate ``name`` so the prior passport
  / resume / rule are not orphaned on disk.
* ``delegate`` fails fast on channel close / reject / expiry rather
  than blocking until the 300s default timeout.
* Expectation dedup keyed by ``(index, name, violator)`` so two
  same-named expectations don't suppress each other.
* ``record_observation`` is idempotent per ``task_id`` so cascade-
  terminal events don't double-count ``Resume.observed.n``.
* ``set_resume`` re-indexes ``claimed_capabilities`` so newly claimed
  caps surface under ``peers(action="find", capability=...)``.
"""

import asyncio
import contextlib
import json
from typing import Any

import pytest

from ag2 import Agent, Context
from ag2.events import ToolCallEvent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_INVITE,
    EV_CHANNEL_INVITE_ACK,
    EV_TEXT,
    AccessDeniedError,
    Envelope,
    Hub,
    HubClient,
    InboxFull,
    LocalLink,
    Passport,
    ProtocolError,
    Resume,
    Rule,
)
from ag2.network.adapters.conversation import (
    CONVERSATION_TYPE,
)
from ag2.network.channel import (
    ChannelManifest,
    ChannelMetadata,
    ChannelState,
    Expectation,
    Participant,
    ParticipantRole,
    ParticipantSchema,
)
from ag2.network.client.tools.delegate import make_delegate_tool
from ag2.network.hub.expectations import (
    AcksWithinEvaluator,
)
from ag2.network.hub.layout import (
    by_capability_path,
    passport_path,
    resume_path,
    rule_path,
    skill_path,
)
from ag2.network.rule import InboxBlock, LimitsBlock
from ag2.stream import MemoryStream
from ag2.task import (
    TaskMetadata,
    TaskSpec,
    TaskState,
)
from ag2.testing import TestConfig

from ._helpers import _MockClock


def _agent(name: str, *events: object) -> Agent:
    return Agent(name=name, config=TestConfig(*events))


async def _invoke(tool: Any, args: dict, *, dependencies: dict | None = None) -> Any:
    """Invoke a ``FunctionTool`` directly and return its underlying value."""
    event = ToolCallEvent(name=tool.name, arguments=json.dumps(args))
    context = Context(stream=MemoryStream(), dependencies=dependencies or {})
    result_event = await tool(event, context)
    parts = getattr(result_event, "result", None)
    if parts is None or not parts.parts:
        return result_event
    first = parts.parts[0]
    if hasattr(first, "content"):
        return first.content
    if hasattr(first, "data"):
        return first.data
    return first


def _ack_only_handler(client: "HubClient | object"):
    """Build a notify handler that auto-acks invites and ignores everything else.

    Used when a test needs a channel to open (handshake completes) but
    wants the recipient to not respond to ``EV_TEXT`` so the hub's
    inbox-pending counter stays high.
    """

    async def _handle(env: Envelope) -> None:
        if env.event_type != EV_CHANNEL_INVITE:
            return
        ack = Envelope(
            channel_id=env.channel_id,
            sender_id=client.agent_id,
            audience=None,
            event_type=EV_CHANNEL_INVITE_ACK,
            event_data={"channel_id": env.channel_id},
            causation_id=env.envelope_id,
        )
        with contextlib.suppress(Exception):
            await client.send_envelope(ack)

    return _handle


# ── Fix #1: Hub.unregister disk cleanup ─────────────────────────────────────


@pytest.mark.asyncio
async def test_unregister_deletes_disk_files_so_hydrate_forgets_agent() -> None:
    """After ``unregister`` the agent's identity files are gone; a fresh
    ``Hub`` over the same store must not see the agent on hydrate."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(
        _agent("alice"),
        Passport(name="alice"),
        Resume(claimed_capabilities=["math"]),
        skill_md="# Alice\nDoes math.",
        rule=Rule(limits=LimitsBlock(channel_ttl_default="4h")),
    )
    agent_id = alice.agent_id

    # All four on-disk files exist before unregister.
    assert await store.read(passport_path(agent_id)) is not None
    assert await store.read(resume_path(agent_id)) is not None
    assert await store.read(rule_path(agent_id)) is not None
    assert await store.read(skill_path(agent_id)) is not None

    await alice_hc.unregister_agent(agent_id)
    await alice_hc.close()
    await hub.close()

    # All four files are gone post-unregister.
    assert await store.read(passport_path(agent_id)) is None
    assert await store.read(resume_path(agent_id)) is None
    assert await store.read(rule_path(agent_id)) is None
    assert await store.read(skill_path(agent_id)) is None

    # Fresh Hub over the same store hydrates without the unregistered agent.
    hub2 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    assert hub2._passports == {}
    assert hub2._resumes == {}
    assert hub2._rules == {}
    assert hub2._skills == {}
    assert hub2._name_to_id == {}
    assert hub2.agents_with_capability("math") == []
    await hub2.close()


# ── Fix #4: Hub.register rejects duplicate name ──────────────────────────────


@pytest.mark.asyncio
async def test_register_rejects_duplicate_name_raises_protocol_error() -> None:
    """A second ``register`` with the same ``name`` must fail loudly so the
    prior passport / resume / rule do not get orphaned on disk."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    await hub.register(_agent("alice"))

    with pytest.raises(ProtocolError, match="already registered"):
        await hub.register(_agent("alice"))

    await hub.close()


@pytest.mark.asyncio
async def test_unregister_then_reregister_with_same_name_works() -> None:
    """Explicit unregister releases the name so the same identity can rejoin."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    hc = HubClient(link, hub=hub)
    first = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await hc.unregister_agent(first.agent_id)

    second = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    assert second.agent_id != first.agent_id

    await hc.close()
    await hub.close()


# ── Fix #2: concurrency caps + inbox limits ──────────────────────────────────


@pytest.mark.asyncio
async def test_create_channel_enforces_max_concurrent_channels() -> None:
    """Hub rejects ``create_channel`` when the creator has more active
    channels than ``max_concurrent_channels``. ``0`` disables."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    capped = Rule(limits=LimitsBlock(max_concurrent_channels=1))
    alice = await hub.register(_agent("alice"), rule=capped)
    bob = await hub.register(_agent("bob"))
    carol = await hub.register(_agent("carol"))

    await alice.open(type=CONVERSATION_TYPE, target=bob.agent_id)

    with pytest.raises(AccessDeniedError, match="max_concurrent_channels"):
        await alice.open(type=CONVERSATION_TYPE, target=carol.agent_id)

    await hub.close()


@pytest.mark.asyncio
async def test_observe_task_enforces_max_concurrent_tasks() -> None:
    """``observe_task`` rejects new tasks when the owner already has
    ``max_concurrent_tasks`` non-terminal tasks observed."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    capped = Rule(limits=LimitsBlock(max_concurrent_tasks=1))
    alice = await hub.register(_agent("alice"), rule=capped)

    await hub.observe_task(
        TaskMetadata(
            task_id="t1",
            owner_id=alice.agent_id,
            spec=TaskSpec(title="task one"),
            state=TaskState.RUNNING,
            created_at="2026-01-01T00:00:00+00:00",
        )
    )

    with pytest.raises(AccessDeniedError, match="max_concurrent_tasks"):
        await hub.observe_task(
            TaskMetadata(
                task_id="t2",
                owner_id=alice.agent_id,
                spec=TaskSpec(title="task two"),
                state=TaskState.RUNNING,
                created_at="2026-01-01T00:00:01+00:00",
            )
        )

    # Terminal task frees the slot for the next.
    await hub.update_task("t1", state=TaskState.COMPLETED)
    await hub.observe_task(
        TaskMetadata(
            task_id="t3",
            owner_id=alice.agent_id,
            spec=TaskSpec(title="task three"),
            state=TaskState.RUNNING,
            created_at="2026-01-01T00:00:02+00:00",
        )
    )

    await hub.close()


@pytest.mark.asyncio
async def test_post_envelope_enforces_inbox_max_pending() -> None:
    """``post_envelope`` raises ``InboxFull`` when the recipient's inbox
    is at capacity. The counter decrements when the recipient sends."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    bob_capped = Rule(limits=LimitsBlock(inbox=InboxBlock(max_pending=2)))
    alice = await hub.register(_agent("alice"))
    bob = await hub.register(
        _agent("bob"),
        rule=bob_capped,
        attach_plugin=False,
    )
    # Auto-ack invites so the channel opens; ignore EV_TEXT so the
    # hub's pending-inbox counter for bob grows.
    bob.on_envelope(_ack_only_handler(bob))

    channel = await alice.open(type=CONVERSATION_TYPE, target=bob.agent_id)

    def _make(text: str) -> Envelope:
        return Envelope(
            channel_id=channel.channel_id,
            sender_id=alice.agent_id,
            audience=[bob.agent_id],
            event_type=EV_TEXT,
            event_data={"text": text},
        )

    await hub.post_envelope(_make("one"))
    await hub.post_envelope(_make("two"))

    with pytest.raises(InboxFull, match="inbox at capacity"):
        await hub.post_envelope(_make("three"))

    # Bob sending anything decrements his counter, freeing one slot.
    await hub.post_envelope(
        Envelope(
            channel_id=channel.channel_id,
            sender_id=bob.agent_id,
            audience=[alice.agent_id],
            event_type=EV_TEXT,
            event_data={"text": "ack"},
        )
    )
    await hub.post_envelope(_make("three"))  # accepted now

    await hub.close()


@pytest.mark.asyncio
async def test_inbox_pending_cleared_on_unregister() -> None:
    """Unregistering an agent drops its inbox accounting so a future
    re-register starts from zero."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume(), attach_plugin=False)
    bob.on_envelope(_ack_only_handler(bob))

    channel = await alice.open(type=CONVERSATION_TYPE, target=bob.agent_id)
    await hub.post_envelope(
        Envelope(
            channel_id=channel.channel_id,
            sender_id=alice.agent_id,
            audience=[bob.agent_id],
            event_type=EV_TEXT,
            event_data={"text": "x"},
        )
    )

    bob_id = bob.agent_id
    assert hub._inbox_pending.get(bob_id, 0) > 0
    await bob_hc.unregister_agent(bob_id)
    assert bob_id not in hub._inbox_pending

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


# ── Fix #3 / #5: delegate inbox race + fail-fast on close ────────────────────


@pytest.mark.asyncio
async def test_delegate_returns_target_reply_without_dropping_fast_reply() -> None:
    """Delegate must not lose the respondent's reply even if it lands
    immediately after ``channel.send`` (the inbox is pre-created)."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    from ag2.network.policies import AGENT_CLIENT_DEP

    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob", "the answer is 42"))

    delegate_tool = make_delegate_tool(alice)
    result = await _invoke(
        delegate_tool,
        {"target": "bob", "prompt": "what's the answer?"},
        dependencies={AGENT_CLIENT_DEP: alice},
    )
    assert "42" in result

    await hub.close()


@pytest.mark.asyncio
async def test_delegate_to_self_returns_actionable_error() -> None:
    """Delegating to one's own name fails fast with a clear message.

    Regression for #2991: a consulting channel needs a respondent distinct
    from the initiator. When the caller delegates to itself the single
    participant collapses both roles, and the consulting adapter rejects the
    open with the opaque ``consulting requires exactly one respondent`` —
    surfacing to the model as a cryptic error and, in multi-agent pipelines,
    a hang. ``delegate`` must short-circuit with an actionable error instead.
    """
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    from ag2.network.policies import AGENT_CLIENT_DEP

    alice = await hub.register(_agent("alice"))

    delegate_tool = make_delegate_tool(alice)
    result = await _invoke(
        delegate_tool,
        {"target": "alice", "prompt": "answer your own question"},
        dependencies={AGENT_CLIENT_DEP: alice},
    )

    assert "cannot delegate to self" in result
    # The opaque adapter-level error must never reach the caller.
    assert "exactly one respondent" not in result
    # No channel should have been opened for the doomed self-consult.
    assert await hub.list_channels(state=ChannelState.ACTIVE) == []

    await hub.close()


@pytest.mark.asyncio
async def test_delegate_fails_fast_when_channel_closes_before_reply() -> None:
    """If the consulting channel closes / expires before the target replies,
    delegate returns immediately with an error — not after the 300s timeout."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    from ag2.network.policies import AGENT_CLIENT_DEP

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"), attach_plugin=False)
    # Bob acks the invite so the channel opens, but never replies to
    # the prompt — alice's delegate would block until timeout without
    # the fail-fast path.
    bob.on_envelope(_ack_only_handler(bob))

    delegate_tool = make_delegate_tool(alice)

    async def _close_after_handshake() -> None:
        for _ in range(100):
            channels = await hub.list_channels(state=ChannelState.ACTIVE)
            if channels:
                await hub.close_channel(channels[0].channel_id, reason="test_close")
                return
            await asyncio.sleep(0.01)

    closer = asyncio.create_task(_close_after_handshake())
    # Long internal timeout — without the fix this would take 300s.
    # The outer wait_for asserts we return well within 10s.
    result = await asyncio.wait_for(
        _invoke(
            delegate_tool,
            {"target": "bob", "prompt": "hi", "timeout": 300.0},
            dependencies={AGENT_CLIENT_DEP: alice},
        ),
        timeout=10.0,
    )
    await closer
    assert isinstance(result, str)
    assert result.startswith("Error:")
    # Two valid fail-fast wordings, depending on whether the close lands while
    # delegate is awaiting the reply ("... channel closed: test_close") or
    # during the prompt send ("prompt send failed: channel '<id>' is closed").
    # Windows scheduling tends to hit the send path; both are correct.
    lowered = result.lower()
    assert "channel closed" in lowered or "prompt send failed" in lowered, result

    await hub.close()


# ── Fix #6: expectation dedup keyed by identity ──────────────────────────────


@pytest.mark.asyncio
async def test_two_same_name_expectations_both_fire() -> None:
    """A manifest with two ``acks_within`` expectations (same name,
    different ``on_violation``) must allow both to fire on the same
    violator — the dedup key must include the expectation's index."""
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    store = MemoryKnowledgeStore()
    hub = Hub(
        store,
        clock=clock,
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    hub.register_expectation_evaluator(AcksWithinEvaluator())
    audit_calls: list[tuple[str, str]] = []

    class _RecordingAudit:
        name = "audit"

        async def handle(self, h: Hub, sid: str, viol: Any) -> None:
            audit_calls.append((sid, viol.expectation.name))

    class _AltAudit:
        name = "alt_audit"

        async def handle(self, h: Hub, sid: str, viol: Any) -> None:
            audit_calls.append((sid, "alt:" + viol.expectation.name))

    hub.register_violation_handler(_RecordingAudit())
    hub.register_violation_handler(_AltAudit())

    # Build a channel metadata directly — two same-name expectations
    # with different handlers in a single manifest.
    manifest = ChannelManifest(
        type="dual_acks",
        version=1,
        participants=ParticipantSchema(min=2, max=2),
        expectations=[
            Expectation(name="acks_within", on_violation="audit", params={"seconds": 30}),
            Expectation(name="acks_within", on_violation="alt_audit", params={"seconds": 30}),
        ],
    )
    metadata = ChannelMetadata(
        channel_id="s1",
        manifest=manifest,
        creator_id="alice",
        participants=[
            Participant(agent_id="alice", role=ParticipantRole.INITIATOR, order=0),
            Participant(agent_id="bob", role=ParticipantRole.RESPONDENT, order=1),
        ],
        state=ChannelState.PENDING,
        created_at="2026-01-01T00:00:00+00:00",
        pending_acks=["bob"],
    )
    hub._channels["s1"] = metadata
    hub._active_channels["s1"] = metadata

    # Advance past the 30s threshold and tick once.
    clock.advance(45)
    await hub._expectation_tick()

    # Both handlers must have fired exactly once for bob.
    assert ("s1", "acks_within") in audit_calls
    assert ("s1", "alt:acks_within") in audit_calls

    # A second tick at the same clock must NOT re-fire either.
    audit_calls.clear()
    await hub._expectation_tick()
    assert audit_calls == []


# ── Fix #7: task_mirror idempotent observation ───────────────────────────────


@pytest.mark.asyncio
async def test_record_observation_dedups_by_task_id() -> None:
    """Calling ``record_observation`` twice with the same ``task_id``
    must update ``Resume.observed.n`` only once."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))

    await hub.record_observation(
        owner_id=alice.agent_id,
        capability="math",
        outcome=TaskState.COMPLETED,
        task_id="task-A",
    )
    await hub.record_observation(
        owner_id=alice.agent_id,
        capability="math",
        outcome=TaskState.COMPLETED,
        task_id="task-A",
    )

    resume = await hub.get_resume(alice.agent_id)
    assert resume.observed["math"].n == 1
    assert resume.observed["math"].completed == 1

    # A different task_id is recorded independently.
    await hub.record_observation(
        owner_id=alice.agent_id,
        capability="math",
        outcome=TaskState.FAILED,
        task_id="task-B",
    )
    resume = await hub.get_resume(alice.agent_id)
    assert resume.observed["math"].n == 2
    assert resume.observed["math"].failed == 1

    await hub.close()


# ── Edge cases the per-fix tests above don't reach ───────────────────────────


@pytest.mark.asyncio
async def test_concurrency_caps_zero_disables() -> None:
    """``max_concurrent_channels=0`` and ``max_concurrent_tasks=0`` both
    disable their respective caps — the documented "0 = unlimited"
    convention used elsewhere (delegation_depth, inbox.max_pending)."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    no_caps = Rule(limits=LimitsBlock(max_concurrent_channels=0, max_concurrent_tasks=0))
    alice = await hub.register(_agent("alice"), rule=no_caps)

    # Five concurrent conversations open without raising.
    for name in ("bob", "carol", "dave", "erin", "frank"):
        peer = await hub.register(_agent(name))
        await alice.open(type=CONVERSATION_TYPE, target=peer.agent_id)

    # Five concurrent tasks observed without raising.
    for i in range(5):
        await hub.observe_task(
            TaskMetadata(
                task_id=f"task-{i}",
                owner_id=alice.agent_id,
                spec=TaskSpec(title=f"task {i}"),
                state=TaskState.RUNNING,
                created_at=f"2026-01-01T00:00:0{i}+00:00",
            )
        )

    await hub.close()


@pytest.mark.asyncio
async def test_inbox_capacity_does_not_block_protocol_events() -> None:
    """Protocol envelopes (invite / ack / opened / closed) bypass the
    inbox-pending counter so a busy agent can still receive new channel
    opens. Without this carve-out a recipient at capacity would never
    learn about a new invite."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    bob_capped = Rule(limits=LimitsBlock(inbox=InboxBlock(max_pending=1)))
    alice = await hub.register(_agent("alice"))
    bob = await hub.register(
        _agent("bob"),
        rule=bob_capped,
        attach_plugin=False,
    )
    bob.on_envelope(_ack_only_handler(bob))
    carol = await hub.register(_agent("carol"))

    # Saturate bob's substantive-inbox budget via channel 1.
    s1 = await alice.open(type=CONVERSATION_TYPE, target=bob.agent_id)
    await hub.post_envelope(
        Envelope(
            channel_id=s1.channel_id,
            sender_id=alice.agent_id,
            audience=[bob.agent_id],
            event_type=EV_TEXT,
            event_data={"text": "fill"},
        )
    )
    assert hub._inbox_pending.get(bob.agent_id, 0) >= 1

    # Carol opens a fresh channel with bob — INVITE / ACK / OPENED must
    # not be rejected by the inbox check; bob's auto-ack runs and the
    # channel reaches ACTIVE.
    s2 = await carol.open(type=CONVERSATION_TYPE, target=bob.agent_id)
    assert s2.state == ChannelState.ACTIVE

    await hub.close()


@pytest.mark.asyncio
async def test_fired_violations_cleared_on_terminal_channel_transition() -> None:
    """``_fired_violations[channel_id]`` is dropped when the channel
    transitions to a terminal state. Without this, a long-running hub
    accumulates dead dedup entries for every channel that ever fired
    a violation."""
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    store = MemoryKnowledgeStore()
    hub = Hub(
        store,
        clock=clock,
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    hub.register_expectation_evaluator(AcksWithinEvaluator())

    class _Recorder:
        name = "audit"

        async def handle(self, h: Hub, sid: str, viol: Any) -> None:
            return None

    hub.register_violation_handler(_Recorder())

    manifest = ChannelManifest(
        type="testtype",
        version=1,
        participants=ParticipantSchema(min=2, max=2),
        expectations=[
            Expectation(name="acks_within", on_violation="audit", params={"seconds": 30}),
        ],
    )
    metadata = ChannelMetadata(
        channel_id="s1",
        manifest=manifest,
        creator_id="alice",
        participants=[
            Participant(agent_id="alice", role=ParticipantRole.INITIATOR, order=0),
            Participant(agent_id="bob", role=ParticipantRole.RESPONDENT, order=1),
        ],
        state=ChannelState.PENDING,
        created_at="2026-01-01T00:00:00+00:00",
        pending_acks=["bob"],
    )
    hub._channels["s1"] = metadata
    hub._active_channels["s1"] = metadata

    clock.advance(45)
    await hub._expectation_tick()
    assert "s1" in hub._fired_violations

    await hub._transition_channel("s1", ChannelState.CLOSED, "test_reason")

    assert "s1" not in hub._fired_violations


@pytest.mark.asyncio
async def test_set_resume_rewrites_by_capability_disk_file() -> None:
    """Adding a claim via ``set_resume`` rewrites
    ``/registry/by_capability.json`` — the on-disk derived cache stays
    in sync with in-memory ``_capability_index``."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))

    initial = await store.read(by_capability_path())
    assert initial is not None
    assert json.loads(initial) == {}

    await alice.set_resume(Resume(claimed_capabilities=["math"]))
    assert json.loads(await store.read(by_capability_path())) == {"math": [alice.agent_id]}

    # Adding a second claim leaves the first intact.
    await alice.set_resume(Resume(claimed_capabilities=["math", "policy"]))
    after_add = json.loads(await store.read(by_capability_path()))
    assert after_add == {
        "math": [alice.agent_id],
        "policy": [alice.agent_id],
    }

    # Removing a claim drops it from disk.
    await alice.set_resume(Resume(claimed_capabilities=["policy"]))
    after_remove = json.loads(await store.read(by_capability_path()))
    assert after_remove == {"policy": [alice.agent_id]}

    await hub.close()


@pytest.mark.asyncio
async def test_delegate_fails_fast_on_channel_expire() -> None:
    """If the consulting channel expires (TTL fires) while delegate is
    awaiting a reply, delegate's terminal-event predicate releases the
    wait and returns ``Error: ... channel closed: ttl_expired`` rather
    than blocking the full 300s timeout. Exercises the
    ``EV_CHANNEL_EXPIRED`` branch of ``_TERMINAL_CHANNEL_EVENTS``
    (the ``EV_CHANNEL_CLOSED`` branch is covered by the close test
    above; ``EV_CHANNEL_INVITE_REJECT`` is unreachable from delegate's
    wait phase because ``open()`` raises before the wait begins)."""
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, clock=clock, ttl_sweep_interval=0, expectation_sweep_interval=0)

    from ag2.network.policies import AGENT_CLIENT_DEP

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"), attach_plugin=False)
    bob.on_envelope(_ack_only_handler(bob))

    delegate_tool = make_delegate_tool(alice)

    async def _expire_after_handshake() -> None:
        for _ in range(100):
            channels = await hub.list_channels(state=ChannelState.ACTIVE)
            if channels:
                # Default channel TTL is 2h; advance past it and trigger
                # the sweeper manually (interval=0 disables auto-tick).
                clock.advance(7200 + 60)
                await hub.expire_due()
                return
            await asyncio.sleep(0.01)

    expirer = asyncio.create_task(_expire_after_handshake())
    result = await asyncio.wait_for(
        _invoke(
            delegate_tool,
            {"target": "bob", "prompt": "hi", "timeout": 300.0},
            dependencies={AGENT_CLIENT_DEP: alice},
        ),
        timeout=10.0,
    )
    await expirer

    assert isinstance(result, str)
    assert result.startswith("Error:")
    # As with the close test, expiry can surface via the wait path
    # ("... channel closed: ttl_expired") or the send path
    # ("prompt send failed: channel '<id>' is expired"). Windows scheduling
    # tends to hit the send path; both are correct fail-fast outcomes.
    lowered = result.lower()
    assert "channel closed" in lowered or "prompt send failed" in lowered, result
    assert "expired" in lowered, result

    await hub.close()
