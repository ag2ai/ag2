# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Hub correctness invariants — registration, concurrency, dispatch, projection.

Covers:

* ``Hub.unregister`` deletes on-disk identity files so ``hydrate()``
  does not re-load unregistered agents.
* Concurrency caps + inbox limits enforced
  (``max_concurrent_sessions`` / ``max_concurrent_tasks`` /
  ``InboxBlock.max_pending``).
* ``delegate`` pre-creates the inbox before sending so a fast reply
  cannot be dropped (race fix).
* ``Hub.register`` rejects a duplicate ``name`` so the prior passport
  / resume / rule are not orphaned on disk.
* ``delegate`` fails fast on session close / reject / expiry rather
  than blocking until the 300s default timeout.
* Expectation dedup keyed by ``(index, name, violator)`` so two
  same-named expectations don't suppress each other.
* ``record_observation`` is idempotent per ``task_id`` so cascade-
  terminal events don't double-count ``Resume.observed.n``.
* ``set_resume`` re-indexes ``claimed_capabilities`` so newly claimed
  caps surface under ``peers(action="find", capability=...)``.
* ``EV_HANDOFF`` envelopes are projected through ``FullTranscript``
  and ``WindowedSummary`` so multi-hop workflows preserve the thread.
"""

import asyncio
import contextlib
import json
from typing import Any

import pytest

from autogen.beta import Agent, Context
from autogen.beta.events import ToolCallEvent
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    EV_HANDOFF,
    EV_SESSION_INVITE,
    EV_SESSION_INVITE_ACK,
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
from autogen.beta.network.adapters.conversation import (
    CONVERSATION_TYPE,
    ConversationAdapter,
)
from autogen.beta.network.client.tools.delegate import make_delegate_tool
from autogen.beta.network.hub.expectations import (
    AcksWithinEvaluator,
)
from autogen.beta.network.hub.layout import (
    by_capability_path,
    passport_path,
    resume_path,
    rule_path,
    skill_path,
)
from autogen.beta.network.rule import InboxBlock, LimitsBlock
from autogen.beta.network.session import (
    Expectation,
    Participant,
    ParticipantRole,
    ParticipantSchema,
    SessionManifest,
    SessionMetadata,
    SessionState,
)
from autogen.beta.network.views.builtin import FullTranscript, WindowedSummary
from autogen.beta.stream import MemoryStream
from autogen.beta.task import (
    TaskMetadata,
    TaskSpec,
    TaskState,
)
from autogen.beta.testing import TestConfig

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

    Used when a test needs a session to open (handshake completes) but
    wants the recipient to not respond to ``EV_TEXT`` so the hub's
    inbox-pending counter stays high.
    """

    async def _handle(env: Envelope) -> None:
        if env.event_type != EV_SESSION_INVITE:
            return
        ack = Envelope(
            session_id=env.session_id,
            sender_id=client.agent_id,
            audience=None,
            event_type=EV_SESSION_INVITE_ACK,
            event_data={"session_id": env.session_id},
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
        rule=Rule(limits=LimitsBlock(session_ttl_default="4h")),
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
    link = LocalLink(hub)

    hc = HubClient(link, hub=hub)
    await hc.register(_agent("alice"), Passport(name="alice"), Resume())

    with pytest.raises(ProtocolError, match="already registered"):
        await hc.register(_agent("alice"), Passport(name="alice"), Resume())

    await hc.close()
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
async def test_create_session_enforces_max_concurrent_sessions() -> None:
    """Hub rejects ``create_session`` when the creator has more active
    sessions than ``max_concurrent_sessions``. ``0`` disables."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    capped = Rule(limits=LimitsBlock(max_concurrent_sessions=1))
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    carol_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume(), rule=capped)
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())
    carol = await carol_hc.register(_agent("carol"), Passport(name="carol"), Resume())

    await alice.open(type=CONVERSATION_TYPE, target=bob.agent_id)

    with pytest.raises(AccessDeniedError, match="max_concurrent_sessions"):
        await alice.open(type=CONVERSATION_TYPE, target=carol.agent_id)

    await alice_hc.close()
    await bob_hc.close()
    await carol_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_observe_task_enforces_max_concurrent_tasks() -> None:
    """``observe_task`` rejects new tasks when the owner already has
    ``max_concurrent_tasks`` non-terminal tasks observed."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    capped = Rule(limits=LimitsBlock(max_concurrent_tasks=1))
    hc = HubClient(link, hub=hub)
    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume(), rule=capped)

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

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_post_envelope_enforces_inbox_max_pending() -> None:
    """``post_envelope`` raises ``InboxFull`` when the recipient's inbox
    is at capacity. The counter decrements when the recipient sends."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    bob_capped = Rule(limits=LimitsBlock(inbox=InboxBlock(max_pending=2)))
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(
        _agent("bob"),
        Passport(name="bob"),
        Resume(),
        rule=bob_capped,
        attach_plugin=False,
    )
    # Auto-ack invites so the session opens; ignore EV_TEXT so the
    # hub's pending-inbox counter for bob grows.
    bob.on_envelope(_ack_only_handler(bob))

    session = await alice.open(type=CONVERSATION_TYPE, target=bob.agent_id)

    def _make(text: str) -> Envelope:
        return Envelope(
            session_id=session.session_id,
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
            session_id=session.session_id,
            sender_id=bob.agent_id,
            audience=[alice.agent_id],
            event_type=EV_TEXT,
            event_data={"text": "ack"},
        )
    )
    await hub.post_envelope(_make("three"))  # accepted now

    await alice_hc.close()
    await bob_hc.close()
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

    session = await alice.open(type=CONVERSATION_TYPE, target=bob.agent_id)
    await hub.post_envelope(
        Envelope(
            session_id=session.session_id,
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
    immediately after ``session.send`` (the inbox is pre-created)."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    from autogen.beta.network.policies import AGENT_CLIENT_DEP

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await bob_hc.register(_agent("bob", "the answer is 42"), Passport(name="bob"), Resume())

    delegate_tool = make_delegate_tool(alice)
    result = await _invoke(
        delegate_tool,
        {"target": "bob", "prompt": "what's the answer?"},
        dependencies={AGENT_CLIENT_DEP: alice},
    )
    assert "42" in result

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_delegate_fails_fast_when_session_closes_before_reply() -> None:
    """If the consulting session closes / expires before the target replies,
    delegate returns immediately with an error — not after the 300s timeout."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    from autogen.beta.network.policies import AGENT_CLIENT_DEP

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume(), attach_plugin=False)
    # Bob acks the invite so the session opens, but never replies to
    # the prompt — alice's delegate would block until timeout without
    # the fail-fast path.
    bob.on_envelope(_ack_only_handler(bob))

    delegate_tool = make_delegate_tool(alice)

    async def _close_after_handshake() -> None:
        for _ in range(100):
            sessions = await hub.list_sessions(state=SessionState.ACTIVE)
            if sessions:
                await hub.close_session(sessions[0].session_id, reason="test_close")
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
    assert "session closed" in result.lower() or "test_close" in result.lower()

    await alice_hc.close()
    await bob_hc.close()
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

    # Build a session metadata directly — two same-name expectations
    # with different handlers in a single manifest.
    manifest = SessionManifest(
        type="dual_acks",
        version=1,
        participants=ParticipantSchema(min=2, max=2),
        expectations=[
            Expectation(name="acks_within", on_violation="audit", params={"seconds": 30}),
            Expectation(name="acks_within", on_violation="alt_audit", params={"seconds": 30}),
        ],
    )
    metadata = SessionMetadata(
        session_id="s1",
        manifest=manifest,
        creator_id="alice",
        participants=[
            Participant(agent_id="alice", role=ParticipantRole.INITIATOR, order=0),
            Participant(agent_id="bob", role=ParticipantRole.RESPONDENT, order=1),
        ],
        state=SessionState.PENDING,
        created_at="2026-01-01T00:00:00+00:00",
        pending_acks=["bob"],
    )
    hub._sessions["s1"] = metadata
    hub._active_sessions["s1"] = metadata

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
    link = LocalLink(hub)

    hc = HubClient(link, hub=hub)
    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())

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

    await hc.close()
    await hub.close()


# ── Fix #9: EV_HANDOFF visible to view projections ───────────────────────────


def _handoff(sender: str, *, tool: str, reason: str = "", session_id: str = "s1") -> Envelope:
    return Envelope(
        envelope_id=f"h-{sender}-{tool}",
        session_id=session_id,
        sender_id=sender,
        audience=None,
        event_type=EV_HANDOFF,
        event_data={"tool": tool, "reason": reason},
    )


def _text(sender: str, body: str, session_id: str = "s1") -> Envelope:
    return Envelope(
        envelope_id=f"t-{sender}-{body}",
        session_id=session_id,
        sender_id=sender,
        audience=None,
        event_type=EV_TEXT,
        event_data={"text": body},
    )


def _render(event: object) -> str:
    """Pull the visible text out of a ``ModelMessage`` or ``ModelRequest``."""
    content = getattr(event, "content", None)
    if isinstance(content, str):
        return content
    parts = getattr(event, "parts", None)
    if parts:
        first = parts[0]
        first_content = getattr(first, "content", None)
        if isinstance(first_content, str):
            return first_content
    return ""


def _two_party_metadata() -> SessionMetadata:
    return SessionMetadata(
        session_id="s1",
        manifest=ConversationAdapter().manifest,
        creator_id="alice",
        participants=[
            Participant(agent_id="alice", role=ParticipantRole.INITIATOR, order=0),
            Participant(agent_id="bob", role=ParticipantRole.RESPONDENT, order=1),
        ],
        state=SessionState.ACTIVE,
        created_at="2026-01-01T00:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_full_transcript_projects_handoff_envelopes() -> None:
    """``EV_HANDOFF`` envelopes render as ``[Handed off via X] reason``
    in the LLM's projected history so multi-hop workflows preserve the
    thread on subsequent turns."""
    metadata = _two_party_metadata()
    wal = [
        _text("alice", "hello bob"),
        _handoff("alice", tool="transfer_to_eng", reason="this needs engineering"),
        _text("bob", "got it"),
    ]
    events = await FullTranscript().project(wal, participant_id="bob", session=metadata)
    rendered = [_render(ev) for ev in events]
    assert "hello bob" in rendered
    handoff_line = next((r for r in rendered if "Handed off via" in r), None)
    assert handoff_line is not None
    assert "transfer_to_eng" in handoff_line
    assert "this needs engineering" in handoff_line


@pytest.mark.asyncio
async def test_windowed_summary_projects_handoff_envelopes() -> None:
    """``WindowedSummary`` keeps handoff envelopes in the recent tail
    instead of dropping them."""
    metadata = _two_party_metadata()
    wal = [
        _text("alice", "msg1"),
        _handoff("alice", tool="transfer_to_eng", reason="route this"),
        _text("bob", "msg2"),
    ]
    events = await WindowedSummary(recent_n=10).project(wal, participant_id="bob", session=metadata)
    rendered = [_render(ev) for ev in events]
    assert any("Handed off via transfer_to_eng" in r for r in rendered)


# ── Edge cases the per-fix tests above don't reach ───────────────────────────


@pytest.mark.asyncio
async def test_concurrency_caps_zero_disables() -> None:
    """``max_concurrent_sessions=0`` and ``max_concurrent_tasks=0`` both
    disable their respective caps — the documented "0 = unlimited"
    convention used elsewhere (delegation_depth, inbox.max_pending)."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    no_caps = Rule(limits=LimitsBlock(max_concurrent_sessions=0, max_concurrent_tasks=0))
    alice_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume(), rule=no_caps)

    # Five concurrent conversations open without raising.
    peer_handles: list[HubClient] = []
    for name in ("bob", "carol", "dave", "erin", "frank"):
        peer_hc = HubClient(link, hub=hub)
        peer = await peer_hc.register(_agent(name), Passport(name=name), Resume())
        peer_handles.append(peer_hc)
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

    for hc in peer_handles:
        await hc.close()
    await alice_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_inbox_capacity_does_not_block_protocol_events() -> None:
    """Protocol envelopes (invite / ack / opened / closed) bypass the
    inbox-pending counter so a busy agent can still receive new session
    opens. Without this carve-out a recipient at capacity would never
    learn about a new invite."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    bob_capped = Rule(limits=LimitsBlock(inbox=InboxBlock(max_pending=1)))
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    carol_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(
        _agent("bob"),
        Passport(name="bob"),
        Resume(),
        rule=bob_capped,
        attach_plugin=False,
    )
    bob.on_envelope(_ack_only_handler(bob))
    carol = await carol_hc.register(_agent("carol"), Passport(name="carol"), Resume())

    # Saturate bob's substantive-inbox budget via session 1.
    s1 = await alice.open(type=CONVERSATION_TYPE, target=bob.agent_id)
    await hub.post_envelope(
        Envelope(
            session_id=s1.session_id,
            sender_id=alice.agent_id,
            audience=[bob.agent_id],
            event_type=EV_TEXT,
            event_data={"text": "fill"},
        )
    )
    assert hub._inbox_pending.get(bob.agent_id, 0) >= 1

    # Carol opens a fresh session with bob — INVITE / ACK / OPENED must
    # not be rejected by the inbox check; bob's auto-ack runs and the
    # session reaches ACTIVE.
    s2 = await carol.open(type=CONVERSATION_TYPE, target=bob.agent_id)
    assert s2.state == SessionState.ACTIVE

    await alice_hc.close()
    await bob_hc.close()
    await carol_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_fired_violations_cleared_on_terminal_session_transition() -> None:
    """``_fired_violations[session_id]`` is dropped when the session
    transitions to a terminal state. Without this, a long-running hub
    accumulates dead dedup entries for every session that ever fired
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

    manifest = SessionManifest(
        type="testtype",
        version=1,
        participants=ParticipantSchema(min=2, max=2),
        expectations=[
            Expectation(name="acks_within", on_violation="audit", params={"seconds": 30}),
        ],
    )
    metadata = SessionMetadata(
        session_id="s1",
        manifest=manifest,
        creator_id="alice",
        participants=[
            Participant(agent_id="alice", role=ParticipantRole.INITIATOR, order=0),
            Participant(agent_id="bob", role=ParticipantRole.RESPONDENT, order=1),
        ],
        state=SessionState.PENDING,
        created_at="2026-01-01T00:00:00+00:00",
        pending_acks=["bob"],
    )
    hub._sessions["s1"] = metadata
    hub._active_sessions["s1"] = metadata

    clock.advance(45)
    await hub._expectation_tick()
    assert "s1" in hub._fired_violations

    await hub._transition_session("s1", SessionState.CLOSED, "test_reason")

    assert "s1" not in hub._fired_violations


@pytest.mark.asyncio
async def test_set_resume_rewrites_by_capability_disk_file() -> None:
    """Adding a claim via ``set_resume`` rewrites
    ``/registry/by_capability.json`` — the on-disk derived cache stays
    in sync with in-memory ``_capability_index``."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    hc = HubClient(link, hub=hub)
    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())

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

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_delegate_fails_fast_on_session_expire() -> None:
    """If the consulting session expires (TTL fires) while delegate is
    awaiting a reply, delegate's terminal-event predicate releases the
    wait and returns ``Error: ... session closed: ttl_expired`` rather
    than blocking the full 300s timeout. Exercises the
    ``EV_SESSION_EXPIRED`` branch of ``_TERMINAL_SESSION_EVENTS``
    (the ``EV_SESSION_CLOSED`` branch is covered by the close test
    above; ``EV_SESSION_INVITE_REJECT`` is unreachable from delegate's
    wait phase because ``open()`` raises before the wait begins)."""
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, clock=clock, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    from autogen.beta.network.policies import AGENT_CLIENT_DEP

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume(), attach_plugin=False)
    bob.on_envelope(_ack_only_handler(bob))

    delegate_tool = make_delegate_tool(alice)

    async def _expire_after_handshake() -> None:
        for _ in range(100):
            sessions = await hub.list_sessions(state=SessionState.ACTIVE)
            if sessions:
                # Default session TTL is 2h; advance past it and trigger
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
    assert "session closed" in result.lower()
    assert "ttl_expired" in result.lower()

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()
