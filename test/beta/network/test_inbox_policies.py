# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`SessionInboxPolicy` and :class:`PreviousOnlyInboxPolicy`.

These are the assembly-policy bridges between the session WAL (the
network's source of truth for what happened between actors) and the
framework-core actor's LLM view. The Phase 1 retrospective explicitly
deferred :class:`SessionInboxPolicy` as 🟡 — this file is the Phase 2
delivery.

Scope:

* Translation from envelopes to ``ModelRequest`` / ``ModelMessage``
  (direction-aware).
* Session discovery via ``Context.variables`` and
  ``Context.dependencies`` — the ``ActorClient``'s wiring contract.
* Standalone-actor fallback: no session id / no hub in the context ⇒
  the policy is a pure pass-through.
* Pipeline replacement: ``PreviousOnlyInboxPolicy`` only shows the last
  cross-actor text envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from autogen.beta.events import BaseEvent, ModelMessage, ModelRequest, TextInput
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    Envelope,
    Hub,
    PreviousOnlyInboxPolicy,
    SessionInboxPolicy,
    SessionType,
)
from autogen.beta.network.envelope import EV_TEXT
from autogen.beta.network.policies import HUB_DEP, SESSION_ID_VAR


def _text(e) -> str | None:
    """Extract first TextInput content from a ModelRequest, or event's .content."""
    from autogen.beta.events import ModelRequest, TextInput
    if isinstance(e, ModelRequest):
        for inp in e.inputs:
            if isinstance(inp, TextInput):
                return inp.content
        return None
    return getattr(e, "content", None)




# ---------------------------------------------------------------------------
# Minimal fake Context
# ---------------------------------------------------------------------------


@dataclass
class _FakeContext:
    variables: dict[str, Any] = field(default_factory=dict)
    dependencies: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hub() -> Hub:
    return Hub(MemoryKnowledgeStore())


async def _seed_session(hub: Hub, *, session_type: SessionType = SessionType.CONVERSATION):
    """Register alice+bob, open a session, and return (alice_id, bob_id, session_meta)."""

    from test.beta.network._harness import (
        FakeClient,
        attach_hub_to_link,
        auto_ack_only,
    )

    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"))
    link = attach_hub_to_link(hub)
    a = FakeClient(hub=hub, link=link, actor_id=alice.actor_id or "")
    b = FakeClient(
        hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only
    )
    await a.start()
    await b.start()
    meta = await hub.create_session(
        creator_id=alice.actor_id or "",
        session_type=session_type,
        participant_names=["bob"],
        invite_ack_timeout_s=0.5,
    )
    return (
        alice.actor_id or "",
        bob.actor_id or "",
        meta,
        a,
        b,
        link,
    )


# ---------------------------------------------------------------------------
# Translation rules — direction-aware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_inbox_policy_prepends_full_wal_to_events(hub: Hub) -> None:
    alice_id, bob_id, meta, a, b, link = await _seed_session(hub)
    try:
        # Plant three prior text envelopes in the WAL via the hub's
        # own API so each append is observable before the next runs
        # (FakeClient.send_text is fire-and-forget over a frame queue).
        await hub.post_envelope(
            Envelope.text(
                session_id=meta.session_id,
                sender_id=alice_id,
                content="1",
                recipient_id=bob_id,
            )
        )
        await hub.post_envelope(
            Envelope.text(
                session_id=meta.session_id,
                sender_id=alice_id,
                content="2",
                recipient_id=bob_id,
            )
        )
        await hub.post_envelope(
            Envelope.text(
                session_id=meta.session_id,
                sender_id=bob_id,
                content="B says hi",
                recipient_id=alice_id,
            )
        )
        policy = SessionInboxPolicy(self_actor_id=alice_id)
        ctx = _FakeContext(
            variables={SESSION_ID_VAR: meta.session_id},
            dependencies={HUB_DEP: hub},
        )
        new_prompts, new_events = await policy.apply(
            prompts=["system"],
            events=[ModelRequest([TextInput("current turn")])],
            context=ctx,
        )
        assert new_prompts == ["system"]
        # WAL text envelopes become ModelRequest (from Bob) or
        # ModelMessage (from Alice, since policy.self_actor_id=alice).
        contents = [
            (type(e).__name__, _text(e)) for e in new_events
        ]
        assert contents == [
            ("ModelMessage", "1"),
            ("ModelMessage", "2"),
            ("ModelRequest", "B says hi"),
            ("ModelRequest", "current turn"),
        ]
    finally:
        await a.stop()
        await b.stop()
        await link.close()


@pytest.mark.asyncio
async def test_session_inbox_policy_skips_own_messages_when_not_included(
    hub: Hub,
) -> None:
    alice_id, bob_id, meta, a, b, link = await _seed_session(hub)
    try:
        await hub.post_envelope(
            Envelope.text(
                session_id=meta.session_id,
                sender_id=alice_id,
                content="mine",
                recipient_id=bob_id,
            )
        )
        await hub.post_envelope(
            Envelope.text(
                session_id=meta.session_id,
                sender_id=bob_id,
                content="theirs",
                recipient_id=alice_id,
            )
        )
        policy = SessionInboxPolicy(self_actor_id=alice_id, include_own=False)
        ctx = _FakeContext(
            variables={SESSION_ID_VAR: meta.session_id},
            dependencies={HUB_DEP: hub},
        )
        _, new_events = await policy.apply(
            prompts=[],
            events=[ModelRequest([TextInput("now")])],
            context=ctx,
        )
        # Alice's own envelope is filtered out; Bob's stays; current turn remains last.
        assert [type(e).__name__ for e in new_events] == [
            "ModelRequest",
            "ModelRequest",
        ]
        assert [_text(e) for e in new_events] == ["theirs", "now"]
    finally:
        await a.stop()
        await b.stop()
        await link.close()


@pytest.mark.asyncio
async def test_session_inbox_policy_noop_without_session_id(hub: Hub) -> None:
    policy = SessionInboxPolicy(self_actor_id="alice")
    ctx = _FakeContext()
    prompts, events = await policy.apply(
        prompts=["p"], events=[ModelRequest([TextInput("q")])], context=ctx
    )
    assert prompts == ["p"]
    assert len(events) == 1 and _text(events[0]) == "q"


@pytest.mark.asyncio
async def test_session_inbox_policy_noop_without_hub(hub: Hub) -> None:
    policy = SessionInboxPolicy(self_actor_id="alice")
    ctx = _FakeContext(variables={SESSION_ID_VAR: "01-missing"})
    _, events = await policy.apply(
        prompts=[], events=[ModelRequest([TextInput("q")])], context=ctx
    )
    assert [_text(e) for e in events] == ["q"]


@pytest.mark.asyncio
async def test_session_inbox_policy_skips_system_envelopes(hub: Hub) -> None:
    alice_id, bob_id, meta, a, b, link = await _seed_session(hub)
    try:
        await hub.post_envelope(
            Envelope.text(
                session_id=meta.session_id,
                sender_id=alice_id,
                content="real",
                recipient_id=bob_id,
            )
        )
        policy = SessionInboxPolicy(self_actor_id=alice_id)
        ctx = _FakeContext(
            variables={SESSION_ID_VAR: meta.session_id},
            dependencies={HUB_DEP: hub},
        )
        _, events = await policy.apply(
            prompts=[], events=[ModelRequest([TextInput("current")])], context=ctx
        )
        # Invite, invite_ack, session_opened should NOT produce events.
        # Only the text envelope + current turn remain.
        assert [type(e).__name__ for e in events] == [
            "ModelMessage",
            "ModelRequest",
        ]
        assert [_text(e) for e in events] == ["real", "current"]
    finally:
        await a.stop()
        await b.stop()
        await link.close()


# ---------------------------------------------------------------------------
# PreviousOnlyInboxPolicy — pipeline semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_previous_only_injects_last_foreign_envelope(hub: Hub) -> None:
    alice_id, bob_id, meta, a, b, link = await _seed_session(
        hub, session_type=SessionType.DISCUSSION
    )
    try:
        # Alice → Bob (foreign from Bob's perspective)
        await hub.post_envelope(
            Envelope.text(
                session_id=meta.session_id, sender_id=alice_id, content="A1"
            )
        )
        # Bob → Alice
        await hub.post_envelope(
            Envelope.text(session_id=meta.session_id, sender_id=bob_id, content="B1")
        )
        # Alice → Bob second (newer from Bob's perspective)
        await hub.post_envelope(
            Envelope.text(
                session_id=meta.session_id, sender_id=alice_id, content="A2"
            )
        )

        policy = PreviousOnlyInboxPolicy(self_actor_id=bob_id)
        ctx = _FakeContext(
            variables={SESSION_ID_VAR: meta.session_id},
            dependencies={HUB_DEP: hub},
        )
        _, events = await policy.apply(
            prompts=[], events=[ModelRequest([TextInput("current")])], context=ctx
        )
        # Only the most recent cross-actor envelope lands — A2.
        assert [type(e).__name__ for e in events] == ["ModelRequest", "ModelRequest"]
        assert [_text(e) for e in events] == ["A2", "current"]
    finally:
        await a.stop()
        await b.stop()
        await link.close()


@pytest.mark.asyncio
async def test_previous_only_noop_when_no_prior_foreign_envelope(hub: Hub) -> None:
    alice_id, bob_id, meta, a, b, link = await _seed_session(hub)
    try:
        policy = PreviousOnlyInboxPolicy(self_actor_id=alice_id)
        ctx = _FakeContext(
            variables={SESSION_ID_VAR: meta.session_id},
            dependencies={HUB_DEP: hub},
        )
        _, events = await policy.apply(
            prompts=[], events=[ModelRequest([TextInput("current")])], context=ctx
        )
        # No prior cross-actor text envelope yet — policy is pass-through.
        assert len(events) == 1
        assert _text(events[0]) == "current"
    finally:
        await a.stop()
        await b.stop()
        await link.close()


@pytest.mark.asyncio
async def test_previous_only_skips_own_envelopes(hub: Hub) -> None:
    alice_id, bob_id, meta, a, b, link = await _seed_session(hub)
    try:
        # Three envelopes all from Alice — none are "foreign" to Alice.
        await hub.post_envelope(
            Envelope.text(
                session_id=meta.session_id,
                sender_id=alice_id,
                content="1",
                recipient_id=bob_id,
            )
        )
        await hub.post_envelope(
            Envelope.text(
                session_id=meta.session_id,
                sender_id=alice_id,
                content="2",
                recipient_id=bob_id,
            )
        )

        policy = PreviousOnlyInboxPolicy(self_actor_id=alice_id)
        ctx = _FakeContext(
            variables={SESSION_ID_VAR: meta.session_id},
            dependencies={HUB_DEP: hub},
        )
        _, events = await policy.apply(
            prompts=[], events=[ModelRequest([TextInput("now")])], context=ctx
        )
        assert len(events) == 1
        assert _text(events[0]) == "now"
    finally:
        await a.stop()
        await b.stop()
        await link.close()
