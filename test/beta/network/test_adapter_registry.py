# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Phase 2 adapter registry.

Phase 2 makes session types an open set: operators call
``hub.register_adapter(adapter)`` to plug a custom
:class:`SessionAdapter` under an arbitrary string name. Registration is
idempotent-ish — the same adapter object can be re-registered under the
same name, but a *different* object replaces the prior one with a log
warning. Unknown types raise :class:`SessionTypeError` at session
creation time.

This file also exercises the ``dict[str, SessionAdapter]`` migration of
the hub's internal index: built-in enum members and plain strings must
be interchangeable as dispatch keys.
"""

from __future__ import annotations

import logging

import pytest

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    Envelope,
    Hub,
    SessionType,
    SessionTypeError,
)
from autogen.beta.network.adapters import (
    AdapterResult,
    ConsultingAdapter,
    SessionAdapter,
)
from autogen.beta.network.session_types import SessionMetadata, SessionState


# ---------------------------------------------------------------------------
# Custom adapters used by these tests
# ---------------------------------------------------------------------------


class _TournamentAdapter:
    """Example operator-registered adapter with a non-builtin name."""

    session_type = "tournament"

    def __init__(self) -> None:
        self.accepted: list[Envelope] = []

    def validate_create(self, metadata: SessionMetadata) -> None:
        if len(metadata.participants) < 2:
            raise SessionTypeError("tournament requires at least two participants")

    def validate_send(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        prior_envelopes: list[Envelope],
    ) -> None:
        if envelope.event_type != "ag2.msg.text":
            raise SessionTypeError("tournament only carries text")

    def on_accepted(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        prior_envelopes: list[Envelope],
    ) -> AdapterResult:
        self.accepted.append(envelope)
        if metadata.state is SessionState.PENDING:
            return AdapterResult(next_state=SessionState.ACTIVE)
        return AdapterResult()


class _StrictConsultingAdapter(ConsultingAdapter):
    """Operator replacement for the built-in consulting adapter."""

    pass


# ---------------------------------------------------------------------------
# register_adapter basic flows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_builtin_adapters_are_registered_by_enum_value_not_qualified_name(
    hub: Hub,
) -> None:
    """Internal dict keys must be the canonical string values.

    Prevents regression of a Python 3.11+ footgun: ``str(SessionType.X)``
    returns the qualified name ``"SessionType.X"`` rather than the
    underlying value ``"x"``. If the registry keyed off ``str(...)``,
    dispatch by plain-string name would silently miss.
    """

    assert hub.adapter_names() == [
        "auction",
        "broadcast",
        "consulting",
        "conversation",
        "discussion",
        "notification",
    ]
    assert hub._adapters["consulting"].__class__.__name__ == "ConsultingAdapter"
    assert hub._adapters["broadcast"].__class__.__name__ == "BroadcastAdapter"
    assert hub._adapters["discussion"].__class__.__name__ == "DiscussionAdapter"
    assert hub._adapters["auction"].__class__.__name__ == "AuctionAdapter"


def test_register_adapter_accepts_custom_string_type_names(hub: Hub) -> None:
    tournament = _TournamentAdapter()
    hub.register_adapter(tournament)
    assert "tournament" in hub.adapter_names()
    assert hub._adapters["tournament"] is tournament


def test_register_adapter_replaces_existing_and_warns(
    hub: Hub, caplog: pytest.LogCaptureFixture
) -> None:
    replacement = _StrictConsultingAdapter()
    with caplog.at_level(logging.WARNING, logger="autogen.beta.network.hub"):
        hub.register_adapter(replacement)
    assert hub._adapters["consulting"] is replacement
    assert any(
        "replacing session adapter for 'consulting'" in rec.message
        for rec in caplog.records
    )


def test_register_same_adapter_twice_is_a_no_op_no_warning(
    hub: Hub, caplog: pytest.LogCaptureFixture
) -> None:
    same = hub._adapters["consulting"]
    with caplog.at_level(logging.WARNING, logger="autogen.beta.network.hub"):
        hub.register_adapter(same)
    assert hub._adapters["consulting"] is same
    assert not any(
        "replacing session adapter" in rec.message for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# create_session dispatches by registered name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_session_raises_on_unknown_type(hub: Hub) -> None:
    alice = await hub.register(ActorIdentity(name="alice"))
    await hub.register(ActorIdentity(name="bob"))
    with pytest.raises(SessionTypeError, match="no adapter registered for 'tournament'"):
        await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type="tournament",
            participant_names=["bob"],
            invite_ack_timeout_s=0.2,
        )


@pytest.mark.asyncio
async def test_create_session_accepts_plain_string_for_builtins(hub: Hub) -> None:
    """Passing ``"consulting"`` must work identically to ``SessionType.CONSULTING``."""

    from test.beta.network._harness import attach_hub_to_link, auto_ack_only, FakeClient

    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"))
    link = attach_hub_to_link(hub)
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only)
    await b.start()
    try:
        meta = await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type="consulting",  # <-- plain string
            participant_names=["bob"],
            invite_ack_timeout_s=0.5,
        )
        assert meta.state is SessionState.ACTIVE
        assert meta.type == "consulting"
        assert meta.type == SessionType.CONSULTING  # still compares equal
    finally:
        await b.stop()
        await link.close()


@pytest.mark.asyncio
async def test_create_session_accepts_enum_member_as_before(hub: Hub) -> None:
    """Backwards compatibility: ``SessionType.CONSULTING`` still dispatches."""

    from test.beta.network._harness import attach_hub_to_link, auto_ack_only, FakeClient

    alice = await hub.register(ActorIdentity(name="alice"))
    bob = await hub.register(ActorIdentity(name="bob"))
    link = attach_hub_to_link(hub)
    b = FakeClient(hub=hub, link=link, actor_id=bob.actor_id or "", handler=auto_ack_only)
    await b.start()
    try:
        meta = await hub.create_session(
            creator_id=alice.actor_id or "",
            session_type=SessionType.CONSULTING,
            participant_names=["bob"],
            invite_ack_timeout_s=0.5,
        )
        assert meta.state is SessionState.ACTIVE
    finally:
        await b.stop()
        await link.close()


# ---------------------------------------------------------------------------
# SessionAdapter protocol compliance
# ---------------------------------------------------------------------------


def test_tournament_adapter_satisfies_protocol() -> None:
    assert isinstance(_TournamentAdapter(), SessionAdapter)


def test_consulting_adapter_session_type_is_plain_string_or_enum_str() -> None:
    # Built-in adapters can still use the enum as their declared type —
    # the hub normalizes to ``.value`` at registration time.
    from autogen.beta.network.hub.core import _type_name

    assert _type_name(ConsultingAdapter().session_type) == "consulting"
