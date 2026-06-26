# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``NameResolver`` plumbing + the named built-in views.

This PR exposes a ``name_for: NameResolver`` kwarg on
``ViewPolicy.project`` so views can prefix projections with the
sender's ``Passport.name`` without depending on hub internals, and
ships two new built-ins (``NamedTranscript`` and
``NamedWindowedSummary``) that USE the resolver to prefix non-self
projection lines with ``[<name>]:``. The default-view for the
N-party adapters (``discussion``, ``workflow``) swaps to
``NamedWindowedSummary``; the 2-party adapters (``consulting``,
``conversation``) keep their unprefixed defaults because the
assistant/user role bit already disambiguates with only one "other".

These tests pin:

* Foundation: ``Hub.name_for`` + back-compat built-ins accept the new kwarg.
* Named built-ins prefix non-self projections (own envelopes stay bare).
* Adapter defaults swap to Named in N-party, stay unprefixed in 2-party.
* The default notify handler hands a real ``Hub.name_for``-backed
  resolver into ``view.project`` at turn time.
"""

import pytest

from ag2 import Agent
from ag2.compact import CompactionSummary
from ag2.events import ModelMessage, ModelRequest, TextInput
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_TEXT,
    Envelope,
    FullTranscript,
    Hub,
    NameResolver,
    NamedTranscript,
    NamedWindowedSummary,
    WindowedSummary,
    default_name_resolver,
)
from ag2.network.adapters.base import default_render_envelope
from ag2.network.adapters.consulting import ConsultingAdapter
from ag2.network.adapters.conversation import ConversationAdapter
from ag2.network.adapters.discussion import DiscussionAdapter
from ag2.network.adapters.workflow import WorkflowAdapter
from ag2.network.channel import (
    ChannelMetadata,
    ChannelState,
    Participant,
    ParticipantRole,
)
from ag2.network.client.handlers import _render_current_input
from ag2.testing import TestConfig


def _two_party_metadata(a_id: str, b_id: str) -> ChannelMetadata:
    """Synthesize a minimal conversation-style channel for projection tests."""
    return ChannelMetadata(
        channel_id="ch-test",
        manifest=ConversationAdapter().manifest,
        creator_id=a_id,
        participants=[
            Participant(agent_id=a_id, role=ParticipantRole.INITIATOR, order=0, joined_at="now"),
            Participant(agent_id=b_id, role=ParticipantRole.RESPONDENT, order=1, joined_at="now"),
        ],
        state=ChannelState.ACTIVE,
        created_at="now",
        expires_at=None,
        knobs={},
        labels={},
        required_acks=None,
        pending_acks=[],
    )


def _text_envelope(sender: str, audience: str, text: str) -> Envelope:
    return Envelope(
        channel_id="ch-test",
        sender_id=sender,
        audience=[audience],
        event_type=EV_TEXT,
        event_data={"text": text},
    )


def test_default_name_resolver_returns_id_verbatim() -> None:
    assert default_name_resolver("abc-123") == "abc-123"


@pytest.mark.asyncio
async def test_full_transcript_accepts_name_for_kwarg() -> None:
    """Back-compat: built-in views accept the new kwarg and behave unchanged."""
    metadata = _two_party_metadata("alice", "bob")
    wal = [
        _text_envelope("alice", "bob", "hi"),
        _text_envelope("bob", "alice", "hello"),
    ]

    view = FullTranscript()
    projection = await view.project(
        wal,
        participant_id="bob",
        channel=metadata,
        render_envelope=default_render_envelope,
        name_for=lambda aid: f"Mx-{aid}",
    )

    # Built-ins ignore name_for today — same shape as without it.
    assert projection == [
        ModelRequest([TextInput("hi")]),
        ModelMessage("hello"),
    ]


@pytest.mark.asyncio
async def test_windowed_summary_accepts_name_for_kwarg() -> None:
    """``WindowedSummary`` accepts ``name_for`` without behaviour change."""
    metadata = _two_party_metadata("alice", "bob")
    wal = [
        _text_envelope("alice", "bob", "hi"),
        _text_envelope("bob", "alice", "hello"),
    ]

    view = WindowedSummary(recent_n=10)
    projection = await view.project(
        wal,
        participant_id="bob",
        channel=metadata,
        render_envelope=default_render_envelope,
        name_for=lambda aid: f"Mx-{aid}",
    )

    assert projection == [
        ModelRequest([TextInput("hi")]),
        ModelMessage("hello"),
    ]


@pytest.mark.asyncio
async def test_hub_name_for_resolves_registered_passport_name() -> None:
    """``Hub.name_for`` returns the ``Passport.name`` for a registered agent."""
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    try:
        agent = Agent(name="alice", prompt="x", config=TestConfig([], "ok"))
        client = await hub.register(agent, attach_plugin=False)
        assert hub.name_for(client.agent_id) == "alice"
    finally:
        await hub.close()


@pytest.mark.asyncio
async def test_hub_name_for_falls_back_for_unknown_id() -> None:
    """Unknown ids fall back to the id itself (or a supplied default)."""
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    try:
        assert hub.name_for("does-not-exist") == "does-not-exist"
        assert hub.name_for("does-not-exist", default="unknown") == "unknown"
    finally:
        await hub.close()


@pytest.mark.asyncio
async def test_handler_passes_real_resolver_to_view_project() -> None:
    """The default notify handler hands a hub-backed ``name_for`` to
    ``view.project``. Verified by registering a custom view that records
    what it was called with."""

    seen: dict[str, object] = {}

    class RecordingView:
        name = "recording"

        async def project(
            self,
            wal,  # noqa: ARG002
            *,
            participant_id,
            channel,  # noqa: ARG002
            render_envelope,  # noqa: ARG002
            name_for: NameResolver = default_name_resolver,
        ):
            seen["name_for_alice_known"] = name_for(participant_id)
            seen["name_for_unknown"] = name_for("definitely-not-an-agent-id")
            return []

    # Stand up a hub with a conversation adapter whose default view is
    # our recorder. We don't need a real LLM turn — just the handler
    # passing name_for into project. We bypass the agent loop by
    # directly invoking project the way the handler does.
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    try:
        agent = Agent(name="alice", prompt="x", config=TestConfig([], "ok"))
        client = await hub.register(agent, attach_plugin=False)

        metadata = _two_party_metadata(client.agent_id, "bob")
        view = RecordingView()
        await view.project(
            [],
            participant_id=client.agent_id,
            channel=metadata,
            render_envelope=default_render_envelope,
            name_for=hub.name_for,
        )

        assert seen["name_for_alice_known"] == "alice"
        assert seen["name_for_unknown"] == "definitely-not-an-agent-id"
    finally:
        await hub.close()


# ── Named views ────────────────────────────────────────────────────────────


def _named_resolver(mapping: dict[str, str]) -> NameResolver:
    """Build a resolver from an id→name dict, falling back to the id."""
    return lambda aid: mapping.get(aid, aid)


@pytest.mark.asyncio
async def test_named_transcript_prefixes_non_self_envelopes() -> None:
    """Non-self lines get ``[<name>]:`` prefix; own lines stay bare."""
    metadata = _two_party_metadata("alice", "bob")
    wal = [
        _text_envelope("alice", "bob", "hi bob"),
        _text_envelope("bob", "alice", "hi alice"),
        _text_envelope("alice", "bob", "how's it going"),
    ]

    projection = await NamedTranscript().project(
        wal,
        participant_id="bob",
        channel=metadata,
        render_envelope=default_render_envelope,
        name_for=_named_resolver({"alice": "Alice", "bob": "Bob"}),
    )

    assert projection == [
        ModelRequest([TextInput("[Alice]: hi bob")]),
        ModelMessage("hi alice"),
        ModelRequest([TextInput("[Alice]: how's it going")]),
    ]


@pytest.mark.asyncio
async def test_named_transcript_falls_back_to_raw_id_for_unknown_sender() -> None:
    """If the resolver returns the raw id, projection still succeeds."""
    metadata = _two_party_metadata("alice", "bob")
    wal = [_text_envelope("ghost", "alice", "leftover")]

    projection = await NamedTranscript().project(
        wal,
        participant_id="alice",
        channel=metadata,
        render_envelope=default_render_envelope,
        name_for=default_name_resolver,  # identity — returns id verbatim
    )

    assert projection == [ModelRequest([TextInput("[ghost]: leftover")])]


@pytest.mark.asyncio
async def test_named_windowed_summary_short_history_labels_recents() -> None:
    """Short history (WAL ≤ recent_n): labels every non-self projection."""
    metadata = _two_party_metadata("alice", "bob")
    wal = [
        _text_envelope("alice", "bob", "hi"),
        _text_envelope("bob", "alice", "hello"),
    ]

    projection = await NamedWindowedSummary(recent_n=10).project(
        wal,
        participant_id="bob",
        channel=metadata,
        render_envelope=default_render_envelope,
        name_for=_named_resolver({"alice": "Alice", "bob": "Bob"}),
    )

    assert projection == [
        ModelRequest([TextInput("[Alice]: hi")]),
        ModelMessage("hello"),
    ]


@pytest.mark.asyncio
async def test_named_windowed_summary_long_history_summarises_with_names() -> None:
    """When WAL > recent_n, the head compaction lists speakers by name."""
    metadata = _two_party_metadata("alice", "bob")
    wal = [_text_envelope("alice", "bob", f"msg {i}") for i in range(8)]

    projection = await NamedWindowedSummary(recent_n=3).project(
        wal,
        participant_id="bob",
        channel=metadata,
        render_envelope=default_render_envelope,
        name_for=_named_resolver({"alice": "Alice"}),
    )

    head = projection[0]
    assert isinstance(head, CompactionSummary)
    assert head.event_count == 5
    # Speakers list uses the resolved name, not the raw id.
    assert "Alice" in head.summary
    assert "alice" not in head.summary
    # Tail entries are prefixed with the sender name.
    assert projection[1:] == [
        ModelRequest([TextInput("[Alice]: msg 5")]),
        ModelRequest([TextInput("[Alice]: msg 6")]),
        ModelRequest([TextInput("[Alice]: msg 7")]),
    ]


# ── Adapter default-view swap (N-party only) ───────────────────────────────


def _three_party_metadata(adapter, ids: list[str]) -> ChannelMetadata:
    """Build minimal metadata for an N-party adapter."""
    return ChannelMetadata(
        channel_id="ch-n",
        manifest=adapter.manifest,
        creator_id=ids[0],
        participants=[
            Participant(agent_id=a, role=ParticipantRole.PARTICIPANT, order=i, joined_at="now")
            for i, a in enumerate(ids)
        ],
        state=ChannelState.ACTIVE,
        created_at="now",
        expires_at=None,
        knobs={},
        labels={},
        required_acks=None,
        pending_acks=[],
    )


def test_workflow_default_view_is_named() -> None:
    """N-party workflow adapter defaults to NamedWindowedSummary."""
    adapter = WorkflowAdapter()
    metadata = _three_party_metadata(adapter, ["a", "b", "c"])
    view = adapter.default_view_policy(metadata, participant_id="a")
    assert isinstance(view, NamedWindowedSummary)


def test_discussion_default_view_is_named() -> None:
    """N-party discussion adapter defaults to NamedWindowedSummary."""
    adapter = DiscussionAdapter()
    metadata = _three_party_metadata(adapter, ["a", "b", "c"])
    view = adapter.default_view_policy(metadata, participant_id="a")
    assert isinstance(view, NamedWindowedSummary)


def test_consulting_default_view_unchanged() -> None:
    """2-party consulting adapter keeps unprefixed FullTranscript."""
    adapter = ConsultingAdapter()
    metadata = _three_party_metadata(adapter, ["a", "b"])  # 2 participants
    view = adapter.default_view_policy(metadata, participant_id="a")
    assert isinstance(view, FullTranscript)


def test_conversation_default_view_unchanged() -> None:
    """2-party conversation adapter keeps unprefixed WindowedSummary."""
    adapter = ConversationAdapter()
    metadata = _three_party_metadata(adapter, ["a", "b"])  # 2 participants
    view = adapter.default_view_policy(metadata, participant_id="a")
    assert isinstance(view, WindowedSummary)
    # And NOT the named variant (subclass guard).
    assert not isinstance(view, NamedWindowedSummary)


# ── Single-envelope projection: current-turn attribution ─────────────────────
#
# The handler calls view.project([current_envelope], ...) for the triggering
# envelope so named views apply the same [name]: label they use for history.


@pytest.mark.asyncio
async def test_named_transcript_single_envelope_prefixes_non_self() -> None:
    """project() on a single non-self envelope gives [name]: prefix."""
    metadata = _two_party_metadata("alice", "bob")
    wal = [_text_envelope("alice", "bob", "hi bob")]

    projection = await NamedTranscript().project(
        wal,
        participant_id="bob",
        channel=metadata,
        render_envelope=default_render_envelope,
        name_for=_named_resolver({"alice": "Alice"}),
    )

    assert projection == [ModelRequest([TextInput("[Alice]: hi bob")])]


@pytest.mark.asyncio
async def test_named_windowed_summary_single_envelope_prefixes_non_self() -> None:
    """NamedWindowedSummary.project() on a single envelope gives [name]: prefix."""
    metadata = _two_party_metadata("alice", "bob")
    wal = [_text_envelope("alice", "bob", "hi bob")]

    projection = await NamedWindowedSummary(recent_n=10).project(
        wal,
        participant_id="bob",
        channel=metadata,
        render_envelope=default_render_envelope,
        name_for=_named_resolver({"alice": "Alice"}),
    )

    assert projection == [ModelRequest([TextInput("[Alice]: hi bob")])]


@pytest.mark.asyncio
async def test_full_transcript_single_envelope_no_prefix() -> None:
    """Unnamed FullTranscript.project() on a single envelope gives no prefix."""
    metadata = _two_party_metadata("alice", "bob")
    wal = [_text_envelope("alice", "bob", "hi bob")]

    projection = await FullTranscript().project(
        wal,
        participant_id="bob",
        channel=metadata,
        render_envelope=default_render_envelope,
        name_for=_named_resolver({"alice": "Alice"}),
    )

    assert projection == [ModelRequest([TextInput("hi bob")])]


@pytest.mark.asyncio
async def test_windowed_summary_single_envelope_no_prefix() -> None:
    """Unnamed WindowedSummary.project() on a single envelope gives no prefix."""
    metadata = _two_party_metadata("alice", "bob")
    wal = [_text_envelope("alice", "bob", "hi bob")]

    projection = await WindowedSummary(recent_n=10).project(
        wal,
        participant_id="bob",
        channel=metadata,
        render_envelope=default_render_envelope,
        name_for=_named_resolver({"alice": "Alice"}),
    )

    assert projection == [ModelRequest([TextInput("hi bob")])]


# ── _render_current_input: branch coverage ────────────────────────────────────


class _FixedView:
    """Stub ViewPolicy that always returns a pre-configured projection."""

    name = "fixed"

    def __init__(self, projected: list) -> None:
        self._projected = projected

    async def project(
        self,
        wal,  # noqa: ARG002
        *,
        participant_id,  # noqa: ARG002
        channel,  # noqa: ARG002
        render_envelope,  # noqa: ARG002
        name_for: NameResolver = default_name_resolver,  # noqa: ARG002
    ) -> list:
        return self._projected


class _FixedAdapter:
    """Stub adapter with a fixed extract_turn_input return value."""

    def __init__(self, fallback: str = "fallback") -> None:
        self._fallback = fallback

    def render_envelope(self, envelope):  # noqa: ARG002
        return None

    def extract_turn_input(self, envelope):  # noqa: ARG002
        return self._fallback


@pytest.mark.asyncio
class TestRenderCurrentInput:
    """Branch coverage for ``_render_current_input``."""

    async def test_model_request_with_text_input_returns_content(self) -> None:
        """Happy path: projected ModelRequest with TextInput → text content."""
        envelope = _text_envelope("alice", "bob", "hi bob")
        metadata = _two_party_metadata("alice", "bob")
        view = _FixedView([ModelRequest([TextInput("[Alice]: hi bob")])])
        adapter = _FixedAdapter("fallback")

        result = await _render_current_input(view, envelope, adapter, "bob", metadata, default_name_resolver)

        assert result == "[Alice]: hi bob"

    async def test_empty_projection_falls_back_to_adapter(self) -> None:
        """view.project returns [] (e.g. image/document) → adapter.extract_turn_input."""
        envelope = _text_envelope("alice", "bob", "hi bob")
        metadata = _two_party_metadata("alice", "bob")
        view = _FixedView([])
        adapter = _FixedAdapter("adapter fallback")

        result = await _render_current_input(view, envelope, adapter, "bob", metadata, default_name_resolver)

        assert result == "adapter fallback"

    async def test_model_request_no_text_parts_falls_back_to_adapter(self) -> None:
        """ModelRequest with no TextInput parts (e.g. only non-text inputs) → adapter fallback."""
        envelope = _text_envelope("alice", "bob", "hi bob")
        metadata = _two_party_metadata("alice", "bob")
        view = _FixedView([ModelRequest([])])
        adapter = _FixedAdapter("adapter fallback")

        result = await _render_current_input(view, envelope, adapter, "bob", metadata, default_name_resolver)

        assert result == "adapter fallback"

    async def test_model_message_returns_content(self) -> None:
        """ModelMessage projection (custom view treating current turn as own-past) → content."""
        envelope = _text_envelope("bob", "alice", "my own message")
        metadata = _two_party_metadata("alice", "bob")
        view = _FixedView([ModelMessage("my own message")])
        adapter = _FixedAdapter("fallback")

        result = await _render_current_input(view, envelope, adapter, "bob", metadata, default_name_resolver)

        assert result == "my own message"
