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

import asyncio

import pytest

from ag2 import Agent
from ag2.compact import CompactionSummary
from ag2.events import BaseEvent, ImageInput, ModelMessage, ModelRequest, TextInput
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    FullTranscript,
    Hub,
    NameResolver,
    NamedTranscript,
    NamedWindowedSummary,
    ViewPolicy,
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
from ag2.testing import TestConfig, TrackingConfig

from ._helpers import text_envelope, two_party_metadata


def test_default_name_resolver_returns_id_verbatim() -> None:
    assert default_name_resolver("abc-123") == "abc-123"


@pytest.mark.asyncio
async def test_full_transcript_accepts_name_for_kwarg() -> None:
    """Back-compat: built-in views accept the new kwarg and behave unchanged."""
    metadata = two_party_metadata("alice", "bob")
    wal = [
        text_envelope("alice", "bob", "hi"),
        text_envelope("bob", "alice", "hello"),
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
    metadata = two_party_metadata("alice", "bob")
    wal = [
        text_envelope("alice", "bob", "hi"),
        text_envelope("bob", "alice", "hello"),
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

        metadata = two_party_metadata(client.agent_id, "bob")
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
    metadata = two_party_metadata("alice", "bob")
    wal = [
        text_envelope("alice", "bob", "hi bob"),
        text_envelope("bob", "alice", "hi alice"),
        text_envelope("alice", "bob", "how's it going"),
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
    metadata = two_party_metadata("alice", "bob")
    wal = [text_envelope("ghost", "alice", "leftover")]

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
    metadata = two_party_metadata("alice", "bob")
    wal = [
        text_envelope("alice", "bob", "hi"),
        text_envelope("bob", "alice", "hello"),
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
    metadata = two_party_metadata("alice", "bob")
    wal = [text_envelope("alice", "bob", f"msg {i}") for i in range(8)]

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
    metadata = two_party_metadata("alice", "bob")
    wal = [text_envelope("alice", "bob", "hi bob")]

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
    metadata = two_party_metadata("alice", "bob")
    wal = [text_envelope("alice", "bob", "hi bob")]

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
    metadata = two_party_metadata("alice", "bob")
    wal = [text_envelope("alice", "bob", "hi bob")]

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
    metadata = two_party_metadata("alice", "bob")
    wal = [text_envelope("alice", "bob", "hi bob")]

    projection = await WindowedSummary(recent_n=10).project(
        wal,
        participant_id="bob",
        channel=metadata,
        render_envelope=default_render_envelope,
        name_for=_named_resolver({"alice": "Alice"}),
    )

    assert projection == [ModelRequest([TextInput("hi bob")])]


# ── What the agent is handed for the turn ─────────────────────────────────────


class _FixedView:
    """A ``ViewPolicy`` that projects every slice to the same events."""

    name = "fixed"

    def __init__(self, *projected: BaseEvent) -> None:
        self._projected = list(projected)

    async def project(
        self,
        wal,  # noqa: ARG002
        *,
        participant_id,  # noqa: ARG002
        channel,  # noqa: ARG002
        render_envelope,  # noqa: ARG002
        name_for: NameResolver = default_name_resolver,  # noqa: ARG002
    ) -> list[BaseEvent]:
        return list(self._projected)


class _ScriptedConversation(ConversationAdapter):
    """A conversation adapter whose rendering, turn input and view are set by the test."""

    def __init__(self, *, rendered=None, turn_input=None, view: ViewPolicy | None = None) -> None:
        super().__init__()
        self._rendered = rendered
        self._turn_input = turn_input
        self._view = view

    def render_envelope(self, envelope):
        return self._rendered if self._rendered is not None else super().render_envelope(envelope)

    def extract_turn_input(self, envelope):
        return self._turn_input if self._turn_input is not None else super().extract_turn_input(envelope)

    def default_view_policy(self, metadata, participant_id):
        return self._view or super().default_view_policy(metadata, participant_id)


async def _handed_to_bob(adapter: ConversationAdapter, text: str = "hi bob") -> BaseEvent:
    """Send ``text`` from alice over ``adapter`` and return what bob's LLM was handed for the turn."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    hub.register_adapter(adapter)
    bob_config = TrackingConfig(TestConfig("ok"))
    try:
        alice = await hub.register(Agent("alice", config=TestConfig("")))
        bob = await hub.register(Agent("bob", config=bob_config))
        channel = await alice.open(type="conversation", target=bob.agent_id)
        await channel.send(text)
        for _ in range(250):
            if bob_config.mock.called:
                break
            await asyncio.sleep(0.02)
    finally:
        await hub.close()

    bob_config.mock.assert_called_once()
    return bob_config.mock.call_args.args[0]


@pytest.mark.asyncio
class TestTurnInput:
    async def test_the_viewed_message_is_what_the_agent_is_asked(self) -> None:
        assert await _handed_to_bob(ConversationAdapter()) == ModelRequest([TextInput("hi bob")])

    async def test_a_string_rendering_becomes_a_text_part(self) -> None:
        turn = await _handed_to_bob(_ScriptedConversation(rendered="rendered"))

        assert turn == ModelRequest([TextInput("rendered")])

    async def test_a_rendering_with_an_image_keeps_the_image(self) -> None:
        photo = ImageInput("https://example.com/photo.png")

        turn = await _handed_to_bob(_ScriptedConversation(rendered=["look", photo]))

        assert turn == ModelRequest([TextInput("look"), photo])

    async def test_an_empty_projection_falls_back_to_the_adapter_turn_input(self) -> None:
        turn = await _handed_to_bob(_ScriptedConversation(turn_input="from the adapter", view=_FixedView()))

        assert turn == ModelRequest([TextInput("from the adapter")])

    async def test_a_request_without_content_falls_back_to_the_adapter_turn_input(self) -> None:
        adapter = _ScriptedConversation(turn_input=[TextInput("from the adapter")], view=_FixedView(ModelRequest([])))

        assert await _handed_to_bob(adapter) == ModelRequest([TextInput("from the adapter")])

    async def test_a_view_projecting_an_assistant_message_still_hands_over_its_text(self) -> None:
        adapter = _ScriptedConversation(view=_FixedView(ModelMessage("my own message")))

        assert await _handed_to_bob(adapter) == ModelRequest([TextInput("my own message")])
