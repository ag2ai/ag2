# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Built-in ``ViewPolicy`` implementations.

* :class:`FullTranscript` — projects every visible substantive envelope
  verbatim. Used by adapters where the assistant/user role bit already
  disambiguates speakers (i.e. 2-party).
* :class:`WindowedSummary` — keeps a bounded tail of recent envelopes
  and replaces older ones with a single :class:`CompactionSummary`
  event (static stat-style summary, no LLM call).
* :class:`NamedTranscript` — like :class:`FullTranscript` but
  prefixes each non-self projection line with ``[<sender name>]:``.
  Necessary in N-party (3+) channels where every non-self envelope
  would otherwise collapse into one indistinguishable ``user``-role
  stream.
* :class:`NamedWindowedSummary` — :class:`WindowedSummary` with the
  same speaker-name prefix. The default view for the ``discussion``
  and ``workflow`` adapters.

Policies dispatch envelope rendering through the adapter via the
``render_envelope`` callable supplied to ``project`` and resolve
sender identity via the ``name_for`` callable supplied by the default
handler — both keep view policies adapter-neutral and hub-independent.

The handler calls ``view.project([current_envelope], ...)`` for the
current-turn input using the same method as for history, so named views
label both history and the triggering message consistently.
"""

from ag2.compact import CompactionSummary
from ag2.events import BaseEvent, ModelMessage, ModelRequest, TextInput

from ..channel import ChannelMetadata
from ..envelope import Envelope, visible_to
from .base import EnvelopeRenderer, NameResolver, default_name_resolver

__all__ = (
    "FullTranscript",
    "NamedTranscript",
    "NamedWindowedSummary",
    "WindowedSummary",
)


class FullTranscript:
    """Translate every envelope visible to ``participant_id``.

    Projects whichever envelope types ``render_envelope`` returns a
    string for; other protocol-level events (``EV_CHANNEL_*``,
    ``EV_TASK_*``, expectation violations) are hub bookkeeping that
    the LLM doesn't need to reason about. The
    ``NetworkContextPolicy`` renders channel expectations / active task
    metadata into the prompt prefix instead.

    Inbound envelopes (sender != participant) become ``ModelRequest``
    (a "user turn"); own past envelopes become ``ModelMessage``. The
    assistant/user role bit is the speaker disambiguator — sufficient
    for 2-party channels where there's only one possible "other".
    Use :class:`NamedTranscript` for N-party channels.
    """

    name = "full_transcript"

    async def project(
        self,
        wal: list[Envelope],
        *,
        participant_id: str,
        channel: ChannelMetadata,
        render_envelope: EnvelopeRenderer,
        name_for: NameResolver = default_name_resolver,  # noqa: ARG002
    ) -> list[BaseEvent]:
        events: list[BaseEvent] = []
        for envelope in wal:
            if not visible_to(envelope, participant_id):
                continue
            text = render_envelope(envelope)
            if text is None:
                continue
            events.append(_to_event(envelope, text, participant_id))
        return events


class WindowedSummary:
    """Keep the last ``recent_n`` visible substantive envelopes
    verbatim; fold everything older into a single
    :class:`CompactionSummary` at the head of the projection.

    Bounds prompt size at any turn count — the projection is at most
    ``recent_n + 1`` events regardless of WAL length. ``CompactionSummary``
    is recognised by ``ag2/policies/conversation.py`` so it
    renders correctly in the LLM-facing message stream.

    The summary is a static stat-style line
    (``"Earlier in this channel: N messages from a, b."``) — no LLM
    call. Use :class:`NamedWindowedSummary` for N-party channels where
    the speaker labels matter.
    """

    name = "windowed_summary"

    def __init__(self, recent_n: int) -> None:
        if recent_n < 1:
            raise ValueError(f"recent_n must be >= 1, got {recent_n}")
        self._recent_n = recent_n

    @property
    def recent_n(self) -> int:
        return self._recent_n

    async def project(
        self,
        wal: list[Envelope],
        *,
        participant_id: str,
        channel: ChannelMetadata,
        render_envelope: EnvelopeRenderer,
        name_for: NameResolver = default_name_resolver,  # noqa: ARG002
    ) -> list[BaseEvent]:
        visible: list[tuple[Envelope, str]] = []
        for envelope in wal:
            if not visible_to(envelope, participant_id):
                continue
            text = render_envelope(envelope)
            if text is None:
                continue
            visible.append((envelope, text))

        if len(visible) <= self._recent_n:
            return [_to_event(env, txt, participant_id) for env, txt in visible]

        cutoff = len(visible) - self._recent_n
        older = visible[:cutoff]
        recent = visible[cutoff:]
        summary = _summarize_older([env for env, _ in older])
        compaction = CompactionSummary(summary=summary, event_count=len(older))
        return [compaction, *(_to_event(env, txt, participant_id) for env, txt in recent)]


class NamedTranscript:
    """Like :class:`FullTranscript` but prefixes non-self lines with the sender name.

    Used in N-party channels (3+ participants) where the assistant/user
    role bit alone can't tell the LLM which "other" said which message.
    Each non-self envelope is projected as
    ``ModelRequest([TextInput(f"[{sender_name}]: {body}")])`` so the
    LLM sees a name-prefixed stream instead of an unattributed collapse.

    Self envelopes still project as ``ModelMessage(text)`` — the
    assistant role already says "I said this," no label needed.
    """

    name = "named_transcript"

    async def project(
        self,
        wal: list[Envelope],
        *,
        participant_id: str,
        channel: ChannelMetadata,
        render_envelope: EnvelopeRenderer,
        name_for: NameResolver = default_name_resolver,
    ) -> list[BaseEvent]:
        events: list[BaseEvent] = []
        for envelope in wal:
            if not visible_to(envelope, participant_id):
                continue
            text = render_envelope(envelope)
            if text is None:
                continue
            events.append(_to_named_event(envelope, text, participant_id, name_for))
        return events


class NamedWindowedSummary:
    """:class:`WindowedSummary` with sender labels on non-self lines.

    The default view for the ``discussion`` and ``workflow`` adapters.
    Same bounded-tail-plus-compaction behaviour as
    :class:`WindowedSummary`; non-self projections are prefixed
    ``[<sender name>]:`` so the manager / next speaker can tell who
    just spoke. The head :class:`CompactionSummary` for elided
    envelopes also lists speakers by name rather than by raw id.
    """

    name = "named_windowed_summary"

    def __init__(self, recent_n: int) -> None:
        if recent_n < 1:
            raise ValueError(f"recent_n must be >= 1, got {recent_n}")
        self._recent_n = recent_n

    @property
    def recent_n(self) -> int:
        return self._recent_n

    async def project(
        self,
        wal: list[Envelope],
        *,
        participant_id: str,
        channel: ChannelMetadata,
        render_envelope: EnvelopeRenderer,
        name_for: NameResolver = default_name_resolver,
    ) -> list[BaseEvent]:
        visible: list[tuple[Envelope, str]] = []
        for envelope in wal:
            if not visible_to(envelope, participant_id):
                continue
            text = render_envelope(envelope)
            if text is None:
                continue
            visible.append((envelope, text))

        if len(visible) <= self._recent_n:
            return [_to_named_event(env, txt, participant_id, name_for) for env, txt in visible]

        cutoff = len(visible) - self._recent_n
        older = visible[:cutoff]
        recent = visible[cutoff:]
        summary = _summarize_older([env for env, _ in older], name_for=name_for)
        compaction = CompactionSummary(summary=summary, event_count=len(older))
        return [
            compaction,
            *(_to_named_event(env, txt, participant_id, name_for) for env, txt in recent),
        ]


def _to_event(envelope: Envelope, text: str, participant_id: str) -> BaseEvent:
    if envelope.sender_id == participant_id:
        return ModelMessage(text)
    return ModelRequest([TextInput(text)])


def _to_named_event(
    envelope: Envelope,
    text: str,
    participant_id: str,
    name_for: NameResolver,
) -> BaseEvent:
    if envelope.sender_id == participant_id:
        return ModelMessage(text)
    return ModelRequest([TextInput(f"[{name_for(envelope.sender_id)}]: {text}")])


def _summarize_older(older: list[Envelope], *, name_for: NameResolver | None = None) -> str:
    if name_for is None:
        speakers = sorted({e.sender_id for e in older})
    else:
        speakers = sorted({name_for(e.sender_id) for e in older})
    plural = "s" if len(older) != 1 else ""
    return f"Earlier in this channel: {len(older)} message{plural} from {', '.join(speakers)}."
