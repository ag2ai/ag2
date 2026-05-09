# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Built-in ``ViewPolicy`` implementations.

* :class:`FullTranscript` — projects every visible substantive envelope
  verbatim.
* :class:`WindowedSummary` — keeps a bounded tail of recent envelopes
  and replaces older ones with a single :class:`CompactionSummary`
  event (static stat-style summary, no LLM call).

Policies dispatch envelope rendering through the adapter via the
``render_envelope`` callable supplied to ``project``, so they stay
adapter-neutral.
"""

from autogen.beta.compact import CompactionSummary
from autogen.beta.events import BaseEvent, ModelMessage, ModelRequest, TextInput

from ..envelope import Envelope, visible_to
from ..session import SessionMetadata
from .base import EnvelopeRenderer

__all__ = ("FullTranscript", "WindowedSummary")


class FullTranscript:
    """Translate every envelope visible to ``participant_id``.

    Projects whichever envelope types ``render_envelope`` returns a
    string for; other protocol-level events (``EV_SESSION_*``,
    ``EV_TASK_*``, expectation violations) are hub bookkeeping that
    the LLM doesn't need to reason about. The
    ``NetworkContextPolicy`` renders session expectations / active task
    metadata into the prompt prefix instead.

    Inbound envelopes (sender != participant) become ``ModelRequest``
    (a "user turn"); own past envelopes become ``ModelMessage``.
    """

    name = "full_transcript"

    async def project(
        self,
        wal: list[Envelope],
        *,
        participant_id: str,
        session: SessionMetadata,
        render_envelope: EnvelopeRenderer,
    ) -> list[BaseEvent]:
        events: list[BaseEvent] = []
        for envelope in wal:
            if not visible_to(envelope, participant_id):
                continue
            text = render_envelope(envelope)
            if text is None:
                continue
            if envelope.sender_id == participant_id:
                events.append(ModelMessage(text))
            else:
                events.append(ModelRequest([TextInput(text)]))
        return events


class WindowedSummary:
    """Keep the last ``recent_n`` visible substantive envelopes
    verbatim; fold everything older into a single
    :class:`CompactionSummary` at the head of the projection.

    Bounds prompt size at any turn count — the projection is at most
    ``recent_n + 1`` events regardless of WAL length. ``CompactionSummary``
    is recognised by ``autogen/beta/policies/conversation.py`` so it
    renders correctly in the LLM-facing message stream.

    The summary is a static stat-style line
    (``"Earlier in this session: N messages from a, b."``) — no LLM
    call.
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
        session: SessionMetadata,
        render_envelope: EnvelopeRenderer,
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


def _to_event(envelope: Envelope, text: str, participant_id: str) -> BaseEvent:
    if envelope.sender_id == participant_id:
        return ModelMessage(text)
    return ModelRequest([TextInput(text)])


def _summarize_older(older: list[Envelope]) -> str:
    speakers = sorted({e.sender_id for e in older})
    plural = "s" if len(older) != 1 else ""
    return f"Earlier in this session: {len(older)} message{plural} from {', '.join(speakers)}."
