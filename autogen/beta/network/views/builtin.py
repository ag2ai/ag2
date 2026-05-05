# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Built-in ``ViewPolicy`` implementations.

* :class:`FullTranscript` — projects every visible substantive envelope
  verbatim.
* :class:`WindowedSummary` — keeps a bounded tail of recent envelopes
  and replaces older ones with a single :class:`CompactionSummary`
  event (static stat-style summary, no LLM call).
"""

from autogen.beta.compact import CompactionSummary
from autogen.beta.events import BaseEvent, ModelMessage, ModelRequest, TextInput

from ..envelope import EV_HANDOFF, EV_TEXT, Envelope, visible_to
from ..session import SessionMetadata

__all__ = ("FullTranscript", "WindowedSummary")


_PROJECTED_EVENT_TYPES = frozenset({EV_TEXT, EV_HANDOFF})


def _envelope_text(envelope: Envelope) -> str | None:
    """Render a substantive envelope into the text the LLM should see.

    Returns ``None`` for envelopes that should be skipped (non-text
    payload, unsupported event type). ``EV_HANDOFF`` envelopes are
    rendered as ``"[Handed off via <tool>] <reason>"`` so multi-hop
    workflows preserve the conversation thread on subsequent turns —
    without this, ``WindowedSummary``/``FullTranscript`` would drop the
    handoff and later turns would show replies without their triggers.
    """
    if envelope.event_type == EV_TEXT:
        text = envelope.event_data.get("text", "")
        if not isinstance(text, str):
            return None
        return text
    if envelope.event_type == EV_HANDOFF:
        tool = envelope.event_data.get("tool", "")
        reason = envelope.event_data.get("reason", "")
        if not isinstance(tool, str):
            tool = str(tool)
        if not isinstance(reason, str):
            reason = str(reason)
        rendered = f"[Handed off via {tool}] {reason}".strip()
        return rendered or None
    return None


class FullTranscript:
    """Translate every envelope visible to ``participant_id``.

    Projects ``EV_TEXT`` and ``EV_HANDOFF`` envelopes. Other protocol-
    level events (``EV_SESSION_*``, ``EV_TASK_*``, expectation
    violations) are hub bookkeeping that the LLM doesn't need to reason
    about. The ``NetworkContextPolicy`` renders session expectations /
    active task metadata into the prompt prefix instead.

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
    ) -> list[BaseEvent]:
        events: list[BaseEvent] = []
        for envelope in wal:
            if not visible_to(envelope, participant_id):
                continue
            if envelope.event_type not in _PROJECTED_EVENT_TYPES:
                continue
            text = _envelope_text(envelope)
            if text is None:
                continue
            if envelope.sender_id == participant_id:
                events.append(ModelMessage(text))
            else:
                events.append(ModelRequest([TextInput(text)]))
        return events


class WindowedSummary:
    """Keep the last ``recent_n`` visible ``EV_TEXT`` envelopes verbatim;
    fold everything older into a single :class:`CompactionSummary` at
    the head of the projection.

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
    ) -> list[BaseEvent]:
        visible: list[tuple[Envelope, str]] = []
        for envelope in wal:
            if not visible_to(envelope, participant_id):
                continue
            if envelope.event_type not in _PROJECTED_EVENT_TYPES:
                continue
            text = _envelope_text(envelope)
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
