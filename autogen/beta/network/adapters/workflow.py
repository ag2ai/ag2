# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``WorkflowAdapter`` — orchestrated multi-party session driven by a
declarative :class:`TransitionGraph`.

The mechanic reuses what ``DiscussionAdapter(round_robin)`` already
does — folded ``expected_next_speaker`` gates ``validate_send`` — and
adds a richer rule for *how* ``expected_next_speaker`` advances. **No
hub changes required.**

knobs:
    graph: dict (TransitionGraph.to_dict() output) — required.

The adapter is stateless; the graph + creator + participant order are
snapshotted into ``WorkflowState`` at ``initial_state`` so ``fold``
(which has no metadata) can compute the next speaker on each accepted
envelope.

Default expectations declared on the manifest:
* ``turn_within(120s, warn)``
* ``turn_within(600s, auto_close)`` — long stalls fail-fast.

The session-level ``acks_within`` / ``reply_within`` / ``max_silence``
expectations apply regardless of adapter — workflow sessions inherit
them through the expectation sweeper.
"""

from dataclasses import dataclass, field
from typing import Any

from ..envelope import (
    EV_HANDOFF,
    EV_SESSION_CLOSED,
    EV_SESSION_EXPIRED,
    EV_SESSION_INVITE,
    EV_SESSION_INVITE_ACK,
    EV_SESSION_INVITE_REJECT,
    EV_SESSION_OPENED,
    EV_TEXT,
    Envelope,
)
from ..errors import ProtocolError
from ..session import (
    Expectation,
    ParticipantSchema,
    SessionManifest,
    SessionMetadata,
    SessionState,
)
from ..transitions import (
    TransitionDecision,
    TransitionGraph,
    WorkflowGraphError,
)
from ..views.base import ViewPolicy
from ..views.builtin import WindowedSummary
from .base import AdapterResult

__all__ = ("WORKFLOW_TYPE", "WorkflowAdapter", "WorkflowState")


WORKFLOW_TYPE = "workflow"


_SESSION_PROTOCOL_EVENTS: frozenset[str] = frozenset({
    EV_SESSION_INVITE,
    EV_SESSION_INVITE_ACK,
    EV_SESSION_INVITE_REJECT,
    EV_SESSION_OPENED,
    EV_SESSION_CLOSED,
    EV_SESSION_EXPIRED,
})


def _is_session_protocol_event(envelope: Envelope) -> bool:
    return envelope.event_type in _SESSION_PROTOCOL_EVENTS


def _is_task_event(envelope: Envelope) -> bool:
    return envelope.event_type.startswith("ag2.task.")


def _is_substantive(envelope: Envelope) -> bool:
    """A turn-advancing envelope: text or a workflow handoff."""
    if _is_session_protocol_event(envelope) or _is_task_event(envelope):
        return False
    return envelope.event_type in (EV_TEXT, EV_HANDOFF)


@dataclass(slots=True)
class WorkflowState:
    """Folded state for a workflow session.

    ``graph_data`` is the JSON-friendly ``TransitionGraph.to_dict()``
    snapshot taken at ``initial_state``. ``fold`` deserialises it on
    each call (cheap; the graph is small) so the adapter stays
    stateless across sessions.

    ``creator_id`` is snapshotted so ``RevertToInitiatorTarget`` can
    resolve without metadata access (``fold`` has no metadata).
    """

    participant_order: list[str] = field(default_factory=list)
    expected_next_speaker: str | None = None
    last_speaker_id: str | None = None
    last_envelope_id: str | None = None
    turn_count: int = 0
    pending_close_reason: str = ""
    creator_id: str = ""
    graph_data: dict[str, Any] = field(default_factory=dict)


class WorkflowAdapter:
    """Generic orchestrated multi-party session.

    Knobs: ``{"graph": <TransitionGraph.to_dict()>}``. Participants:
    2+. Default view: :class:`WindowedSummary(recent_n=N*2)` with
    ``N`` = participant count.
    """

    def __init__(self) -> None:
        self.manifest = SessionManifest(
            type=WORKFLOW_TYPE,
            version=1,
            participants=ParticipantSchema(min=2),
            knobs_schema={"graph": "TransitionGraph"},
            default_view_policy=WindowedSummary.name,
            expectations=[
                Expectation(
                    name="turn_within",
                    on_violation="warn",
                    params={"seconds": 120},
                ),
                Expectation(
                    name="turn_within",
                    on_violation="auto_close",
                    params={"seconds": 600},
                ),
            ],
        )

    # ── Adapter Protocol ────────────────────────────────────────────────────

    def initial_state(self, metadata: SessionMetadata) -> WorkflowState:
        graph_data = metadata.knobs.get("graph")
        if not isinstance(graph_data, dict):
            raise ProtocolError(
                "workflow requires knobs['graph'] as a dict — call "
                "TransitionGraph.to_dict() before passing"
            )
        try:
            graph = TransitionGraph.loads(graph_data)
        except WorkflowGraphError as exc:
            raise ProtocolError(f"invalid workflow graph: {exc}") from exc

        order = [
            p.agent_id
            for p in sorted(metadata.participants, key=lambda p: p.order)
        ]
        if graph.initial_speaker not in order:
            raise ProtocolError(
                f"workflow initial_speaker {graph.initial_speaker!r} not in "
                f"participants {order!r}"
            )
        return WorkflowState(
            participant_order=order,
            expected_next_speaker=graph.initial_speaker,
            creator_id=metadata.creator_id,
            graph_data=graph_data,
        )

    def fold(self, envelope: Envelope, state: WorkflowState) -> WorkflowState:
        if not _is_substantive(envelope):
            return state

        graph = TransitionGraph.loads(state.graph_data)

        # Build the post-fold state with bookkeeping advanced; speaker
        # selection happens against this state so transitions see the
        # turn count and last speaker that include this envelope.
        new_state = WorkflowState(
            participant_order=state.participant_order,
            expected_next_speaker=state.expected_next_speaker,
            last_speaker_id=envelope.sender_id,
            last_envelope_id=envelope.envelope_id,
            turn_count=state.turn_count + 1,
            pending_close_reason="",
            creator_id=state.creator_id,
            graph_data=state.graph_data,
        )

        decision = self._select(graph, new_state, envelope)
        new_state.expected_next_speaker = decision.next_speaker
        new_state.pending_close_reason = decision.close_reason
        return new_state

    def validate_create(self, metadata: SessionMetadata) -> None:
        if len(metadata.participants) < 2:
            raise ProtocolError(
                f"workflow requires at least 2 participants, got "
                f"{len(metadata.participants)}"
            )
        graph_data = metadata.knobs.get("graph")
        if not isinstance(graph_data, dict):
            raise ProtocolError(
                "workflow requires knobs['graph'] as a dict — call "
                "TransitionGraph.to_dict() before passing"
            )
        try:
            graph = TransitionGraph.loads(graph_data)
        except WorkflowGraphError as exc:
            raise ProtocolError(f"invalid workflow graph: {exc}") from exc
        order = {p.agent_id for p in metadata.participants}
        if graph.initial_speaker not in order:
            raise ProtocolError(
                f"workflow initial_speaker {graph.initial_speaker!r} not "
                f"in participants {sorted(order)!r}"
            )

    def validate_send(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        state: WorkflowState,
    ) -> None:
        if not _is_substantive(envelope):
            return
        if (
            state.expected_next_speaker
            and envelope.sender_id != state.expected_next_speaker
        ):
            raise ProtocolError(
                f"workflow {metadata.session_id!r} expects "
                f"{state.expected_next_speaker!r} to speak, got "
                f"{envelope.sender_id!r}"
            )

    def on_accepted(
        self,
        metadata: SessionMetadata,
        envelope: Envelope,
        state: WorkflowState,
    ) -> AdapterResult:
        if not _is_substantive(envelope):
            return AdapterResult()

        graph = TransitionGraph.loads(state.graph_data)
        if graph.max_turns is not None and state.turn_count >= graph.max_turns:
            return AdapterResult(
                next_state=SessionState.CLOSED,
                auto_close_reason="max_turns",
            )

        if state.expected_next_speaker is None:
            reason = state.pending_close_reason or "workflow_terminated"
            return AdapterResult(
                next_state=SessionState.CLOSED,
                auto_close_reason=reason,
            )
        return AdapterResult()

    def default_view_policy(
        self,
        metadata: SessionMetadata,
        participant_id: str,
    ) -> ViewPolicy:
        recent_n = max(len(metadata.participants) * 2, 4)
        return WindowedSummary(recent_n=recent_n)

    # ── Internals ───────────────────────────────────────────────────────────

    @staticmethod
    def _select(
        graph: TransitionGraph,
        state: WorkflowState,
        envelope: Envelope,
    ) -> TransitionDecision:
        for tr in sorted(graph.transitions, key=lambda t: t.priority):
            if tr.when.evaluate(state, envelope):
                return tr.then.resolve(state, envelope)
        return graph.default_target.resolve(state, envelope)
