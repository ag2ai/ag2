# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``WorkflowAdapter`` — orchestrated multi-party channel driven by a
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

The channel-level ``acks_within`` / ``reply_within`` / ``max_silence``
expectations apply regardless of adapter — workflow channels inherit
them through the expectation sweeper.
"""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ag2.events import BaseEvent, DataInput, ToolCallEvent, ToolResultEvent

from ..channel import (
    ChannelManifest,
    ChannelMetadata,
    ChannelState,
    Expectation,
    ParticipantSchema,
)
from ..envelope import (
    EV_CHANNEL_CLOSED,
    EV_CHANNEL_EXPIRED,
    EV_CHANNEL_INVITE,
    EV_CHANNEL_INVITE_ACK,
    EV_CHANNEL_INVITE_REJECT,
    EV_CHANNEL_OPENED,
    EV_CONTEXT_SET,
    EV_PACKET,
    EV_TEXT,
    Envelope,
)
from ..errors import ProtocolError
from ..handoff import Finish, Handoff
from ..transitions import (
    ToolCalled,
    TransitionDecision,
    TransitionGraph,
    WorkflowGraphError,
)
from ..views.base import ViewPolicy
from ..views.builtin import NamedWindowedSummary
from .base import (
    AdapterResult,
    ExpectedTurn,
    default_build_packet_envelope,
    default_build_text_envelope,
    default_render_envelope,
    default_tools_for,
)

if TYPE_CHECKING:
    from ag2.agent import AgentReply

    from ..hub.core import Hub

__all__ = ("WORKFLOW_TYPE", "WorkflowAdapter", "WorkflowState")


WORKFLOW_TYPE = "workflow"


_CHANNEL_PROTOCOL_EVENTS: frozenset[str] = frozenset({
    EV_CHANNEL_INVITE,
    EV_CHANNEL_INVITE_ACK,
    EV_CHANNEL_INVITE_REJECT,
    EV_CHANNEL_OPENED,
    EV_CHANNEL_CLOSED,
    EV_CHANNEL_EXPIRED,
})


def _is_channel_protocol_event(envelope: Envelope) -> bool:
    return envelope.event_type in _CHANNEL_PROTOCOL_EVENTS


def _is_task_event(envelope: Envelope) -> bool:
    return envelope.event_type.startswith("ag2.task.")


def _is_substantive(envelope: Envelope) -> bool:
    """A turn-advancing envelope: text or a packet (one Agent.ask round)."""
    if _is_channel_protocol_event(envelope) or _is_task_event(envelope):
        return False
    return envelope.event_type in (EV_TEXT, EV_PACKET)


@dataclass(slots=True)
class WorkflowState:
    """Folded state for a workflow channel.

    ``graph_data`` is the JSON-friendly ``TransitionGraph.to_dict()``
    snapshot taken at ``initial_state``. ``fold`` deserialises it on
    each call (cheap; the graph is small) so the adapter stays
    stateless across channels.

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
    context_vars: dict[str, Any] = field(default_factory=dict)


class WorkflowAdapter:
    """Generic orchestrated multi-party channel.

    Knobs: ``{"graph": <TransitionGraph.to_dict()>}``. Participants:
    2+. Default view: :class:`NamedWindowedSummary(recent_n=N*2)`
    with ``N`` = participant count — bounded prompt size plus sender
    labels on non-self projection lines so the orchestrator / next
    speaker can tell its peers apart in a 3+ party chat (the
    assistant/user role bit alone collapses every "other" into one
    indistinguishable stream).
    """

    def __init__(self) -> None:
        self.manifest = ChannelManifest(
            type=WORKFLOW_TYPE,
            version=1,
            participants=ParticipantSchema(min=2),
            knobs_schema={"graph": "TransitionGraph"},
            default_view_policy=NamedWindowedSummary.name,
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

    def initial_state(self, metadata: ChannelMetadata) -> WorkflowState:
        graph_data = metadata.knobs.get("graph")
        if not isinstance(graph_data, dict):
            raise ProtocolError(
                "workflow requires knobs['graph'] as a dict — call TransitionGraph.to_dict() before passing"
            )
        try:
            graph = TransitionGraph.loads(graph_data)
        except WorkflowGraphError as exc:
            raise ProtocolError(f"invalid workflow graph: {exc}") from exc

        order = [p.agent_id for p in sorted(metadata.participants, key=lambda p: p.order)]
        if graph.initial_speaker not in order:
            raise ProtocolError(f"workflow initial_speaker {graph.initial_speaker!r} not in participants {order!r}")
        initial_context = metadata.knobs.get("context_vars", {})
        if not isinstance(initial_context, dict):
            raise ProtocolError("workflow knobs['context_vars'] must be a dict if provided")
        return WorkflowState(
            participant_order=order,
            expected_next_speaker=graph.initial_speaker,
            creator_id=metadata.creator_id,
            graph_data=graph_data,
            context_vars=dict(initial_context),
        )

    def fold(self, envelope: Envelope, state: WorkflowState) -> WorkflowState:
        # Context-variable mutations don't advance turn bookkeeping —
        # they're auxiliary metadata, applied before the substantive
        # gate so the new value is visible to the next fold.
        if envelope.event_type == EV_CONTEXT_SET:
            new_vars = dict(state.context_vars)
            for key in envelope.event_data.get("delete", []) or []:
                new_vars.pop(key, None)
            new_vars.update(envelope.event_data.get("set", {}) or {})
            return WorkflowState(
                participant_order=state.participant_order,
                expected_next_speaker=state.expected_next_speaker,
                last_speaker_id=state.last_speaker_id,
                last_envelope_id=state.last_envelope_id,
                turn_count=state.turn_count,
                pending_close_reason=state.pending_close_reason,
                creator_id=state.creator_id,
                graph_data=state.graph_data,
                context_vars=new_vars,
            )

        if not _is_substantive(envelope):
            return state

        graph = TransitionGraph.loads(state.graph_data)

        # Apply context mutations carried on the packet BEFORE
        # ``select_next`` so a ``ContextEquals`` rule can match values
        # the same packet just set (atomic state-update + speaker-
        # advance).
        new_context = dict(state.context_vars)
        if envelope.event_type == EV_PACKET:
            updates = envelope.event_data.get("context_updates", {}) or {}
            for key in updates.get("delete", []) or []:
                new_context.pop(key, None)
            new_context.update(updates.get("set", {}) or {})

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
            context_vars=new_context,
        )

        # Routing resolution.
        # ``kind: "finish"`` short-circuits to termination — a tool
        # that returned ``Finish`` is the most explicit intent the
        # agent can express. ``kind: "handoff"`` with a pre-resolved
        # ``routing.target`` (from a typed ``Handoff`` return) trusts
        # the tool's pick directly. Otherwise run ``select_next`` so
        # static rules (``ToolCalled``, ``ContextEquals``,
        # ``FromSpeaker``) decide.
        finish_reason: str | None = None
        pre_resolved = None
        if envelope.event_type == EV_PACKET:
            routing = envelope.event_data.get("routing", {}) or {}
            if routing.get("kind") == "finish":
                finish_reason = routing.get("reason") or "finished"
            elif routing.get("kind") == "handoff":
                pre_resolved = routing.get("target")

        if finish_reason is not None:
            new_state.expected_next_speaker = None
            new_state.pending_close_reason = finish_reason
        elif pre_resolved:
            new_state.expected_next_speaker = pre_resolved
            new_state.pending_close_reason = ""
        else:
            decision = self._select(graph, new_state, envelope)
            new_state.expected_next_speaker = decision.next_speaker
            new_state.pending_close_reason = decision.close_reason
        return new_state

    def validate_create(self, metadata: ChannelMetadata) -> None:
        if len(metadata.participants) < 2:
            raise ProtocolError(f"workflow requires at least 2 participants, got {len(metadata.participants)}")
        graph_data = metadata.knobs.get("graph")
        if not isinstance(graph_data, dict):
            raise ProtocolError(
                "workflow requires knobs['graph'] as a dict — call TransitionGraph.to_dict() before passing"
            )
        try:
            graph = TransitionGraph.loads(graph_data)
        except WorkflowGraphError as exc:
            raise ProtocolError(f"invalid workflow graph: {exc}") from exc
        order = {p.agent_id for p in metadata.participants}
        if graph.initial_speaker not in order:
            raise ProtocolError(
                f"workflow initial_speaker {graph.initial_speaker!r} not in participants {sorted(order)!r}"
            )

    def validate_send(
        self,
        metadata: ChannelMetadata,
        envelope: Envelope,
        state: WorkflowState,
    ) -> None:
        if envelope.event_type == EV_CONTEXT_SET:
            # Context variables can be set by any valid participant at any time
            participant_ids = {p.agent_id for p in metadata.participants}
            if envelope.sender_id not in participant_ids:
                raise ProtocolError(
                    f"workflow {metadata.channel_id!r} only accepts EV_CONTEXT_SET "
                    f"from participants, got {envelope.sender_id!r}"
                )
            return
        if not _is_substantive(envelope):
            return
        if state.expected_next_speaker and envelope.sender_id != state.expected_next_speaker:
            raise ProtocolError(
                f"workflow {metadata.channel_id!r} expects "
                f"{state.expected_next_speaker!r} to speak, got "
                f"{envelope.sender_id!r}"
            )

    def on_accepted(
        self,
        metadata: ChannelMetadata,
        envelope: Envelope,
        state: WorkflowState,
    ) -> AdapterResult:
        if not _is_substantive(envelope):
            return AdapterResult()

        # Prefer graph-emitted termination reason over max_turns
        if state.expected_next_speaker is None:
            reason = state.pending_close_reason or "workflow_terminated"
            return AdapterResult(
                next_state=ChannelState.CLOSED,
                auto_close_reason=reason,
            )

        graph = TransitionGraph.loads(state.graph_data)
        if graph.max_turns is not None and state.turn_count >= graph.max_turns:
            return AdapterResult(
                next_state=ChannelState.CLOSED,
                auto_close_reason="max_turns",
            )
        return AdapterResult()

    def expected_next(
        self,
        metadata: ChannelMetadata,
        state: WorkflowState,
    ) -> ExpectedTurn | None:
        if state.expected_next_speaker is None:
            return None
        return ExpectedTurn(
            agent_id=state.expected_next_speaker,
            triggering_envelope_id=state.last_envelope_id,
        )

    def default_view_policy(
        self,
        metadata: ChannelMetadata,
        participant_id: str,
    ) -> ViewPolicy:
        recent_n = max(len(metadata.participants) * 2, 4)
        return NamedWindowedSummary(recent_n=recent_n)

    def extract_turn_input(self, envelope: Envelope) -> str | None:
        """Decode an inbound substantive envelope into the next
        speaker's prompt. Workflow handles ``EV_TEXT`` and
        ``EV_PACKET`` (concatenates the routing-handoff line with
        the body)."""
        if envelope.event_type == EV_TEXT:
            text = envelope.event_data.get("text", "")
            return text if isinstance(text, str) else None
        if envelope.event_type == EV_PACKET:
            return _packet_turn_text(envelope) or None
        return None

    def build_round_envelope(
        self,
        metadata: ChannelMetadata,
        sender_id: str,
        reply: "AgentReply",
        events: list[BaseEvent],
        state: "WorkflowState | None",
        hub: "Hub",
    ) -> Envelope | None:
        """Build the ``EV_PACKET`` envelope capturing this round.

        Walks the agent's local-stream events to determine routing
        intent (Handoff result first, then ToolCalled match, else
        text). ``select_next`` resolves the target at fold time for
        static routing; dynamic Handoff carries its resolved target
        on the packet's ``routing.target`` field.

        Returns ``None`` for silent rounds (no body and no routing
        tool fired) — matches pre-packet "no envelope" behaviour.
        """
        graph: TransitionGraph | None = None
        if state is not None and state.graph_data:
            try:
                graph = TransitionGraph.loads(state.graph_data)
            except WorkflowGraphError:
                graph = None

        routing = _resolve_routing(events, graph, hub.name_to_id_map())
        body = reply.body or ""

        if routing["kind"] == "text" and not body:
            return None

        return Envelope(
            channel_id=metadata.channel_id,
            sender_id=sender_id,
            audience=None,
            event_type=EV_PACKET,
            event_data={
                "routing": routing,
                "context_updates": {"set": {}, "delete": []},
                "body": body,
            },
        )

    def render_envelope(self, envelope):
        """Project ``EV_PACKET`` via :func:`_packet_text`; defer to
        :func:`default_render_envelope` for everything else (notably
        ``EV_TEXT`` for participant text emitted outside the
        workflow's round-end packet)."""
        if envelope.event_type == EV_PACKET:
            return _packet_text(envelope)
        return default_render_envelope(envelope)

    def tools_for(self, client, metadata, state, participant_id):
        """Workflow offers no adapter-level tools.

        Handoff routing is encoded by user-authored ``@tool`` functions
        that return :class:`Handoff(target=, reason=)`. The handler
        merges those tools (already on ``agent.tools``) with the
        identity-level ``NetworkPlugin`` set; the workflow adapter
        itself contributes nothing.
        """
        return default_tools_for(client, metadata, state, participant_id)

    def build_text_envelope(self, channel_id, sender_id, text, *, audience=None, causation_id=None):
        """Workflow accepts text seeds (e.g. for an initiator's first
        turn) as plain ``EV_TEXT`` — the adapter folds them into the
        round-end packet downstream."""
        return default_build_text_envelope(channel_id, sender_id, text, audience=audience, causation_id=causation_id)

    def build_packet_envelope(
        self,
        channel_id,
        sender_id,
        body,
        *,
        handoff=None,
        context_set=None,
        audience=None,
        causation_id=None,
    ):
        """Workflow's native round-end shape — handoff + context_set
        live in ``routing`` / ``context`` fields."""
        return default_build_packet_envelope(
            channel_id,
            sender_id,
            body,
            handoff=handoff,
            context_set=context_set,
            audience=audience,
            causation_id=causation_id,
        )

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


# ── Packet-construction helpers ──────────────────────────────────────────────


def _packet_turn_text(envelope: Envelope) -> str:
    """Synthesise a turn prompt from an ``EV_PACKET`` envelope.

    Concatenates the handoff signal line (if any) with the body
    (if any) using a newline. Mirrors the view projection so the
    next speaker sees the same shape on their first turn after a
    handoff that they'd see in the conversation history afterwards.
    """
    routing = envelope.event_data.get("routing", {}) or {}
    body = envelope.event_data.get("body", "")
    if not isinstance(body, str):
        body = ""

    parts: list[str] = []
    if routing.get("kind") == "handoff":
        tool = routing.get("tool", "handoff")
        reason = routing.get("reason", "")
        if not isinstance(tool, str):
            tool = str(tool)
        if not isinstance(reason, str):
            reason = str(reason)
        if reason:
            parts.append(f"[Handed off via {tool}] {reason}")
        else:
            parts.append(f"[Handed off via {tool}]")
    if body:
        parts.append(body)
    return "\n".join(parts)


def _packet_text(envelope: Envelope) -> "str | None":
    """Render an ``EV_PACKET`` envelope for view projection.

    Concatenates the handoff signal line (if ``routing.kind ==
    "handoff"``) with the body (if any) using a newline. Returns
    ``None`` if both are empty so the projection skips the envelope.
    """
    routing = envelope.event_data.get("routing", {}) or {}
    body = envelope.event_data.get("body", "")
    if not isinstance(body, str):
        body = ""

    parts: list[str] = []
    if routing.get("kind") == "handoff":
        tool = routing.get("tool", "")
        reason = routing.get("reason", "")
        if not isinstance(tool, str):
            tool = str(tool)
        if not isinstance(reason, str):
            reason = str(reason)
        handoff_line = f"[Handed off via {tool}] {reason}".strip()
        if handoff_line:
            parts.append(handoff_line)
    if body:
        parts.append(body)

    if not parts:
        return None
    return "\n".join(parts)


def _resolve_routing(
    events: list[BaseEvent],
    graph: TransitionGraph | None,
    name_to_id: dict[str, str],
) -> dict[str, Any]:
    """Determine the packet's ``routing`` field from agent local-stream events.

    First-emitted-wins: walk ``ToolCallEvent``s in stream emission
    order (which mirrors the LLM's left-to-right tool-call order).
    For each call, in priority:

    1. Dynamic ``Finish`` — does the corresponding
       ``ToolResultEvent`` carry a ``Finish`` instance? If so, the
       call ends the channel; ``reason``/``summary`` ride on the
       packet's routing field.
    2. Dynamic ``Handoff`` — does the corresponding
       ``ToolResultEvent`` carry a ``Handoff`` instance? If so, the
       call's name + Handoff's ``target``/``reason`` becomes the
       routing.
    3. Static ``ToolCalled`` match — does the call's name match a
       graph rule? If so, the call's name becomes the routing tool;
       ``select_next`` resolves the target at fold time.

    Returns ``{"kind": "text"}`` if no routing tool fired — the
    body drives ``select_next``.
    """
    if graph is None:
        return {"kind": "text"}

    tool_called_names: set[str] = set()
    for tr in graph.transitions:
        if isinstance(tr.when, ToolCalled):
            tool_called_names.add(tr.when.tool_name)

    calls: list[ToolCallEvent] = []
    results_by_parent: dict[str, ToolResultEvent] = {}
    for ev in events:
        if isinstance(ev, ToolCallEvent):
            calls.append(ev)
        elif isinstance(ev, ToolResultEvent):
            results_by_parent[ev.parent_id] = ev

    for call in calls:
        result_event = results_by_parent.get(call.id)
        finish = _extract_finish(result_event) if result_event else None
        if finish is not None:
            return {
                "kind": "finish",
                "tool": call.name,
                "reason": finish.reason,
                "summary": finish.summary,
            }
        handoff = _extract_handoff(result_event) if result_event else None
        if handoff is not None:
            target = name_to_id.get(handoff.target, handoff.target)
            return {
                "kind": "handoff",
                "tool": call.name,
                "reason": handoff.reason,
                "target": target,
            }
        if call.name in tool_called_names:
            return {
                "kind": "handoff",
                "tool": call.name,
                "reason": _extract_call_reason(call),
            }

    return {"kind": "text"}


def _extract_handoff(result_event: ToolResultEvent) -> Handoff | None:
    """Look for a ``Handoff`` instance in a tool result's parts."""
    result = result_event.result
    if result is None or not result.parts:
        return None
    for part in result.parts:
        if isinstance(part, DataInput) and isinstance(part.data, Handoff):
            return part.data
    return None


def _extract_finish(result_event: ToolResultEvent) -> Finish | None:
    """Look for a ``Finish`` instance in a tool result's parts."""
    result = result_event.result
    if result is None or not result.parts:
        return None
    for part in result.parts:
        if isinstance(part, DataInput) and isinstance(part.data, Finish):
            return part.data
    return None


def _extract_call_reason(call: ToolCallEvent) -> str:
    """Pull the LLM-supplied ``reason`` argument out of a tool call.

    Routing tools conventionally take a ``reason: str`` argument
    that the LLM fills in to explain its decision; the reason
    lands in the projection as ``[Handed off via X] reason``.
    Returns empty string if absent or unparsable.
    """
    arguments = call.arguments
    if not isinstance(arguments, str) or not arguments:
        return ""
    try:
        parsed = json.loads(arguments)
    except (ValueError, TypeError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    reason = parsed.get("reason", "")
    return reason if isinstance(reason, str) else ""
