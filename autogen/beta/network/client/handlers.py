# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Default notify handler.

Routes inbound envelopes to the right action:

* ``EV_SESSION_INVITE`` → auto-ack (post ``EV_SESSION_INVITE_ACK``).
* ``EV_SESSION_*`` other → no-op (state is bookkeeping; the agent
  doesn't need to react).
* ``EV_TEXT`` → read WAL, project view, run ``agent.ask`` with the
  projection pre-populated as stream history, send any non-empty
  reply via ``Session.send``.
* ``ag2.task.*`` → no-op (mirrored separately by ``TaskMirror``).

The handler is decomposed into small public hooks
(``read_wal_until``, ``resolve_view_policy``, ``stamp_dependencies``)
so user-supplied overrides can replace only the parts they care about.
"""

import contextlib
from typing import TYPE_CHECKING

from autogen.beta.events import BaseEvent
from autogen.beta.stream import MemoryStream

from ..envelope import (
    EV_SESSION_INVITE,
    EV_SESSION_INVITE_ACK,
    Envelope,
)
from ..policies import AGENT_CLIENT_DEP, HUB_DEP, SESSION_DEP, SESSION_STATE_DEP
from ..session import SessionMetadata, SessionState
from ..task_mirror import TaskMirror
from ..views.base import ViewPolicy
from .session import Session

if TYPE_CHECKING:
    from .agent_client import AgentClient

__all__ = (
    "default_handler",
    "read_wal_until",
    "resolve_view_policy",
    "stamp_dependencies",
)


def _is_task_event(event_type: str) -> bool:
    return event_type.startswith("ag2.task.")


async def read_wal_until(client: "AgentClient", envelope: Envelope) -> list[Envelope]:
    """Return WAL slice up to (but excluding) ``envelope``.

    The current envelope is the prompt for this turn — it is fed to
    ``agent.ask`` separately rather than mixed into the projected
    history.
    """
    wal = await client._hub_client.read_wal(envelope.session_id)
    history: list[Envelope] = []
    for env in wal:
        if env.envelope_id == envelope.envelope_id:
            break
        history.append(env)
    return history


def resolve_view_policy(
    client: "AgentClient",
    metadata: SessionMetadata,
) -> ViewPolicy:
    """Return the adapter's default view policy for this participant."""
    return client._hub_client.default_view_policy(metadata.session_id, client.agent_id)


def stamp_dependencies(
    client: "AgentClient",
    session: Session,
) -> dict[object, object]:
    """Build the ``context.dependencies`` dict for the LLM turn.

    ``SESSION_STATE_DEP`` resolves to the adapter's current State
    object (``WorkflowState`` / ``DiscussionState`` / ...). Tools that
    need to read session-scoped state (e.g. ``context_vars`` on a
    workflow session) inject it via ``SessionStateInject``.
    """
    return {
        SESSION_DEP: session,
        AGENT_CLIENT_DEP: client,
        HUB_DEP: client._hub,
        SESSION_STATE_DEP: client._hub_client.adapter_state(session.session_id),
    }


async def _auto_ack_invite(envelope: Envelope, client: "AgentClient") -> None:
    """Default behaviour: ack any invite addressed to us.

    Policy-based rejection (``EV_SESSION_INVITE_REJECT`` on access
    denial / capacity) is the override path — replace this handler in a
    custom callback wired via ``AgentClient.on_envelope``.
    """
    ack = Envelope(
        session_id=envelope.session_id,
        sender_id=client.agent_id,
        audience=None,
        event_type=EV_SESSION_INVITE_ACK,
        event_data={"session_id": envelope.session_id},
        causation_id=envelope.envelope_id,
    )
    # An ack failure shouldn't crash the agent — the hub will time
    # out and close the session via ``invite_ack_timeout``.
    with contextlib.suppress(Exception):
        await client.send_envelope(ack)


async def _process_substantive(envelope: Envelope, client: "AgentClient") -> None:
    """Run the agent's LLM on an inbound substantive envelope and
    post the round-end envelope built by the adapter.

    Adapter-agnostic: dispatches both inbound decoding (envelope →
    LLM prompt input) and outbound encoding (round result → envelope)
    through ``adapter.extract_turn_input`` /
    ``adapter.build_round_envelope`` so this handler stays free of
    adapter-specific knowledge.
    """
    metadata = await client._hub_client.get_session(envelope.session_id)
    if metadata.is_terminal() or metadata.state != SessionState.ACTIVE:
        return

    # "Can we respond now?" — ask the hub via the public probe surface
    # so the handler doesn't need to reach into adapter internals.
    if not client._hub_client.can_send(envelope.session_id, client.agent_id):
        return  # not our turn / session closing — don't engage LLM

    adapter = client._hub_client.adapter_for(metadata.session_id)
    session = Session(metadata=metadata, client=client)
    view = resolve_view_policy(client, metadata)

    history_envelopes = await read_wal_until(client, envelope)
    projection: list[BaseEvent] = await view.project(
        history_envelopes,
        participant_id=client.agent_id,
        session=metadata,
        render_envelope=adapter.render_envelope,
    )

    current_input = adapter.extract_turn_input(envelope)
    if not current_input:
        return

    # Pre-populate a fresh stream's history with the projection so the
    # agent's middleware sees the prior conversation context. The
    # current turn's user message is passed via ``msg`` and gets
    # appended to history naturally by ``Agent.ask``.
    stream = MemoryStream()
    if projection:
        await stream.history.storage.set_history(stream.id, projection)

    dependencies = stamp_dependencies(client, session)

    # Attach the TaskMirror for the duration of the LLM turn so any
    # ``agent.task(...)`` (typically via the ``tasks(action="start")``
    # tool) surfaces ``ag2.task.*`` envelopes to the hub and triggers
    # ``record_observation`` on capability-tagged terminal events.
    mirror = TaskMirror(
        hub_client=client._hub_client,
        owner_id=client.agent_id,
        session_id=metadata.session_id,
    )
    sub_ids = mirror.attach(stream)
    try:
        reply = await client.agent.ask(
            current_input,
            stream=stream,
            dependencies=dependencies,
        )
    finally:
        mirror.detach(stream, sub_ids)

    # Adapter encodes the round-end envelope.
    # For example, Workflow returns EV_PACKET.
    # Default implementations returns EV_TEXT(body) or None.
    state = client._hub_client.adapter_state(metadata.session_id)
    events = list(await stream.history.get_events())
    out_envelope = adapter.build_round_envelope(
        metadata=metadata,
        sender_id=client.agent_id,
        reply=reply,
        events=events,
        state=state,
        hub=client._hub,
    )
    if out_envelope is None:
        return
    out_envelope.causation_id = envelope.envelope_id
    await client.send_envelope(out_envelope)


async def default_handler(envelope: Envelope, client: "AgentClient") -> None:
    """Route an inbound envelope to its handler.

    Override via :meth:`AgentClient.on_envelope` — the default
    delegates to the per-event helpers above which can be composed in
    custom handlers.

    Substantive routing is delegated to the session's adapter via
    ``adapter.extract_turn_input`` (returns empty for envelope types
    the adapter doesn't act on, ending the handler chain).
    """
    event_type = envelope.event_type
    if event_type == EV_SESSION_INVITE:
        await _auto_ack_invite(envelope, client)
        return
    if event_type.startswith("ag2.session.") or event_type.startswith("ag2.task."):
        # Bookkeeping: state changes are visible via Session.info();
        # task events are mirrored separately by TaskMirror.
        return
    await _process_substantive(envelope, client)
    # Other ag2.session.* events (OPENED/CLOSED/EXPIRED) and ag2.task.*
    # events: no LLM action. Session state changes are reflected in
    # the next ``Session.info()`` call; task events are mirrored by
    # ``TaskMirror`` separately.
