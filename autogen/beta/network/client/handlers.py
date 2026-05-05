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
    EV_HANDOFF,
    EV_SESSION_INVITE,
    EV_SESSION_INVITE_ACK,
    EV_TEXT,
    Envelope,
)
from ..policies import AGENT_CLIENT_DEP, HUB_DEP, SESSION_DEP
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
    """Build the ``context.dependencies`` dict for the LLM turn."""
    return {
        SESSION_DEP: session,
        AGENT_CLIENT_DEP: client,
        HUB_DEP: client._hub,
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


def _extract_turn_text(envelope: Envelope) -> str:
    """Pull the user-message body out of a substantive envelope.

    ``EV_TEXT`` carries the text in ``event_data['text']``.
    ``EV_HANDOFF`` carries the reason in ``event_data['reason']`` and
    the tool name in ``event_data['tool']``; we synthesise a short
    handoff prompt so the next speaker's LLM has context.
    """
    if envelope.event_type == EV_TEXT:
        text = envelope.event_data.get("text", "")
        return text if isinstance(text, str) else ""
    if envelope.event_type == EV_HANDOFF:
        tool = envelope.event_data.get("tool", "handoff")
        reason = envelope.event_data.get("reason", "")
        if reason:
            return f"[Handed off via {tool}] {reason}"
        return f"[Handed off via {tool}]"
    return ""


async def _process_text(envelope: Envelope, client: "AgentClient") -> None:
    """Run the agent's LLM on the inbound substantive envelope and
    send its reply.

    Handles ``EV_TEXT`` and ``EV_HANDOFF``. Only engages the LLM when
    the adapter would accept a reply from this agent right now — for
    consulting, that means we're the respondent and haven't replied
    yet; for workflow, that we're ``expected_next_speaker``. After
    each turn the adapter rotates so this same handler firing for a
    different participant's notify is a no-op via the probe.
    """
    metadata = await client._hub_client.get_session(envelope.session_id)
    if metadata.is_terminal() or metadata.state != SessionState.ACTIVE:
        return

    # "Can we respond now?" — ask the hub via the public probe surface
    # so the handler doesn't need to reach into adapter internals.
    if not client._hub_client.can_send(envelope.session_id, client.agent_id):
        return  # not our turn / session closing — don't engage LLM

    session = Session(metadata=metadata, client=client)
    view = resolve_view_policy(client, metadata)

    history_envelopes = await read_wal_until(client, envelope)
    projection: list[BaseEvent] = await view.project(
        history_envelopes,
        participant_id=client.agent_id,
        session=metadata,
    )

    current_text = _extract_turn_text(envelope)
    if not current_text:
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
            current_text,
            stream=stream,
            dependencies=dependencies,
        )
    finally:
        mirror.detach(stream, sub_ids)
    body = reply.body
    if body:
        await session.send(body, causation_id=envelope.envelope_id)


async def default_handler(envelope: Envelope, client: "AgentClient") -> None:
    """Route an inbound envelope to its handler.

    Override via :meth:`AgentClient.on_envelope` — the default
    delegates to the per-event helpers above which can be composed in
    custom handlers.
    """
    event_type = envelope.event_type
    if event_type == EV_SESSION_INVITE:
        await _auto_ack_invite(envelope, client)
        return
    if event_type in (EV_TEXT, EV_HANDOFF):
        await _process_text(envelope, client)
        return
    # Other ag2.session.* events (OPENED/CLOSED/EXPIRED) and ag2.task.*
    # events: no LLM action. Session state changes are reflected in
    # the next ``Session.info()`` call; task events are mirrored by
    # ``TaskMirror`` separately.
