# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Default notify handler.

Routes inbound envelopes to the right action:

* ``EV_CHANNEL_INVITE`` → auto-ack (post ``EV_CHANNEL_INVITE_ACK``).
* ``EV_CHANNEL_*`` other → no-op (state is bookkeeping; the agent
  doesn't need to react).
* ``EV_TEXT`` → read WAL, project view, run ``agent.ask`` with the
  projection pre-populated as stream history, send any non-empty
  reply via ``Channel.send``.
* ``ag2.task.*`` → no-op (mirrored separately by ``TaskMirror``).

The handler is decomposed into small public hooks
(``read_wal_until``, ``resolve_view_policy``, ``stamp_dependencies``)
so user-supplied overrides can replace only the parts they care about.
"""

import contextlib
import logging
from typing import TYPE_CHECKING

from ag2.events import BaseEvent, ModelMessage, ModelRequest, TextInput
from ag2.stream import MemoryStream

from ..channel import ChannelMetadata, ChannelState
from ..envelope import (
    EV_CHANNEL_INVITE,
    EV_CHANNEL_INVITE_ACK,
    Envelope,
)
from ..policies import AGENT_CLIENT_DEP, CHANNEL_DEP, CHANNEL_STATE_DEP, HUB_DEP
from ..task_mirror import TaskMirror
from ..views.base import NameResolver, ViewPolicy
from .channel import Channel

if TYPE_CHECKING:
    from ..adapters.base import ChannelAdapter
    from .agent_client import AgentClient

__all__ = (
    "default_handler",
    "read_wal_until",
    "resolve_view_policy",
    "stamp_dependencies",
)


logger = logging.getLogger(__name__)


def _is_task_event(event_type: str) -> bool:
    return event_type.startswith("ag2.task.")


async def _render_current_input(
    view: ViewPolicy,
    envelope: Envelope,
    adapter: "ChannelAdapter",
    participant_id: str,
    metadata: ChannelMetadata,
    name_for: NameResolver,
) -> str | None:
    """Render the current-turn envelope through the view.

    Calls ``view.project([envelope])`` so named views apply consistent
    values between current envelope and history. Falls back to
    ``adapter.extract_turn_input`` when the view returns nothing (e.g.
    rich input types like images or documents that ``render_envelope``
    cannot represent as a string).
    """
    projected = await view.project(
        [envelope],
        participant_id=participant_id,
        channel=metadata,
        render_envelope=adapter.render_envelope,
        name_for=name_for,
    )
    if projected:
        event = projected[0]
        if isinstance(event, ModelRequest):
            for part in event.parts:
                if isinstance(part, TextInput):
                    return part.content
        if isinstance(event, ModelMessage):
            return event.content
    return adapter.extract_turn_input(envelope)


async def read_wal_until(client: "AgentClient", envelope: Envelope) -> list[Envelope]:
    """Return WAL slice up to (but excluding) ``envelope``.

    The current envelope is the prompt for this turn — it is fed to
    ``agent.ask`` separately rather than mixed into the projected
    history.
    """
    wal = await client._hub_client.read_wal(envelope.channel_id)
    history: list[Envelope] = []
    for env in wal:
        if env.envelope_id == envelope.envelope_id:
            break
        history.append(env)
    return history


def resolve_view_policy(
    client: "AgentClient",
    metadata: ChannelMetadata,
) -> ViewPolicy:
    """Return the adapter's default view policy for this participant.

    Resolves the adapter from the metadata in hand (no I/O) so the same
    path works whether the hub is in-process or remote.
    """
    return client._hub_client.adapter_for_metadata(metadata).default_view_policy(metadata, client.agent_id)


def stamp_dependencies(
    client: "AgentClient",
    channel: Channel,
    state: object | None,
) -> dict[object, object]:
    """Build the ``context.dependencies`` dict for the LLM turn.

    ``CHANNEL_STATE_DEP`` resolves to the adapter's current State
    object (``WorkflowState`` / ``DiscussionState`` / ...), passed in by
    the caller (folded once per turn). Tools that need channel-scoped
    state (e.g. ``context_vars`` on a workflow channel) inject it via
    ``ChannelStateInject``. ``HUB_DEP`` resolves to the ``HubClient``
    seam — the in-process-or-remote control surface — so an injected
    dependency works in both deployment modes.
    """
    return {
        CHANNEL_DEP: channel,
        AGENT_CLIENT_DEP: client,
        HUB_DEP: client._hub_client,
        CHANNEL_STATE_DEP: state,
    }


async def _auto_ack_invite(envelope: Envelope, client: "AgentClient") -> None:
    """Default behaviour: ack any invite addressed to us.

    Policy-based rejection (``EV_CHANNEL_INVITE_REJECT`` on access
    denial / capacity) is the override path — replace this handler in a
    custom callback wired via ``AgentClient.on_envelope``.
    """
    ack = Envelope(
        channel_id=envelope.channel_id,
        sender_id=client.agent_id,
        audience=None,
        event_type=EV_CHANNEL_INVITE_ACK,
        event_data={"channel_id": envelope.channel_id},
        causation_id=envelope.envelope_id,
    )
    # An ack failure shouldn't crash the agent — the hub will time
    # out and close the channel via ``invite_ack_timeout``.
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

    Per-turn LLM tools come from ``adapter.tools_for(client, metadata,
    state, participant_id)``. Identity-level tools (``peers`` /
    ``channels`` / ``tasks`` / ``context`` / ``delegate``) live on
    ``agent.tools`` via ``NetworkPlugin``; adapter-scoped tools (e.g.
    ``say`` for consulting / conversation / discussion) merge in
    per-call via ``agent.ask(tools=...)``.

    The full turn-processing path — channel resolution, view projection,
    adapter input extraction, ``agent.ask``, round-envelope construction,
    outbound send — is wrapped in one try/except. On exception the
    handler routes the failure through ``HubClient.report_turn_failure``
    (which fans out to every :class:`HubListener`, including the built-in
    audit log) and returns cleanly. The channel stays alive and
    subsequent envelopes flow. No reply envelope is posted on failure —
    the framework does not invent content on the agent's behalf.
    """
    out_envelope = None
    try:
        metadata = await client._hub_client.get_channel(envelope.channel_id)
        if metadata.is_terminal() or metadata.state != ChannelState.ACTIVE:
            return

        # At-least-once redelivery guard. If this agent already posted a
        # reply caused by this envelope, the turn is already done —
        # re-running the LLM would duplicate the post. The reply carries
        # ``causation_id = envelope.envelope_id`` (stamped below), so a
        # prior reply is found under this agent's (channel, causation)
        # key. Checked before the turn-ownership probe so a redelivery is
        # a no-op regardless of whose turn it now is.
        prior = await client._hub_client.find_envelope_by_causation(
            metadata.channel_id,
            sender_id=client.agent_id,
            causation_id=envelope.envelope_id,
        )
        if prior is not None:
            return

        # "Can we respond now?" — ask the hub via the public probe surface
        # so the handler doesn't need to reach into adapter internals.
        if not await client._hub_client.can_send(envelope.channel_id, client.agent_id):
            return  # not our turn / channel closing — don't engage LLM

        adapter = client._hub_client.adapter_for_metadata(metadata)
        channel = Channel(metadata=metadata, client=client)
        view = resolve_view_policy(client, metadata)

        history_envelopes = await read_wal_until(client, envelope)
        projection: list[BaseEvent] = await view.project(
            history_envelopes,
            participant_id=client.agent_id,
            channel=metadata,
            render_envelope=adapter.render_envelope,
            name_for=client._hub_client.name_for,
        )

        current_input = await _render_current_input(
            view, envelope, adapter, client.agent_id, metadata, client._hub_client.name_for
        )
        if not current_input:
            return

        # Pre-populate a fresh stream's history with the projection so the
        # agent's middleware sees the prior conversation context. The
        # current turn's user message is passed via ``msg`` and gets
        # appended to history naturally by ``Agent.ask``.
        stream = MemoryStream()
        if projection:
            await stream.history.storage.set_history(stream.id, projection)

        # Fold the adapter state once for this turn (re-read after the
        # LLM turn below, since tools may advance it). Cross-process this
        # folds from the WAL; in-process it reads the hub's cache.
        state = await client._hub_client.adapter_state(metadata.channel_id)
        dependencies = stamp_dependencies(client, channel, state)

        # Adapter-scoped LLM tools for this turn (e.g. ``say`` for
        # consulting / discussion). Resolution is cached on the adapter
        # so the schema build cost is paid once per (adapter, client)
        # — see ``ChannelAdapter.tools_for`` default implementation.
        adapter_tools: list = []
        try:
            adapter_tools = list(adapter.tools_for(client, metadata, state, client.agent_id))
        except Exception:
            logger.exception(
                "adapter.tools_for raised: channel=%s adapter=%s",
                envelope.channel_id,
                type(adapter).__name__,
            )

        # Attach the TaskMirror for the duration of the LLM turn so any
        # ``agent.task(...)`` (typically via the ``tasks(action="start")``
        # tool) surfaces ``ag2.task.*`` envelopes to the hub and triggers
        # ``record_observation`` on capability-tagged terminal events.
        mirror = TaskMirror(
            hub_client=client._hub_client,
            owner_id=client.agent_id,
            channel_id=metadata.channel_id,
        )
        sub_ids = mirror.attach(stream)
        try:
            reply = await client.agent.ask(
                current_input,
                stream=stream,
                dependencies=dependencies,
                tools=adapter_tools,
            )

            # Adapter encodes the round-end envelope.
            # For example, Workflow returns EV_PACKET.
            # Default implementations returns EV_TEXT(body) or None.
            # State may have advanced inside the LLM turn (e.g. an
            # ``EV_CONTEXT_SET`` fold) — re-read here.
            state = await client._hub_client.adapter_state(metadata.channel_id)
            events = list(await stream.history.get_events())
            out_envelope = adapter.build_round_envelope(
                metadata=metadata,
                sender_id=client.agent_id,
                reply=reply,
                events=events,
                state=state,
                hub=client._hub_client,
            )
        finally:
            mirror.detach(stream, sub_ids)

        if out_envelope is None:
            return
        out_envelope.causation_id = envelope.envelope_id
        await client.send_envelope(out_envelope)
    except Exception as exc:
        logger.exception(
            "notify handler raised: channel=%s agent=%s envelope=%s",
            envelope.channel_id,
            client.agent_id,
            envelope.envelope_id,
        )
        await _report_turn_failure(client, envelope, exc)
        return


async def _report_turn_failure(
    client: "AgentClient",
    envelope: Envelope,
    exc: BaseException,
) -> None:
    """Route an exception raised inside the notify handler through the hub.

    The hub stays the single owner of audit and listener fan-out — the
    handler calls the public ``HubClient.report_turn_failure`` surface
    rather than touching hub internals. ``AuditLog`` (built-in listener)
    records the failure; tenant listeners react however they choose.
    """
    hub_client = getattr(client, "_hub_client", None)
    if hub_client is None:
        return
    with contextlib.suppress(Exception):
        await hub_client.report_turn_failure(
            channel_id=envelope.channel_id,
            agent_id=client.agent_id,
            envelope_id=envelope.envelope_id,
            exc=exc,
        )


async def default_handler(envelope: Envelope, client: "AgentClient") -> None:
    """Route an inbound envelope to its handler.

    Override via :meth:`AgentClient.on_envelope` — the default
    delegates to the per-event helpers above which can be composed in
    custom handlers.

    Substantive routing is delegated to the channel's adapter via
    ``adapter.extract_turn_input`` (returns empty for envelope types
    the adapter doesn't act on, ending the handler chain).
    """
    event_type = envelope.event_type
    if event_type == EV_CHANNEL_INVITE:
        await _auto_ack_invite(envelope, client)
        return
    if event_type.startswith("ag2.channel.") or event_type.startswith("ag2.task."):
        # Bookkeeping: state changes are visible via Channel.info();
        # task events are mirrored separately by TaskMirror.
        return
    await _process_substantive(envelope, client)
    # Other ag2.channel.* events (OPENED/CLOSED/EXPIRED) and ag2.task.*
    # events: no LLM action. Channel state changes are reflected in
    # the next ``Channel.info()`` call; task events are mirrored by
    # ``TaskMirror`` separately.
