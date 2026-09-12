# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""AG-UI transport: serve a turn over AG-UI so CopilotKit's
``@copilotkit/a2ui-renderer`` renders the agent's A2UI output.

Reuses the A2UI turn core (``_A2UITurnCore.run_turn``) and the AG-UI history
mapping from ``ag2.ag_ui`` **without modifying** either. The agent's
validated A2UI messages are collected per turn and emitted as a single AG-UI
``ActivitySnapshotEvent`` whose ``content`` carries them under the
``a2ui_operations`` key — the exact wire contract the renderer consumes
(verified against CopilotKit
``packages/react-core/src/v2/a2ui/A2UIMessageRenderer.tsx``).

Because the prose comes from the turn core's final, A2UI-stripped message (not
live model chunks), the raw ``<a2ui-json>`` block never leaks into the streamed
text. Importing this module requires Starlette and ``ag2[ag-ui]``.
"""

import functools
import logging
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from ag_ui.core import (
    ActivitySnapshotEvent,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    TextMessageChunkEvent,
)
from ag_ui.encoder import EventEncoder
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from ag2.ag_ui.interrupts import (
    DEFAULT_RETENTION,
    ClientInterrupter,
    Retention,
    ServedTurn,
    ServedTurns,
    TurnOutput,
    interrupt_capabilities,
    serve_exchange,
    success_outcome,
    timestamp_ms,
    utc_now,
)
from ag2.ag_ui.stream import AGStreamInput, map_agui_messages_to_events, map_usage_events_to_ag_ui
from ag2.events import TextInput, UsageEvent

from .._types import JsonObject, ServerToClientMessage
from ..dispatch import A2UIMessageFrame, A2UIProseFrame
from ..incoming import iter_incoming_prompts, parse_incoming_interactions
from ..request import A2UIServerRequest

if TYPE_CHECKING:
    from ..dispatch import _A2UITurnCore

logger = logging.getLogger(__name__)

# Wire contract consumed by ``@copilotkit/a2ui-renderer`` (``createA2UIMessageRenderer``):
# an AG-UI activity message with this ``activity_type`` whose ``content`` carries
# the A2UI operations under this key. Both strings are matched verbatim by the
# renderer — see CopilotKit ``react-core/src/v2/a2ui/A2UIMessageRenderer.tsx``
# (``activityType: "a2ui-surface"``, ``A2UI_OPERATIONS_KEY = "a2ui_operations"``).
_A2UI_ACTIVITY_TYPE = "a2ui-surface"
_A2UI_OPERATIONS_KEY = "a2ui_operations"


class AgUiTransport:
    """Serve the turn over AG-UI for CopilotKit's A2UI renderer.

    A turn that asks the human a question is held here, suspended where it
    stopped, until a later run on the same thread answers it — the same
    behaviour, on the same wire, as ``ag2.ag_ui.AGUIStream``, over the same
    shared registry. :meth:`aclose` cancels whatever is still held; the
    :class:`~ag2.a2ui.A2UIServer` calls it on shutdown.

    Args:
        path: The POST route path. Defaults to ``"/"``.
        retention: How long an unanswered question is held, and how many at
            once. The time bound is what a client is shown as the interrupt's
            deadline.
        now: The clock deadlines are read off. For tests; see
            :class:`~ag2.ag_ui.AGUIStream`.
    """

    __slots__ = ("_path", "_turns")

    def __init__(
        self,
        *,
        path: str = "/",
        retention: Retention = DEFAULT_RETENTION,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self._path = path
        self._turns = ServedTurns(retention=retention, now=now)

    def routes(self, core: "_A2UITurnCore") -> list[Route]:
        endpoint = functools.partial(_endpoint, self._turns, core)
        # GET on the same route answers a client asking what this agent can do,
        # exactly as the other AG-UI transport does: the two are meant to be
        # indistinguishable, and a client decides up front whether to offer the
        # interrupt UI.
        return [Route(self._path, endpoint, methods=["GET", "POST"])]

    async def aclose(self) -> None:
        """Cancel every turn this transport is still running."""
        await self._turns.release_all()


async def _endpoint(turns: ServedTurns, core: "_A2UITurnCore", request: Request) -> Response:
    if request.method == "GET":
        capabilities = interrupt_capabilities(core.agent.name)
        return JSONResponse(capabilities.model_dump(by_alias=True, exclude_none=True))

    try:
        body = await request.body()
        incoming = RunAgentInput.model_validate_json(body)
    except Exception:  # noqa: BLE001 - bad/short body or disconnect → 400, not 500
        return Response('{"error": "invalid AG-UI RunAgentInput body"}', status_code=400, media_type="application/json")

    encoder = EventEncoder(accept=request.headers.get("accept", ""))
    return StreamingResponse(
        _dispatch(turns, core, incoming, encoder=encoder),
        media_type=encoder.get_content_type(),
    )


def _click_envelopes(forwarded_props: object) -> list[JsonObject]:
    """Extract A2UI client→server ``action`` envelopes from a run's ``forwardedProps``.

    CopilotKit's ``@copilotkit/a2ui-renderer`` relays a button click by setting
    ``forwardedProps.a2uiAction = {"userAction": {name, surfaceId, sourceComponentId?,
    context?, timestamp?, dataContextPath?}}`` and re-running the agent (verified
    against CopilotKit ``react-core`` ``A2UIMessageRenderer`` and their server
    examples). Map that to the ``{"action": {...}}`` envelope the A2UI incoming
    pipeline already parses; returns ``[]`` when no usable click is present.
    """
    if not isinstance(forwarded_props, dict):
        return []
    a2ui_action = forwarded_props.get("a2uiAction")
    if not isinstance(a2ui_action, dict):
        return []
    user_action = a2ui_action.get("userAction")
    if not isinstance(user_action, dict) or not user_action.get("name"):
        return []
    context = user_action.get("context")
    return [
        {
            "action": {
                "name": user_action["name"],
                "surfaceId": user_action.get("surfaceId", ""),
                "sourceComponentId": user_action.get("sourceComponentId", ""),
                "timestamp": user_action.get("timestamp", ""),
                "context": context if isinstance(context, dict) else {},
            },
        },
    ]


def _request_from_agui(core: "_A2UITurnCore", incoming: RunAgentInput) -> A2UIServerRequest:
    """Map an AG-UI ``RunAgentInput`` to a transport-neutral A2UI turn.

    Reuses ``ag2.ag_ui``'s history mapping (system/developer prompt,
    prior turns, trailing user turn) unchanged, then folds in any button click:
    CopilotKit relays a click as ``forwardedProps.a2uiAction`` and re-runs the
    agent (no new chat message), so the click is rewritten into the current turn
    and surfaced as a client interaction — mirroring the REST transport's
    handling of inbound ``a2ui`` envelopes.
    """
    variables = incoming.state if isinstance(incoming.state, dict) else {}
    prompt, history, current_inputs = map_agui_messages_to_events(
        AGStreamInput(incoming=incoming, variables=variables),
    )
    envelopes = _click_envelopes(incoming.forwarded_props)
    current_inputs.extend(TextInput(p) for p in iter_incoming_prompts(envelopes, core.runtime.get_action))
    return A2UIServerRequest(
        current_inputs=current_inputs,
        history=history,
        prompt=prompt,
        variables=variables,
        client_interactions=parse_incoming_interactions(envelopes),
    )


def _dispatch(
    turns: ServedTurns,
    core: "_A2UITurnCore",
    incoming: RunAgentInput,
    *,
    encoder: EventEncoder,
) -> AsyncIterator[str]:
    """Drive one exchange over a turn of this transport's, and yield its events.

    The exchange itself — begin or resume, carry the run to its terminating
    event, leave a held turn running — is ``ag2.ag_ui``'s, shared so that the
    two AG-UI transports cannot drift into behaving differently on a wire a
    client reads the same way.
    """
    return serve_exchange(turns, incoming, encoder, functools.partial(_start_turn, turns, core, incoming))


def _start_turn(
    turns: ServedTurns,
    core: "_A2UITurnCore",
    incoming: RunAgentInput,
    output: TurnOutput,
) -> ServedTurn:
    """Launch a turn of this transport's, writing to ``output``.

    The turn is tracked on the registry, so a question it raises can be answered
    by a later exchange.
    """
    turn = ServedTurn(output)
    # This transport's turn core carries a plain agent, so the only hook
    # there can be is the one the agent was constructed with. With none,
    # the question goes to the client rather than killing the turn.
    interrupter = None if core.agent._hitl_hook is not None else ClientInterrupter(turn, turns)
    turns.track(turn, turn.start(_run_turn(core, incoming, output, interrupter)))
    return turn


async def _run_turn(
    core: "_A2UITurnCore",
    incoming: RunAgentInput,
    output: TurnOutput,
    interrupter: ClientInterrupter | None,
) -> None:
    """Run one turn, writing its AG-UI events to ``output``.

    Emits (``TextMessageChunk`` if there is prose) → (one ``ActivitySnapshot``
    carrying all A2UI operations, if any) → ``RunFinished``; the exchange emits
    ``RunStarted`` for itself, since a resumed turn outlives the run that
    started it. A mid-turn failure surfaces as a ``RunError`` event (the run has
    already started 200 OK on the wire) rather than as a raise: this transport
    reports a failed turn and ends it.

    Both terminating events carry the turn's token usage, so what a client can
    report does not depend on which AG-UI endpoint it connected to. The records
    are collected live rather than read back from history, because the turn core
    owns the stream and this transport never sees it — and collecting live is
    also what leaves the failure path with something to report.
    """
    request = _request_from_agui(core, incoming)
    text_message_id = uuid4().hex
    operations: list[ServerToClientMessage] = []
    usage_records: list[UsageEvent] = []

    try:
        async for frame in core.run_turn(request, usage_records=usage_records, interrupter=interrupter):
            if isinstance(frame, A2UIProseFrame):
                if frame.text:
                    await output.send(
                        TextMessageChunkEvent(message_id=text_message_id, role="assistant", delta=frame.text),
                    )
            elif isinstance(frame, A2UIMessageFrame):
                operations.append(frame.message)

        if operations:
            # One snapshot per turn (replace=True default): the renderer rebuilds
            # the surface(s) from the full operations list.
            await output.send(
                ActivitySnapshotEvent(
                    message_id=uuid4().hex,
                    activity_type=_A2UI_ACTIVITY_TYPE,
                    content={_A2UI_OPERATIONS_KEY: operations},
                ),
            )
    except Exception as e:  # noqa: BLE001 - report as a RunError frame, don't tear down the stream silently
        logger.exception("A2UI AG-UI turn failed")
        await output.send(
            RunErrorEvent(message=repr(e), timestamp=timestamp_ms(), usage=map_usage_events_to_ag_ui(usage_records))
        )
    else:
        await output.send(
            RunFinishedEvent(
                thread_id=output.thread_id,
                run_id=output.run_id,
                timestamp=timestamp_ms(),
                usage=map_usage_events_to_ag_ui(usage_records),
                outcome=success_outcome(),
            )
        )
    finally:
        # The exchange reading this turn ends on its terminating event, but the
        # channel is the turn's: closed here, on every path including
        # cancellation while held.
        await output.aclose()


__all__ = ("AgUiTransport",)
