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
from base64 import b64encode
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from ag_ui.core import (
    ActivitySnapshotEvent,
    ReasoningEncryptedValueEvent,
    RunAgentInput,
    TextMessageChunkEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from ag2.ag_ui.capabilities import served_capabilities
from ag2.ag_ui.interrupts import (
    DEFAULT_RETENTION,
    ClientInterrupter,
    Retention,
    ServedTurn,
    ServedTurns,
    TurnOutput,
    drive_run,
    serve_exchange,
    utc_now,
)
from ag2.ag_ui.run_input import read_run_input
from ag2.ag_ui.stream import (
    AGStreamInput,
    client_context_prompt,
    downgrade_tool_result,
    map_agui_messages_to_events,
    map_task_event_to_ag_ui,
    map_tool_result_to_ag_ui,
    provider_of,
)
from ag2.config.gemini.events import GeminiToolCallEvent
from ag2.events import TextInput, ToolCallEvent, ToolResultEvent

from .._types import JsonObject, ServerToClientMessage
from ..dispatch import A2UIMessageFrame, A2UIProseFrame, TaskEvent
from ..incoming import iter_incoming_prompts, parse_incoming_interactions
from ..request import A2UIServerRequest

if TYPE_CHECKING:
    from ag2.agent import Agent

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

    A turn that asks the human a question is held until a later run on the same
    thread answers it, exactly as ``ag2.ag_ui.AGUIStream`` does.
    :class:`~ag2.a2ui.A2UIServer` calls :meth:`aclose` on shutdown.

    Args:
        path: The route path, serving POST runs and GET capabilities; GET
            ``{path}/capabilities`` serves them too.
        retention: How long an unanswered question is held, and how many at once.
        now: The clock deadlines are read off. For tests.
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
        capabilities = functools.partial(_capabilities, core)
        # GET on the run route answers a client asking what this agent can do,
        # exactly as the other AG-UI transport does, and so does the sub-path
        # other AG-UI integrations serve it on: a client decides up front
        # whether to offer the interrupt UI.
        return [
            Route(self._path, endpoint, methods=["GET", "POST"]),
            Route(f"{self._path.rstrip('/')}/capabilities", capabilities, methods=["GET"]),
        ]

    async def aclose(self) -> None:
        """Cancel every turn this transport is still running."""
        await self._turns.release_all()


async def _capabilities(core: "_A2UITurnCore", request: Request) -> JSONResponse:
    # The turn core never sees the run's `tools`, and nothing here sends a
    # state snapshot.
    capabilities = served_capabilities(core.agent, client_tools=False, state_snapshots=False)
    return JSONResponse(capabilities.model_dump(by_alias=True, exclude_none=True))


async def _endpoint(turns: ServedTurns, core: "_A2UITurnCore", request: Request) -> Response:
    if request.method == "GET":
        return await _capabilities(core, request)

    try:
        incoming = read_run_input(await request.body())
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
        provider=provider_of(core.agent.config),
        config=core.agent.config,
    )
    if shared := client_context_prompt(incoming.context or []):
        prompt.append(shared)
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
    # The exchange itself is ``ag2.ag_ui``'s, shared so the two AG-UI transports
    # cannot drift apart on a wire a client reads the same way.
    return serve_exchange(turns, incoming, encoder, functools.partial(_start_turn, turns, core, incoming))


def _start_turn(
    turns: ServedTurns,
    core: "_A2UITurnCore",
    incoming: RunAgentInput,
    output: TurnOutput,
) -> ServedTurn:
    # Tracked on the registry, so a question it raises can be answered by a
    # later exchange.
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

    Emits any prose, then one ``ActivitySnapshot`` of the A2UI operations, then
    ``RunFinished`` — or ``RunError``, since the run has already started 200 OK
    on the wire and a failure is reported rather than raised. Both terminating
    events carry the run's token usage.
    """
    await drive_run(output, _serve_turn(core, incoming, output, interrupter))


async def _serve_turn(
    core: "_A2UITurnCore",
    incoming: RunAgentInput,
    output: TurnOutput,
    interrupter: ClientInterrupter | None,
) -> None:
    # Inside the run: input that cannot be mapped fails a run already started.
    request = _request_from_agui(core, incoming)
    text_message_id = uuid4().hex
    operations: list[ServerToClientMessage] = []

    # Usage is metered live rather than read back from history: the turn core
    # owns the stream and this transport never sees it.
    async for frame in core.run_turn(
        request,
        usage_records=output.usage,
        interrupter=interrupter,
        on_task=functools.partial(_report_delegation, output),
        on_tool_call=functools.partial(_report_tool_call, output),
        on_tool_result=functools.partial(_report_tool_result, output, core.agent, incoming.protocol_version is None),
    ):
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


async def _report_delegation(output: TurnOutput, event: TaskEvent) -> None:
    await output.send(map_task_event_to_ag_ui(event))


async def _report_tool_call(output: TurnOutput, event: ToolCallEvent) -> None:
    timestamp = int(utc_now().timestamp() * 1000)
    if isinstance(event, GeminiToolCallEvent) and event.thought_signature is not None:
        await output.send(
            ReasoningEncryptedValueEvent(
                subtype="tool-call",
                entity_id=event.id,
                encrypted_value=b64encode(event.thought_signature).decode(),
                timestamp=timestamp,
            )
        )
    await output.send(
        ToolCallStartEvent(
            tool_call_id=event.id,
            tool_call_name=event.name,
            timestamp=timestamp,
        )
    )
    await output.send(ToolCallArgsEvent(tool_call_id=event.id, delta=event.arguments, timestamp=timestamp))
    await output.send(ToolCallEndEvent(tool_call_id=event.id, timestamp=timestamp))


async def _report_tool_result(output: TurnOutput, agent: "Agent", predates_parts: bool, event: ToolResultEvent) -> None:
    content = map_tool_result_to_ag_ui(event.result, agent._serializer)
    await output.send(
        ToolCallResultEvent(
            tool_call_id=event.parent_id,
            content=downgrade_tool_result(content) if predates_parts else content,
            message_id=uuid4().hex,
            timestamp=int(utc_now().timestamp() * 1000),
            role="tool",
        )
    )


__all__ = ("AgUiTransport",)
