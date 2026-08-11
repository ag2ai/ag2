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
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from uuid import uuid4

from ag_ui.core import (
    ActivitySnapshotEvent,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageChunkEvent,
)
from ag_ui.encoder import EventEncoder
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from ag2.ag_ui.stream import AGStreamInput, map_agui_messages_to_events
from ag2.events import TextInput

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

    Args:
        path: The POST route path. Defaults to ``"/"``.
    """

    __slots__ = ("_path",)

    def __init__(self, *, path: str = "/") -> None:
        self._path = path

    def routes(self, core: "_A2UITurnCore") -> list[Route]:
        endpoint = functools.partial(_endpoint, core)
        return [Route(self._path, endpoint, methods=["POST"])]


async def _endpoint(core: "_A2UITurnCore", request: Request) -> Response:
    try:
        body = await request.body()
        incoming = RunAgentInput.model_validate_json(body)
    except Exception:  # noqa: BLE001 - bad/short body or disconnect → 400, not 500
        return Response('{"error": "invalid AG-UI RunAgentInput body"}', status_code=400, media_type="application/json")

    encoder = EventEncoder(accept=request.headers.get("accept", ""))
    return StreamingResponse(_dispatch(core, incoming, encoder=encoder), media_type=encoder.get_content_type())


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


async def _dispatch(core: "_A2UITurnCore", incoming: RunAgentInput, *, encoder: EventEncoder) -> AsyncIterator[str]:
    """Run one turn and yield encoded AG-UI events.

    Emits ``RunStarted`` → (``TextMessageChunk`` if there is prose) → (one
    ``ActivitySnapshot`` carrying all A2UI operations, if any) → ``RunFinished``.
    A mid-turn failure surfaces as a ``RunError`` event (the run has already
    started 200 OK on the wire).
    """
    request = _request_from_agui(core, incoming)
    text_message_id = uuid4().hex
    operations: list[ServerToClientMessage] = []

    yield encoder.encode(RunStartedEvent(thread_id=incoming.thread_id, run_id=incoming.run_id))
    try:
        async for frame in core.run_turn(request):
            if isinstance(frame, A2UIProseFrame):
                if frame.text:
                    yield encoder.encode(
                        TextMessageChunkEvent(message_id=text_message_id, role="assistant", delta=frame.text),
                    )
            elif isinstance(frame, A2UIMessageFrame):
                operations.append(frame.message)

        if operations:
            # One snapshot per turn (replace=True default): the renderer rebuilds
            # the surface(s) from the full operations list.
            yield encoder.encode(
                ActivitySnapshotEvent(
                    message_id=uuid4().hex,
                    activity_type=_A2UI_ACTIVITY_TYPE,
                    content={_A2UI_OPERATIONS_KEY: operations},
                ),
            )
    except Exception as e:  # noqa: BLE001 - report as a RunError frame, don't tear down the stream silently
        logger.exception("A2UI AG-UI turn failed")
        yield encoder.encode(RunErrorEvent(message=repr(e)))
        return

    yield encoder.encode(RunFinishedEvent(thread_id=incoming.thread_id, run_id=incoming.run_id))


__all__ = ("AgUiTransport",)
