# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Serve a plain :class:`~autogen.beta.Agent` over AG-UI so CopilotKit's
``@copilotkit/a2ui-renderer`` renders the agent's A2UI output.

This is a standalone transport: it reuses the A2UI turn core
(:func:`~autogen.beta.a2ui.rest.dispatch.stream_turn`) and the AG-UI history
mapping from ``autogen.beta.ag_ui`` **without modifying** either. The agent's
validated A2UI messages are collected per turn and emitted as a single AG-UI
``ActivitySnapshotEvent`` whose ``content`` carries them under the
``a2ui_operations`` key — the exact wire contract the renderer consumes
(verified against CopilotKit ``packages/react-core/src/v2/a2ui/A2UIMessageRenderer.tsx``).

Because the prose comes from ``stream_turn``'s final, A2UI-stripped message (not
live model chunks), the raw ``<a2ui-json>`` block never leaks into the streamed
text.
"""

import logging
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
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

from autogen.beta.ag_ui.stream import AGStreamInput, map_agui_messages_to_events
from autogen.beta.agent import Agent
from autogen.beta.events import TextInput

from .._runtime import _A2UIRuntime
from .._types import A2UIVersion, JsonSchema
from ..incoming import iter_incoming_prompts, parse_incoming_interactions
from ..rest.dispatch import A2UIMessageFrame, A2UIProseFrame, stream_turn
from ..rest.request import A2UIServerRequest

if TYPE_CHECKING:
    from starlette.applications import Starlette

logger = logging.getLogger(__name__)

# Wire contract consumed by ``@copilotkit/a2ui-renderer`` (``createA2UIMessageRenderer``):
# an AG-UI activity message with this ``activity_type`` whose ``content`` carries
# the A2UI operations under this key. Both strings are matched verbatim by the
# renderer — see CopilotKit ``react-core/src/v2/a2ui/A2UIMessageRenderer.tsx``
# (``activityType: "a2ui-surface"``, ``A2UI_OPERATIONS_KEY = "a2ui_operations"``).
_A2UI_ACTIVITY_TYPE = "a2ui-surface"
_A2UI_OPERATIONS_KEY = "a2ui_operations"


def _click_envelopes(forwarded_props: object) -> list[dict[str, Any]]:
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


class A2UIAGUIServer:
    """Serve a plain :class:`~autogen.beta.Agent` over AG-UI for CopilotKit's A2UI renderer.

    Hold a normal ``Agent``, configure A2UI with the same flat kwargs as
    :class:`~autogen.beta.a2ui.rest.A2UIServer`, then either drive
    :meth:`dispatch` directly (yields encoded AG-UI events) or call
    :meth:`build_app` for a ready-to-serve Starlette ASGI app. The server is
    stateless — clients send the full conversation each turn.

    Example::

        from autogen.beta import Agent
        from autogen.beta.a2ui import a2ui_action
        from autogen.beta.a2ui.ag_ui import A2UIAGUIServer


        @a2ui_action(description="Schedule all posts for the given time")
        def schedule_posts(time: str) -> str: ...


        agent = Agent(name="ui", config=..., tools=[schedule_posts])
        app = A2UIAGUIServer(agent, protocol_version="v0.9").build_app()
    """

    __slots__ = ("_agent", "_runtime")

    def __init__(
        self,
        agent: Agent,
        *,
        protocol_version: A2UIVersion = "v0.9",
        custom_catalog: "str | os.PathLike[str] | JsonSchema | None" = None,
        custom_catalog_rules: str | None = None,
        include_schema_in_prompt: bool = True,
        include_rules_in_prompt: bool = True,
        validate_responses: bool = True,
        validation_retries: int = 1,
        system_message: str | None = None,
    ) -> None:
        """Wrap ``agent`` and configure A2UI (see :class:`A2UIServer` for the kwargs)."""
        self._agent = agent
        self._runtime = _A2UIRuntime(
            agent,
            protocol_version=protocol_version,
            custom_catalog=custom_catalog,
            custom_catalog_rules=custom_catalog_rules,
            include_schema_in_prompt=include_schema_in_prompt,
            include_rules_in_prompt=include_rules_in_prompt,
            validate_responses=validate_responses,
            validation_retries=validation_retries,
            system_message=system_message,
        )

    @property
    def agent(self) -> Agent:
        return self._agent

    def _request_from_agui(self, incoming: RunAgentInput) -> A2UIServerRequest:
        """Map an AG-UI ``RunAgentInput`` to a transport-neutral A2UI turn.

        Reuses ``autogen.beta.ag_ui``'s history mapping (system/developer prompt,
        prior turns, trailing user turn) unchanged, then folds in any button
        click: CopilotKit relays a click as ``forwardedProps.a2uiAction`` and
        re-runs the agent (no new chat message), so the click is rewritten into
        the current turn and surfaced as a client interaction — mirroring the
        REST transport's handling of inbound ``a2ui`` envelopes.
        """
        variables = incoming.state if isinstance(incoming.state, dict) else {}
        prompt, history, current_inputs = map_agui_messages_to_events(
            AGStreamInput(incoming=incoming, variables=variables),
        )
        envelopes = _click_envelopes(incoming.forwarded_props)
        current_inputs.extend(TextInput(p) for p in iter_incoming_prompts(envelopes, self._runtime.get_action))
        return A2UIServerRequest(
            current_inputs=current_inputs,
            history=history,
            prompt=prompt,
            variables=variables,
            client_interactions=parse_incoming_interactions(envelopes),
        )

    async def dispatch(self, incoming: RunAgentInput, *, accept: str | None = None) -> AsyncIterator[str]:
        """Run one turn and yield encoded AG-UI events.

        Emits ``RunStarted`` → (``TextMessageChunk`` if there is prose) →
        (one ``ActivitySnapshot`` carrying all A2UI operations, if any) →
        ``RunFinished``. A mid-turn failure surfaces as a ``RunError`` event
        (the run has already started 200 OK on the wire).
        """
        encoder = EventEncoder(accept=accept)
        request = self._request_from_agui(incoming)
        text_message_id = uuid4().hex
        operations: list[object] = []

        yield encoder.encode(RunStartedEvent(thread_id=incoming.thread_id, run_id=incoming.run_id))
        try:
            async for frame in stream_turn(self._agent, self._runtime, request):
                if isinstance(frame, A2UIProseFrame):
                    if frame.text:
                        yield encoder.encode(
                            TextMessageChunkEvent(message_id=text_message_id, role="assistant", delta=frame.text),
                        )
                elif isinstance(frame, A2UIMessageFrame):
                    operations.append(frame.message)

            if operations:
                # One snapshot per turn (replace=True default): the renderer
                # rebuilds the surface(s) from the full operations list.
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

    def build_app(self, *, path: str = "/") -> "Starlette":
        """Starlette ASGI app serving the turn as AG-UI events at ``path`` (POST)."""
        from .asgi import build_app

        return build_app(self, path=path)


__all__ = ("A2UIAGUIServer",)
