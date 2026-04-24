# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""A2A executor that preserves A2UI DataParts in responses.

Extends the standard ``AutogenAgentExecutor`` to parse agent responses for
A2UI JSON content, split into ``TextPart`` + ``DataPart``, and publish both
through the A2A artifact. Also handles A2UI extension negotiation.

Requires the ``a2a`` extra: ``pip install ag2[a2a]``
"""


import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ....import_utils import optional_import_block

with optional_import_block():
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.server.tasks import TaskUpdater
    from a2a.types import DataPart, InternalError, Message, Part, Task, TaskState, TaskStatus, TextPart
    from a2a.utils.errors import ServerError

    from ....a2a.utils import (  # type: ignore[attr-defined]
        make_input_required_message,
        request_message_from_a2a,
    )
    from ....agentchat.remote import AgentService

from ....a2a.constants import A2UI_MIME_TYPE
from .a2a_helpers import (
    A2UI_DEFAULT_DELIMITER,
    MIME_TYPE_KEY,
    create_a2ui_part,
    try_activate_a2ui_extension,
)
from .response_parser import A2UIResponseParser

if TYPE_CHECKING:
    from ....agentchat.conversable_agent import ConversableAgent

logger = logging.getLogger(__name__)


class A2UIAgentExecutor(AgentExecutor):  # type: ignore[misc]
    """A2A executor that preserves A2UI content as DataParts.

    When an agent's response contains A2UI JSON (separated by a delimiter),
    this executor splits it into:
    - A ``TextPart`` for the conversational text
    - A ``DataPart`` with MIME type ``application/json+a2ui`` for the A2UI operations

    Also handles A2UI extension negotiation via ``try_activate_a2ui_extension()``.
    """

    def __init__(
        self,
        agent: ConversableAgent,
        delimiter: str = A2UI_DEFAULT_DELIMITER,
        version_string: str = "v0.9",
    ) -> None:
        self._agent = agent
        self._agent_service = AgentService(agent)
        self._parser = A2UIResponseParser(version_string=version_string, delimiter=delimiter)

    def _extract_incoming_action(self, context: RequestContext) -> dict[str, Any] | None:
        """Check incoming message parts for A2UI action DataParts.

        Supports two formats:
        1. A2UI MIME-typed DataPart: ``{"messages": [{"action": {...}}]}``
        2. genui v0.9 DataPart (no MIME): ``{"version": "v0.9", "action": {...}}``
        """
        assert context.message
        for part in context.message.parts:
            if isinstance(part.root, DataPart):
                data = part.root.data
                # Format 1: A2UI MIME-typed DataPart with messages wrapper
                if part.root.metadata and part.root.metadata.get(MIME_TYPE_KEY) == A2UI_MIME_TYPE:
                    messages = data.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, dict) and "action" in msg:
                            action: dict[str, Any] = msg["action"]
                            return action
                # Format 2: genui v0.9 action DataPart (from genui SurfaceController)
                elif isinstance(data, dict) and "action" in data:
                    action2: dict[str, Any] = data["action"]
                    return action2
        return None

    async def _handle_action(
        self,
        action: dict[str, Any],
        task: Task,
        updater: TaskUpdater,
        context: RequestContext,
    ) -> bool:
        """Handle an incoming A2UI action. Returns True if handled."""
        from .actions import A2UIAction

        action_name = action.get("name", "")
        action_context = action.get("context", {})
        logger.info("Action context for '%s': %s", action_name, action_context)

        # Look up registered action on the agent
        action_def: A2UIAction | None = None
        if hasattr(self._agent, "get_action"):
            action_def = self._agent.get_action(action_name)

        if action_def is None:
            logger.warning("Received unknown A2UI action: %s", action_name)
            return False

        if action_def.tool_name:
            # Tool action: find and call the tool directly
            tool_func = None
            for tool in getattr(self._agent, "_tools", []):
                if hasattr(tool, "name") and tool.name == action_def.tool_name:
                    tool_func = tool
                    break
            # Also check functions registered via register_for_llm
            if tool_func is None:
                for func_map in self._agent.function_map.values():
                    # Check by function name
                    if callable(func_map) and getattr(func_map, "__name__", "") == action_def.tool_name:
                        tool_func = func_map
                        break

            if tool_func is None:
                logger.error("Tool '%s' not found on agent for action '%s'", action_def.tool_name, action_name)
                result_text = f"Error: tool '{action_def.tool_name}' not found."
            else:
                try:
                    result_text = str(tool_func(**action_context))
                    logger.info("Action '%s' called tool '%s': %s", action_name, action_def.tool_name, result_text)
                except Exception as e:
                    logger.error("Tool '%s' failed: %s", action_def.tool_name, e, exc_info=True)
                    result_text = f"Error calling {action_def.tool_name}: {e}"

            # Publish text-only result via status update
            text_part = Part(root=TextPart(text=result_text))
            message = Message(role="agent", message_id=str(uuid4()), parts=[text_part])
            await updater.update_status(state=TaskState.completed, message=message, final=True)
            return True

        else:
            # LLM action: construct a prompt and let the agent handle it
            prompt = f"The user clicked the '{action_name}' button."
            if action_def.description:
                prompt += f" Action: {action_def.description}."
            if action_context:
                prompt += f" Context: {json.dumps(action_context)}"

            logger.info("Action '%s' routed to LLM with prompt: %s", action_name, prompt)

            # Replace only the action DataPart with the prompt, keeping conversation history parts
            assert context.message
            new_parts = []
            for part in context.message.parts:
                if (
                    isinstance(part.root, DataPart)
                    and part.root.metadata
                    and part.root.metadata.get(MIME_TYPE_KEY) == A2UI_MIME_TYPE
                ):
                    # Replace action DataPart with text prompt
                    new_parts.append(Part(root=TextPart(text=prompt)))
                else:
                    new_parts.append(part)
            context.message.parts = new_parts
            return False  # Let normal execute() flow continue

    async def _setup_task(self, context: RequestContext, event_queue: EventQueue) -> tuple[Task, TaskUpdater]:
        """Create or retrieve the task and return it with a TaskUpdater."""
        assert context.message
        task = context.current_task
        if not task:
            request = context.message
            task = Task(
                status=TaskStatus(
                    state=TaskState.submitted,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                id=request.task_id or str(uuid4()),
                context_id=request.context_id or str(uuid4()),
                history=[request],
            )
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(state=TaskState.working)
        return task, updater

    async def _stream_agent_response(
        self, context: RequestContext, task: Task, updater: TaskUpdater
    ) -> tuple[str, bool]:
        """Stream the agent response, forwarding chunks via status updates.

        Returns:
            A tuple of (full_response_text, streaming_started).
        """
        full_response_text = ""
        streaming_started = False
        assert context.message

        async for response in self._agent_service(request_message_from_a2a(context.message)):
            if response.input_required:
                await updater.requires_input(
                    message=make_input_required_message(
                        context_id=task.context_id,
                        task_id=task.id,
                        text=response.input_required,
                        context=response.context,
                    ),
                    final=True,
                )
                return "", False  # Caller should return early

            if response.streaming_text:
                full_response_text += response.streaming_text
                text_part = Part(root=TextPart(text=response.streaming_text))
                message = Message(role="agent", message_id=str(uuid4()), parts=[text_part])
                await updater.update_status(state=TaskState.working, message=message)
                streaming_started = True

            elif response.message:
                content = response.message.get("content", "")
                if isinstance(content, str):
                    full_response_text = content

        return full_response_text, streaming_started

    def _build_final_parts(self, full_response_text: str, use_a2ui: bool) -> list[Part]:
        """Parse the response and build final artifact parts.

        If A2UI content is found, returns TextPart + DataPart.
        Otherwise returns a single TextPart.
        """
        if use_a2ui and full_response_text:
            parse_result = self._parser.parse(full_response_text)

            if parse_result.has_a2ui and parse_result.operations and not parse_result.parse_error:
                parts: list[Part] = []
                if parse_result.text:
                    parts.append(Part(root=TextPart(text=parse_result.text)))
                a2ui_parts = create_a2ui_part(parse_result.operations)
                if isinstance(a2ui_parts, list):
                    parts.extend(a2ui_parts)
                else:
                    parts.append(a2ui_parts)
                return parts

        if full_response_text:
            return [Part(root=TextPart(text=full_response_text))]
        return []

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        assert context.message

        # Negotiate A2UI extension
        use_a2ui = try_activate_a2ui_extension(context)
        if use_a2ui:
            logger.info("A2UI extension activated for this request.")

        # Check for incoming A2UI actions
        incoming_action = self._extract_incoming_action(context)
        if incoming_action:
            logger.info("Received A2UI action: %s", incoming_action.get("name", "?"))

        task, updater = await self._setup_task(context, event_queue)

        # Handle incoming A2UI action if present
        if incoming_action:
            handled = await self._handle_action(incoming_action, task, updater, context)
            if handled:
                return

        try:
            full_response_text, streaming_started = await self._stream_agent_response(context, task, updater)
        except Exception as e:
            raise ServerError(error=InternalError()) from e

        # input_required was handled inside _stream_agent_response
        if not full_response_text and not streaming_started:
            return

        final_parts = self._build_final_parts(full_response_text, use_a2ui)
        if not final_parts:
            final_parts = [Part(root=TextPart(text=full_response_text))] if full_response_text else []

        # Send final response via status update so genui_a2a connector can process it
        message = Message(role="agent", message_id=str(uuid4()), parts=final_parts)
        await updater.update_status(state=TaskState.completed, message=message, final=True)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass
