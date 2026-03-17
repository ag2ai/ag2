# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A2A executor that preserves A2UI DataParts in responses.

Extends the standard ``AutogenAgentExecutor`` to parse agent responses for
A2UI JSON content, split into ``TextPart`` + ``DataPart``, and publish both
through the A2A artifact. Also handles A2UI extension negotiation.

Requires the ``a2a`` extra: ``pip install ag2[a2a]``
"""

from __future__ import annotations

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
    from a2a.types import Artifact, Part, Task, TaskState, TaskStatus, TextPart
    from a2a.utils.errors import ServerError

    from ....a2a.utils import make_input_required_message, new_artifact, request_message_from_a2a
    from ....agentchat.remote import AgentService

from .a2a_helpers import A2UI_MIME_TYPE, MIME_TYPE_KEY, create_a2ui_part, try_activate_a2ui_extension
from .response_parser import A2UIResponseParser

if TYPE_CHECKING:
    from ....agentchat.conversable_agent import ConversableAgent

logger = logging.getLogger(__name__)

CONTEXT_KEY = "ag2_context"
RESULT_ARTIFACT_NAME = "ag2_result"


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
        delimiter: str = "---a2ui_JSON---",
        version_string: str = "v0.9",
    ) -> None:
        self._agent = agent
        self._agent_service = AgentService(agent)
        self._parser = A2UIResponseParser(version_string=version_string, delimiter=delimiter)

    def _extract_incoming_action(self, context: RequestContext) -> dict[str, Any] | None:
        """Check incoming message parts for A2UI action DataParts."""
        from a2a.types import DataPart

        for part in context.message.parts:
            if isinstance(part.root, DataPart) and part.root.metadata:
                if part.root.metadata.get(MIME_TYPE_KEY) == A2UI_MIME_TYPE:
                    messages = part.root.data.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, dict) and "action" in msg:
                            return msg["action"]
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
                    if callable(func_map):
                        # Check by function name
                        if getattr(func_map, "__name__", "") == action_def.tool_name:
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
                    logger.error("Tool '%s' failed: %s", action_def.tool_name, e)
                    result_text = f"Error calling {action_def.tool_name}: {e}"

            # Publish text-only result
            text_part = Part(root=TextPart(text=result_text))
            await updater.add_artifact(
                parts=[text_part],
                artifact_id=str(uuid4()),
                name=RESULT_ARTIFACT_NAME,
                append=False,
                last_chunk=True,
            )
            await updater.complete()
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
            from a2a.types import DataPart

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

        # Handle incoming A2UI action if present
        if incoming_action:
            handled = await self._handle_action(incoming_action, task, updater, context)
            if handled:
                return
            # If not handled (LLM routing), continue with normal flow
            # _handle_action already modified context.message to contain the prompt

        artifact = new_artifact(name=RESULT_ARTIFACT_NAME, parts=[], description=None)
        full_response_text = ""
        streaming_started = False

        try:
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
                    return

                if response.streaming_text:
                    full_response_text += response.streaming_text
                    # Stream text chunks as TextPart (standard behavior)
                    text_part = Part(root=TextPart(text=response.streaming_text))
                    artifact = Artifact(
                        artifact_id=artifact.artifact_id,
                        name=artifact.name,
                        parts=[text_part],
                    )
                    await updater.add_artifact(
                        parts=artifact.parts,
                        artifact_id=artifact.artifact_id,
                        name=artifact.name,
                        append=streaming_started,
                        last_chunk=False,
                    )
                    streaming_started = True

                elif response.message:
                    content = response.message.get("content", "")
                    if isinstance(content, str):
                        full_response_text = content
                    response_context = response.context

        except Exception as e:
            from a2a.types import InternalError

            raise ServerError(error=InternalError()) from e

        # Parse the full response for A2UI content
        if use_a2ui and full_response_text:
            parse_result = self._parser.parse(full_response_text)

            if parse_result.has_a2ui and parse_result.operations and not parse_result.parse_error:
                # Split into TextPart + single DataPart containing all operations.
                # Per spec: "The data field of the DataPart contains a list of A2UI
                # JSON messages. It MUST be an array of messages."
                final_parts: list[Part] = []

                if parse_result.text:
                    final_parts.append(Part(root=TextPart(text=parse_result.text)))

                final_parts.append(create_a2ui_part(parse_result.operations))

                artifact = Artifact(
                    artifact_id=artifact.artifact_id,
                    name=artifact.name,
                    parts=final_parts,
                )

                await updater.add_artifact(
                    artifact_id=artifact.artifact_id,
                    name=artifact.name,
                    parts=artifact.parts,
                    append=streaming_started,
                    last_chunk=True,
                )
                await updater.complete()
                return

        # No A2UI content or extension not active — standard text response
        if full_response_text:
            text_part = Part(root=TextPart(text=full_response_text))
            artifact = Artifact(
                artifact_id=artifact.artifact_id,
                name=artifact.name,
                parts=[text_part],
            )

        await updater.add_artifact(
            artifact_id=artifact.artifact_id,
            name=artifact.name,
            parts=artifact.parts,
            append=streaming_started,
            last_chunk=True,
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass
