# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A2A executor that understands A2UI message splitting and user actions."""

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, Role, TaskState

from autogen.beta.a2a.executor import AgentExecutor
from autogen.beta.a2a.extension import CONTEXT_UPDATE_METADATA_KEY
from autogen.beta.a2a.mappers import ParsedMessage, struct_to_dict, task_state_to_status_update
from autogen.beta.context import ConversationContext
from autogen.beta.events import BaseEvent, ClientToolCallEvent
from autogen.beta.stream import MemoryStream

from .._types import A2UIVersion, JsonObject, ServerToClientMessage
from ..agent import A2UIAgent
from ..capabilities import (
    A2UIClientCapabilities,
    capabilities_to_prompt,
    parse_client_capabilities,
)
from ..constants import A2UI_MIME_TYPE
from ..events import A2UIMessageEvent
from ..incoming import (
    action_to_prompt,
    error_to_prompt,
    function_response_to_prompt,
    parse_incoming_message,
)
from .extension import try_activate_a2ui_extension
from .parts import create_a2ui_parts, get_a2ui_data

logger = logging.getLogger(__name__)


class A2UIAgentExecutor(AgentExecutor):
    """A2A executor that preserves A2UI content as DataParts.

    Extends :class:`autogen.beta.a2a.AgentExecutor` to:

    1. Negotiate the A2UI extension when the client requests it.
    2. Read ``a2uiClientCapabilities`` from the request message metadata and
       fold catalog negotiation into the turn's system prompt
       (:meth:`_extra_system_prompt`).
    3. Detect incoming A2UI DataParts on the request message:
       - ``action`` envelopes are rewritten as a ``TextInput`` prompt
         (Approach 3.B: server-actions and LLM-actions both flow through
         the agent's normal ``ask`` loop, so client-tool round-trips,
         streaming, and middleware all work uniformly).
       - ``error`` envelopes (e.g. ``VALIDATION_FAILED``) are rewritten as
         a corrective ``TextInput`` so the agent can regenerate.
    4. Collect the :class:`A2UIMessageEvent`s the validation middleware emits
       during the turn and split the completed task into a text ``Part`` (the
       conversational prose, already stripped from ``ModelResponse.content``)
       plus a canonical A2UI DataPart (MIME ``application/a2ui+json``) carrying
       the collected message list.
    """

    def __init__(self, agent: A2UIAgent) -> None:
        super().__init__(agent)
        self._a2ui_agent = agent

    @property
    def protocol_version(self) -> A2UIVersion:
        return self._a2ui_agent.protocol_version

    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        if request_context.message is not None:
            try_activate_a2ui_extension(request_context)
            self._rewrite_incoming_a2ui_parts(request_context)
        await super().execute(request_context, event_queue)

    def _rewrite_incoming_a2ui_parts(self, request_context: RequestContext) -> None:
        """Replace A2UI DataParts (``action``/``error``) with synthesized text prompts."""
        msg = request_context.message
        if msg is None:
            return

        new_parts: list[Part] = []
        rewritten = False
        for part in msg.parts:
            envelopes = _extract_a2ui_envelopes(part)
            if not envelopes:
                new_parts.append(part)
                continue

            for envelope in envelopes:
                result = parse_incoming_message(envelope)
                if result.kind == "action" and result.action is not None:
                    action_def = self._a2ui_agent.get_action(result.action.name)
                    prompt = action_to_prompt(result.action, action_def)
                    if prompt is None:
                        logger.warning(
                            "Dropping A2UI action '%s' — no matching A2UIAction registration.",
                            result.action.name,
                        )
                        continue
                    new_parts.append(Part(text=prompt))
                    rewritten = True
                elif result.kind == "functionResponse" and result.function_response is not None:
                    if not result.function_response.function_call_id:
                        logger.warning(
                            "Dropping A2UI functionResponse — missing functionCallId, cannot correlate to a callFunction.",
                        )
                        continue
                    new_parts.append(Part(text=function_response_to_prompt(result.function_response)))
                    rewritten = True
                elif result.kind == "error" and result.error is not None:
                    new_parts.append(Part(text=error_to_prompt(result.error)))
                    rewritten = True
                else:
                    logger.debug("Skipping A2UI envelope of unknown kind: %s", result.parse_error)

        if rewritten:
            del msg.parts[:]
            msg.parts.extend(new_parts)

    def _extra_system_prompt(self, request_context: RequestContext) -> Sequence[str]:
        """Fold negotiated client capabilities into the turn's system prompt.

        Reads ``a2uiClientCapabilities`` straight off the incoming message
        metadata and renders it as a per-turn prompt fragment so the LLM only
        targets components the client can render. Returns nothing when the
        client advertised no usable capabilities.
        """
        caps = self._client_capabilities(request_context)
        if caps is None:
            return ()
        fragment = capabilities_to_prompt(caps, catalog_id=self._a2ui_agent.catalog_id)
        return (fragment,) if fragment else ()

    def _client_capabilities(self, request_context: RequestContext) -> "A2UIClientCapabilities | None":
        """Decode ``a2uiClientCapabilities`` from the request message metadata.

        ``RequestContext.metadata`` is a read-only copy, so the capabilities are
        read from their source — ``message.metadata`` — converting the protobuf
        ``Struct`` to a plain dict first.
        """
        msg = request_context.message
        if msg is None or not msg.metadata:
            return None
        version_key = self._a2ui_agent.schema_manager.version_string
        return parse_client_capabilities(struct_to_dict(msg.metadata), version_key=version_key)

    async def _run_one_turn(
        self,
        parsed: ParsedMessage,
        updater: TaskUpdater,
        stream: MemoryStream,
        lifecycle_ctx: ConversationContext,
        text_pieces: list[str],
        pending_client_calls: list[ClientToolCallEvent],
        task_id: str,
        context_id: str,
        extra_prompt: Sequence[str] = (),
    ) -> None:
        client_tools = [self._make_client_tool(s) for s in parsed.tool_schemas]
        initial_event = self._build_initial_event(parsed)

        # The validation middleware emits one A2UIMessageEvent per validated
        # A2UI message onto this turn's stream. Collect them as the single
        # source of UI content (the event seam) instead of re-parsing text.
        a2ui_messages: list[ServerToClientMessage] = []

        @stream.subscribe
        async def _collect_a2ui_messages(event: BaseEvent) -> None:
            if isinstance(event, A2UIMessageEvent):
                a2ui_messages.append(event.message)

        response, final_variables = await self._dispatch_to_agent(
            initial_event,
            stream,
            client_tools,
            incoming_variables=parsed.context_update,
            extra_prompt=extra_prompt,
        )

        prose_text = response.message.content if response.message else ""
        agent_msg = self._build_a2ui_message(updater, prose_text or "", a2ui_messages, final_variables)

        # v1.0: a server-initiated callFunction(wantResponse=true) pauses the
        # task awaiting the client's functionResponse. Unlike client tool calls
        # (streamed as artifacts during the turn), the callFunction DataPart only
        # lives in the finalization message — so it must ride the input-required
        # transition, otherwise the client never learns what function to run.
        wants_client_response = any(
            isinstance(m, dict) and "callFunction" in m and m.get("wantResponse") for m in a2ui_messages
        )
        has_pending = bool(response.tool_calls and response.tool_calls.calls and response.response_force)
        if wants_client_response:
            if has_pending or pending_client_calls:
                # By construction these are mutually exclusive (the validation
                # middleware skips A2UI parsing when the response carries tool
                # calls), so reaching here means a malformed turn — surface it
                # rather than silently dropping the pending tool-call state.
                logger.warning(
                    "callFunction(wantResponse) emitted alongside pending tool calls; "
                    "pausing for the client function and not handling the pending tool calls this turn.",
                )
            await updater.requires_input(message=agent_msg)
            await lifecycle_ctx.send(
                task_state_to_status_update(
                    TaskState.TASK_STATE_INPUT_REQUIRED,
                    task_id=task_id,
                    context_id=context_id,
                    message=agent_msg,
                    timestamp=datetime.now(tz=timezone.utc),
                ),
            )
            return

        if has_pending or pending_client_calls:
            await updater.requires_input()
            await lifecycle_ctx.send(
                task_state_to_status_update(
                    TaskState.TASK_STATE_INPUT_REQUIRED,
                    task_id=task_id,
                    context_id=context_id,
                    timestamp=datetime.now(tz=timezone.utc),
                ),
            )
            return

        await updater.complete(message=agent_msg)
        await lifecycle_ctx.send(
            task_state_to_status_update(
                TaskState.TASK_STATE_COMPLETED,
                task_id=task_id,
                context_id=context_id,
                message=agent_msg,
                timestamp=datetime.now(tz=timezone.utc),
            ),
        )

    def _build_a2ui_message(
        self,
        updater: TaskUpdater,
        prose_text: str,
        a2ui_messages: list[ServerToClientMessage],
        final_variables: dict[str, Any],
    ) -> Message | None:
        """Build a finalization message that splits prose from A2UI messages.

        When the turn produced A2UI messages (collected from the stream's
        :class:`A2UIMessageEvent`s), a single canonical A2UI DataPart carrying
        the full message list is emitted alongside a text ``Part`` for the
        conversational prose. With no A2UI content this falls back to a single
        text ``Part`` — the same shape the base executor would produce.
        """
        parts: list[Part] = []

        if prose_text:
            parts.append(Part(text=prose_text))
        if a2ui_messages:
            parts.extend(create_a2ui_parts(a2ui_messages))

        if not parts and not final_variables:
            return None

        metadata: dict[str, Any] | None = None
        if final_variables:
            metadata = {CONTEXT_UPDATE_METADATA_KEY: final_variables}

        return updater.new_agent_message(parts=parts, metadata=metadata)


def _extract_a2ui_envelopes(part: Part) -> list[JsonObject]:
    """Return zero or more A2UI envelope dicts from an A2A ``Part``.

    Supports three on-the-wire shapes:

    - Canonical A2A v1.0: DataPart ``data`` is a JSON list of envelopes.
    - Legacy single dict: DataPart ``data`` is one envelope object.
    - Legacy wrapper:     DataPart ``data`` is ``{"messages": [envelope, ...]}``.

    Returns only entries that carry an ``action``, ``functionResponse``, or
    ``error`` payload.
    """
    data = get_a2ui_data(part)
    if data is None:
        return []

    raw_entries: list[JsonObject] = []
    if isinstance(data, list):
        raw_entries.extend(d for d in data if isinstance(d, dict))
    elif isinstance(data, dict):
        messages = data.get("messages")
        if isinstance(messages, list):
            raw_entries.extend(m for m in messages if isinstance(m, dict))
        else:
            raw_entries.append(data)

    return [
        e
        for e in raw_entries
        if isinstance(e.get("action"), dict)
        or isinstance(e.get("functionResponse"), dict)
        or isinstance(e.get("error"), dict)
    ]


# Re-export for callers that want to type-check against the protobuf Role.
__all__ = ("A2UI_MIME_TYPE", "A2UIAgentExecutor", "Role")
