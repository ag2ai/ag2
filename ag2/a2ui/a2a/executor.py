# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A2A executor that understands A2UI message splitting and user actions."""

import contextvars
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, Part, TaskState

from ag2.a2a.executor import AgentExecutor, _TaskWriter
from ag2.a2a.extension import CONTEXT_UPDATE_METADATA_KEY
from ag2.a2a.mappers import ParsedMessage, struct_to_dict
from ag2.agent import Agent
from ag2.context import strip_reserved_variables
from ag2.events import BaseEvent
from ag2.stream import MemoryStream

from .._runtime import _A2UIRuntime
from .._types import A2UIVersion, JsonObject, JsonSchema, ServerToClientMessage
from ..actions import A2UIAction, collect_action_declarations, collect_server_actions
from ..capabilities import (
    A2UIClientCapabilities,
    parse_client_capabilities,
)
from ..data_model import data_model_to_prompt, parse_client_data_model
from ..events import A2UIMessageEvent
from ..incoming import (
    A2UIIncomingParseResult,
    iter_incoming_prompts,
    parse_incoming_interactions,
)
from ..middleware import A2UIInboundMiddleware
from ..server_action import run_client_interactions
from .extension import try_activate_a2ui_extension
from .parts import create_a2ui_parts, get_a2ui_data

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class _IncomingRequest:
    interactions: tuple[A2UIIncomingParseResult, ...]
    awaiting_input: bool


# Per-request carrier for the client→server interactions parsed in ``execute``
# (where the raw A2UI DataParts are available, before they are rewritten to
# text) so ``_run_one_turn`` can surface them as A2UIClientEvents. A ContextVar
# keeps this coroutine-local — the executor instance is shared across requests,
# so a plain attribute would race between concurrent turns.
_INCOMING_REQUEST: "contextvars.ContextVar[_IncomingRequest | None]" = contextvars.ContextVar(
    "a2ui_incoming_request", default=None
)


class A2UIAgentExecutor(AgentExecutor):
    """A2A executor that preserves A2UI content as DataParts.

    Extends :class:`ag2.a2a.AgentExecutor` to:

    1. Negotiate the A2UI extension when the client requests it.
    2. Read ``a2uiClientCapabilities`` (``a2uiRendererCapabilities`` in v1.0) from the request message metadata and
       fold catalog negotiation into the turn's system prompt
       (:meth:`_extra_system_prompt`).
    3. Detect incoming A2UI DataParts on the request message:
       - an ``action`` envelope for a **registered** action runs that action's
         handler on the server (the agent is not invoked); its messages lead the
         finalization DataPart.
       - an ``action`` envelope with no registered action is rewritten as a
         generic ``TextInput`` prompt so the agent can react to the button it
         itself rendered.
       - a v1.0 ``callAgentFunction`` envelope is answered on the server with an
         ``agentFunctionResponse`` (the agent is not invoked).
       - a v1.0 ``rendererFunctionResponse`` (the reply to a ``callRendererFunction``)
         and ``error`` envelopes (e.g. ``VALIDATION_FAILED``) are rewritten as
         a ``TextInput`` so the agent can continue or regenerate.
    4. Collect the :class:`A2UIMessageEvent`s the validation middleware emits
       during the turn and split the completed task into a text ``Part`` (the
       conversational prose, already stripped from ``ModelResponse.content``)
       plus a canonical A2UI DataPart (MIME ``application/a2ui+json``) carrying
       the collected message list.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        actions: Sequence[A2UIAction] = (),
        protocol_version: A2UIVersion = "v0.9",
        custom_catalog: "str | os.PathLike[str] | JsonSchema | None" = None,
        custom_catalog_rules: str | None = None,
        include_schema_in_prompt: bool = True,
        include_rules_in_prompt: bool = True,
        validate_responses: bool = True,
        validation_retries: int = 1,
        system_message: str | None = None,
    ) -> None:
        """Wrap a plain ``Agent`` as an A2UI-aware A2A executor.

        Takes the same flat A2UI kwargs as :class:`A2UIServer`. Clickable buttons
        (``@a2ui_action``) are declared via ``actions`` — not on the agent — so it
        stays plain and reusable across deployments. A click on a registered
        action runs its handler on the server (the agent is not invoked); a click
        on any other button is rewritten into a prompt for the agent.
        """
        super().__init__(agent)
        action_objs = tuple(actions)
        self._server_actions = collect_server_actions(action_objs)
        self._runtime = _A2UIRuntime(
            actions=collect_action_declarations(action_objs),
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
    def protocol_version(self) -> A2UIVersion:
        return self._runtime.protocol_version

    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        token: contextvars.Token[_IncomingRequest | None] | None = None
        if request_context.message is not None:
            try_activate_a2ui_extension(request_context, version=self.protocol_version)
            interactions = self._rewrite_incoming_a2ui_parts(request_context)
            task = request_context.current_task
            token = _INCOMING_REQUEST.set(
                _IncomingRequest(
                    interactions=tuple(interactions),
                    awaiting_input=task is not None and task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED,
                )
            )
        try:
            await super().execute(request_context, event_queue)
        finally:
            if token is not None:
                _INCOMING_REQUEST.reset(token)

    def _rewrite_incoming_a2ui_parts(self, request_context: RequestContext) -> list[A2UIIncomingParseResult]:
        """Consume incoming A2UI DataParts, rewriting them into text prompts.

        Every A2UI DataPart is removed from the message: a click on an
        unregistered button (and an ``error`` envelope) becomes a synthesized
        text prompt for the agent; a click on a **registered** action yields no
        prompt — it runs on the server in ``_run_one_turn`` — but the raw DataPart
        is still dropped so it never reaches the agent.

        Returns the typed client→server interactions parsed from those DataParts
        so ``_run_one_turn`` can run their server handlers and surface them as
        ``A2UIClientEvent``s (the parts are consumed here, so they must be
        captured now).
        """
        msg = request_context.message
        if msg is None:
            return []

        new_parts: list[Part] = []
        consumed_a2ui = False
        all_envelopes: list[JsonObject] = []
        for part in msg.parts:
            envelopes = _extract_a2ui_envelopes(part)
            if not envelopes:
                new_parts.append(part)
                continue

            # An A2UI DataPart is always consumed (never forwarded raw); a
            # registered action simply produces no replacement prompt.
            consumed_a2ui = True
            all_envelopes.extend(envelopes)
            for prompt in iter_incoming_prompts(envelopes, self._runtime.get_action):
                new_parts.append(Part(text=prompt))

        if consumed_a2ui:
            del msg.parts[:]
            msg.parts.extend(new_parts)

        return parse_incoming_interactions(all_envelopes)

    def _extra_system_prompt(self, request_context: RequestContext) -> Sequence[str]:
        """Apply the A2UI prompt to the plain agent and fold in client capabilities.

        The agent is a plain ``Agent`` with no A2UI prompt baked in, so the
        runtime's system-prompt section is prepended here. Negotiated
        ``a2uiClientCapabilities`` / ``a2uiRendererCapabilities`` (read off the incoming message metadata) are
        appended as a per-turn fragment so the LLM only targets components the
        client can render; so is the renderer's data model (``a2uiRendererDataModel``).
        """
        prompts: list[str] = [self._runtime.system_prompt_section]
        caps = self._client_capabilities(request_context)
        if caps is not None:
            fragment = self._runtime.capabilities_prompt(caps)
            if fragment:
                prompts.append(fragment)
        msg = request_context.message
        if msg is not None and msg.metadata:
            surfaces = parse_client_data_model(struct_to_dict(msg.metadata), version_key=self._runtime.version_string)
            if data_model_fragment := data_model_to_prompt(surfaces):
                prompts.append(data_model_fragment)
        return tuple(prompts)

    def _client_capabilities(self, request_context: RequestContext) -> "A2UIClientCapabilities | None":
        """Decode ``a2uiClientCapabilities`` (``a2uiRendererCapabilities`` in v1.0) from the request message metadata.

        ``RequestContext.metadata`` is a read-only copy, so the capabilities are
        read from their source — ``message.metadata`` — converting the protobuf
        ``Struct`` to a plain dict first.
        """
        msg = request_context.message
        if msg is None or not msg.metadata:
            return None
        version_key = self._runtime.version_string
        return parse_client_capabilities(struct_to_dict(msg.metadata), version_key=version_key)

    async def _run_one_turn(
        self,
        parsed: ParsedMessage,
        writer: _TaskWriter,
        stream: MemoryStream,
        extra_prompt: Sequence[str] = (),
    ) -> None:
        # Buttons run on the server, not as agent tools — only client-side tool
        # schemas the request carried are injected for the turn.
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

        # Surface each incoming client→server interaction (captured in execute,
        # before the A2UI DataParts were rewritten to text) as an A2UIClientEvent
        # on the turn's stream, alongside the validation middleware.
        extra_middleware = list(self._runtime.middleware_factories())
        incoming = _INCOMING_REQUEST.get()
        interactions = incoming.interactions if incoming is not None else ()
        if interactions:
            extra_middleware.append(A2UIInboundMiddleware(interactions))

        # Answer the interactions the server handles itself — registered clicks and
        # callAgentFunction calls (the prompt rewriter already skipped these, so the
        # agent never sees them). Their messages lead the finalization DataPart,
        # ahead of the agent's own. Handlers resolve their dependencies against the
        # agent's DI surface, exactly like the agent's own tools (and the REST path).
        server_messages, handled_server = await run_client_interactions(
            interactions,
            self._server_actions,
            self._agent,
            variables=parsed.context_update,
            version=self._runtime.version_string,
            catalog_id=self._runtime.catalog_id,
        )
        a2ui_messages.extend(server_messages)

        # Pure server interactions skip the agent, but their messages still go
        # through the shared finalization below: a handler can itself request a
        # renderer function and must leave the task open for its response.
        server_only = handled_server and not parsed.inputs and not parsed.tool_results
        if server_only:
            agent_msg = self._build_a2ui_message(writer, "", a2ui_messages, {})
            # An unrelated server RPC does not answer the renderer function or
            # client tool that left this task awaiting input on an earlier turn.
            has_pending = incoming is not None and incoming.awaiting_input
        else:
            response, final_variables = await self._dispatch_to_agent(
                initial_event,
                stream,
                client_tools,
                incoming_variables=parsed.context_update,
                extra_prompt=extra_prompt,
                additional_middleware=extra_middleware,
            )
            prose_text = response.message.content if response.message else ""
            agent_msg = self._build_a2ui_message(writer, prose_text or "", a2ui_messages, final_variables)
            has_pending = bool(response.tool_calls and response.tool_calls.calls and response.response_force)

        # v1.0: a callRendererFunction pauses the task awaiting the renderer's
        # rendererFunctionResponse. Unlike client tool calls (streamed as
        # artifacts during the turn), the callRendererFunction DataPart only lives
        # in the finalization message — so it must ride the input-required
        # transition, otherwise the client never learns what function to run.
        calls_renderer_function = any(isinstance(m, dict) and "callRendererFunction" in m for m in a2ui_messages)
        if calls_renderer_function:
            if has_pending or writer.pending_client_calls:
                # A callRendererFunction and pending tool calls can co-occur in
                # a turn; give the callRendererFunction precedence and defer the tool calls.
                logger.warning(
                    "callRendererFunction emitted alongside pending tool calls; "
                    "pausing for the renderer function and deferring the pending tool calls this turn.",
                )
            await writer.requires_input(message=agent_msg)
            return

        if has_pending or writer.pending_client_calls:
            pending_message = agent_msg if server_only else None
            await writer.requires_input(message=pending_message)
            return

        await writer.complete(message=agent_msg)

    def _build_a2ui_message(
        self,
        writer: _TaskWriter,
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

        outgoing = strip_reserved_variables(final_variables, source="an outgoing A2UI response", warn=False)
        if not parts and not outgoing:
            return None

        metadata: dict[str, Any] | None = None
        if outgoing:
            metadata = {CONTEXT_UPDATE_METADATA_KEY: outgoing}

        return writer.new_agent_message(parts=parts, metadata=metadata)


_RENDERER_ENVELOPE_KEYS = ("action", "callAgentFunction", "rendererFunctionResponse", "error")


def _extract_a2ui_envelopes(part: Part) -> list[JsonObject]:
    """Return the A2UI envelope dicts the renderer sends the agent from an A2A ``Part``.

    Those carry ``action``, ``callAgentFunction``, ``rendererFunctionResponse`` or ``error``.
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

    return [e for e in raw_entries if any(isinstance(e.get(key), dict) for key in _RENDERER_ENVELOPE_KEYS)]


__all__ = ("A2UIAgentExecutor",)
