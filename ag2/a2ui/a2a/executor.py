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
from ag2.a2a.mappers import ParsedMessage, part_data_to_python, struct_to_dict
from ag2.agent import Agent
from ag2.context import strip_reserved_variables
from ag2.stream import MemoryStream

from .._messages import is_client_envelope
from .._runtime import build_runtime
from .._turn import A2UITurn, open_turn
from .._types import A2UIVersion, JsonSchema, JsonValue, ServerToClientMessage
from ..actions import A2UIAction
from ..incoming import A2UIRendererInput, read_renderer_input
from .extension import try_activate_a2ui_extension
from .parts import create_a2ui_parts, is_a2ui_part

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class _IncomingRequest:
    turn: A2UITurn
    awaiting_input: bool


# Per-request carrier for the ``A2UITurn`` opened in ``execute`` (where the raw
# A2UI DataParts are still available, before they are rewritten to text) and for
# whether the task was awaiting input, so ``_run_one_turn`` reuses that turn. A ContextVar
# keeps this coroutine-local — the executor instance is shared across requests,
# so a plain attribute would race between concurrent turns.
_INCOMING_REQUEST: "contextvars.ContextVar[_IncomingRequest | None]" = contextvars.ContextVar(
    "a2ui_incoming_request", default=None
)


class A2UIAgentExecutor(AgentExecutor):
    """A2A executor that preserves A2UI content as DataParts.

    Extends :class:`ag2.a2a.AgentExecutor` to:

    1. Negotiate the A2UI extension when the client requests it (``X-A2A-Extensions`` or ``message.extensions``).
    2. Read ``a2uiClientCapabilities`` (``a2uiRendererCapabilities`` in v1.0) from the request message metadata,
       then from each DataPart's metadata, and offer only the agent's catalogs the renderer names. In v1.0 a
       message without capabilities offers no A2UI: the turn runs without the A2UI prompt and its reply carries
       no A2UI DataPart (A2A extension v1.0, "Capability scoping").
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
        self._runtime = build_runtime(
            actions,
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
        return self._runtime.protocol.version

    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        token: contextvars.Token[_IncomingRequest | None] | None = None
        if request_context.message is not None:
            try_activate_a2ui_extension(request_context, version=self.protocol_version)
            renderer = self._read_renderer_input(request_context.message)
            task = request_context.current_task
            token = _INCOMING_REQUEST.set(
                _IncomingRequest(
                    turn=open_turn(self._runtime, renderer),
                    awaiting_input=task is not None and task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED,
                )
            )
        try:
            await super().execute(request_context, event_queue)
        finally:
            if token is not None:
                _INCOMING_REQUEST.reset(token)

    def _read_renderer_input(self, msg: Message) -> A2UIRendererInput:
        """Decode what the renderer sent with ``msg``, consuming its A2UI DataParts.

        ``a2uiClientCapabilities`` (``a2uiRendererCapabilities`` in v1.0) are read, as the A2A extension says, from
        ``Message.metadata`` first and then from each DataPart's metadata in order; the first entry for the served
        version wins. They are read before the DataParts are consumed. ``RequestContext.metadata`` is a read-only
        copy, so the metadata is read from its source, converting each protobuf ``Struct`` to a plain dict. The data
        model (``a2uiRendererDataModel``) sits in ``Message.metadata``.

        Every A2UI DataPart is removed from the message: a click on an unregistered button (and an ``error``
        envelope) becomes a synthesized text prompt for the agent, in the part's place; a click on a **registered**
        action yields no prompt (it runs on the server in ``_run_one_turn``), but the raw DataPart is still dropped
        so it never reaches the agent. The decoded interactions are returned so ``_run_one_turn`` can run their
        server handlers and surface them as ``A2UIClientEvent``s.
        """
        metadata = struct_to_dict(msg.metadata) if msg.metadata else None
        capability_sources = [
            metadata,
            *(struct_to_dict(part.metadata) for part in msg.parts if part.HasField("data") and part.metadata),
        ]
        groups = [_extract_a2ui_envelopes(part) for part in msg.parts]
        renderer = read_renderer_input(
            self._runtime.protocol.version,
            self._runtime.get_action,
            envelope_groups=[group for group in groups if group is not None],
            capability_sources=capability_sources,
            data_model_source=metadata,
        )
        if any(group is not None for group in groups):
            # An A2UI DataPart is always consumed (never forwarded raw); its prompts, if any, take its place.
            prompts = iter(renderer.prompts)
            new_parts: list[Part] = []
            for part, group in zip(msg.parts, groups, strict=True):
                if group is None:
                    new_parts.append(part)
                else:
                    new_parts.extend(Part(text=prompt) for prompt in next(prompts))
            del msg.parts[:]
            msg.parts.extend(new_parts)
        return renderer

    def _extra_system_prompt(self, request_context: RequestContext) -> Sequence[str]:
        """Apply the turn's A2UI prompt to the plain agent: the offer's A2UI section (the catalogs the renderer
        supports; none when the turn offers no A2UI), then the renderer's capabilities and data model."""
        return self._turn_of(_INCOMING_REQUEST.get()).system_prompt

    def _turn_of(self, incoming: _IncomingRequest | None) -> A2UITurn:
        """The request's turn; with no request captured (no message to read), that of a request without
        capabilities, which in v1.0 withdraws A2UI (a2ui_protocol.md, "Renderer capabilities")."""
        return incoming.turn if incoming is not None else open_turn(self._runtime, A2UIRendererInput())

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

        # The interactions were captured in execute, before the A2UI DataParts were rewritten to text.
        incoming = _INCOMING_REQUEST.get()
        turn = self._turn_of(incoming)
        # The validation middleware emits one A2UIMessageEvent per validated
        # A2UI message onto this turn's stream. Collect them as the single
        # source of UI content (the event seam) instead of re-parsing text.
        agent_messages = turn.collect(stream)

        # Answer the interactions the server handles itself — registered clicks and
        # callAgentFunction calls (the prompt rewriter already skipped these, so the
        # agent never sees them). Their messages lead the finalization DataPart,
        # ahead of the agent's own. Handlers resolve their dependencies against the
        # agent's DI surface, exactly like the agent's own tools (and the REST path).
        answers = await turn.answer_interactions(self._agent, parsed.context_update)

        # Pure server interactions skip the agent, but their messages still go
        # through the shared finalization below: a handler can itself request a
        # renderer function and must leave the task open for its response.
        server_only = answers.complete_without_agent(has_input=bool(parsed.inputs or parsed.tool_results))
        a2ui_messages: list[ServerToClientMessage]
        if server_only:
            a2ui_messages = answers.messages
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
                # Validates the A2UI output and surfaces each incoming interaction as an A2UIClientEvent.
                additional_middleware=turn.middleware,
            )
            prose_text = response.message.content if response.message else ""
            a2ui_messages = [*answers.messages, *turn.deliverable(agent_messages)]
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


def _extract_a2ui_envelopes(part: Part) -> list[JsonValue] | None:
    """The envelopes an A2UI DataPart carries for the agent, or ``None`` when ``part`` is not an A2UI DataPart.

    Those carry ``action``, ``callAgentFunction``, ``rendererFunctionResponse`` or ``error``. Every A2UI DataPart
    is consumed, so none reaches the agent raw. The A2A extension says ``data`` "MUST be an array of messages", so
    one whose ``data`` is not an array yields none. In an array each message is processed on its own, and the
    receiver "MUST continue processing the remaining messages in the list" ("Processing Rules"), so an entry that
    is not a renderer-to-agent message is dropped with a warning and the rest are kept.
    """
    if not is_a2ui_part(part):
        return None
    data = part_data_to_python(part)
    if not isinstance(data, list):
        logger.warning("Dropping an A2UI DataPart whose 'data' is not an array of messages.")
        return []
    envelopes: list[JsonValue] = [entry for entry in data if is_client_envelope(entry)]
    if dropped := len(data) - len(envelopes):
        logger.warning("Dropping %d A2UI DataPart entry(ies): not a renderer-to-agent message.", dropped)
    return envelopes


__all__ = ("A2UIAgentExecutor",)
