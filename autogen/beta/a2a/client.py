# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Iterable, Sequence
from dataclasses import replace as dataclass_replace

import httpx
from a2a.client import Client, ClientCallInterceptor, ClientConfig, ClientFactory
from a2a.types import (
    AgentCard,
    GetTaskRequest,
    Message,
    SendMessageRequest,
    Task,
    TaskState,
)
from fast_depends.library.serializer import SerializerProto

from autogen.beta.config.client import LLMClient
from autogen.beta.context import ConversationContext
from autogen.beta.events import (
    BaseEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolResultEvent,
)
from autogen.beta.response import ResponseProto
from autogen.beta.tools.schemas import ToolSchema

from .cards import fetch_card
from .client_tools import (
    PENDING_TOOL_CALL_ID_VAR_KEY,
    find_pending_tool_result,
    parse_tool_call_request,
    schemas_to_wire,
    tool_result_payload,
)
from .errors import (
    A2AAuthRequiredError,
    A2ANoTaskError,
    A2AReconnectError,
    A2AResponseSchemaNotSupportedError,
    A2ATaskFailedError,
    A2ATaskRejectedError,
)
from .mappers import (
    CLIENT_TOOLS_EXTENSION_URI,
    CLIENT_TOOLS_KEY,
    RESULT_ARTIFACT_NAME,
    TOOL_CALL_RESULT_KEY,
    artifact_metadata_dict,
    artifact_text,
    finish_reason_for,
    finish_reason_from_metadata,
    followup_user_message,
    model_from_metadata,
    model_request_to_a2a_message,
    usage_from_metadata,
)
from .streams import StreamOutcome, drain, reconnect
from .types import TRANSPORT_ERRORS, HttpxClientFactory

CONTEXT_ID_VAR_KEY = "ag:a2a:context_id"
"""``Context.variables`` key for the server-issued A2A ``context_id``."""

TASK_ID_VAR_KEY = "ag:a2a:task_id"
"""``Context.variables`` key for the current A2A ``task_id``."""


class A2AClient(LLMClient):
    def __init__(
        self,
        url: str,
        *,
        client_factory: HttpxClientFactory | None,
        client_config: ClientConfig | None,
        interceptors: list[ClientCallInterceptor],
        max_reconnects: int,
        reconnect_backoff: float,
        agent_card: AgentCard | None,
    ) -> None:
        self._url = url
        self._client_factory = client_factory
        self._user_client_config = client_config
        self._interceptors = interceptors
        self._max_reconnects = max_reconnects
        self._reconnect_backoff = reconnect_backoff
        self._agent_card = agent_card

        self._httpx_client: httpx.AsyncClient | None = None
        self._a2a_client: Client | None = None
        self._init_lock = asyncio.Lock()

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: ConversationContext,
        *,
        tools: Iterable[ToolSchema],
        response_schema: ResponseProto | None,
        serializer: SerializerProto,
    ) -> ModelResponse:
        if response_schema is not None:
            raise A2AResponseSchemaNotSupportedError()

        card, client = await self._ensure_client()
        outgoing = self._build_outgoing(messages, context, tuple(tools))

        outcome = StreamOutcome()
        while True:
            outcome = await self._exchange(client, outgoing, context, outcome)

            tool_call = parse_tool_call_request(outcome)
            if tool_call is not None:
                return self._build_tool_call_response(card, outcome, tool_call, context)

            if outcome.input_required and outcome.task is not None:
                user_text = await context.input(outcome.input_prompt or "")
                outgoing = SendMessageRequest(
                    message=followup_user_message(
                        user_text,
                        context_id=outcome.task.context_id,
                        task_id=outcome.task.id,
                    ),
                )
                outcome = StreamOutcome(text=outcome.text, reasoning=outcome.reasoning)
                continue
            break

        return self._build_final_response(card, outcome, context)

    async def aclose(self) -> None:
        # Httpx clients produced by a user-supplied ``client_factory`` remain
        # the caller's responsibility — the factory contract is "you own what
        # you build". We only close clients we created ourselves.
        async with self._init_lock:
            if self._httpx_client is not None and self._client_factory is None:
                await self._httpx_client.aclose()
            self._httpx_client = None
            self._a2a_client = None

    async def _ensure_client(self) -> tuple[AgentCard, Client]:
        async with self._init_lock:
            if self._httpx_client is None:
                self._httpx_client = (
                    self._client_factory()
                    if self._client_factory
                    else httpx.AsyncClient(
                        # A2A is a streaming long-poll protocol — the default 5s
                        # read timeout aborts mid-SSE on any non-trivial reply.
                        timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0),
                    )
                )
            if self._agent_card is None:
                self._agent_card = await fetch_card(self._httpx_client, self._url)
            if self._a2a_client is None:
                self._a2a_client = self._build_a2a_client(self._agent_card)
            return self._agent_card, self._a2a_client

    def _build_a2a_client(self, card: AgentCard) -> Client:
        streaming = bool(card.capabilities.streaming)
        config = self._user_client_config or ClientConfig(
            httpx_client=self._httpx_client,
            streaming=streaming,
            polling=not streaming,
        )
        if config.httpx_client is None:
            config = dataclass_replace(config, httpx_client=self._httpx_client)
        return ClientFactory(config).create(card, interceptors=self._interceptors)

    def _build_outgoing(
        self,
        messages: Sequence[BaseEvent],
        context: ConversationContext,
        tools: tuple[ToolSchema, ...],
    ) -> SendMessageRequest:
        pending_id = context.variables.get(PENDING_TOOL_CALL_ID_VAR_KEY)
        if pending_id is not None:
            tool_result = find_pending_tool_result(messages, pending_id)
            task_id = context.variables.get(TASK_ID_VAR_KEY)
            context_id = context.variables.get(CONTEXT_ID_VAR_KEY)
            if tool_result is not None and task_id and context_id:
                return SendMessageRequest(
                    message=self._build_tool_result_followup(tool_result, context_id=context_id, task_id=task_id),
                )

        request = _last_model_request(messages)
        if request is None:
            raise ValueError("A2AClient requires at least one ModelRequest in messages")

        wire_tools = schemas_to_wire(tools)
        metadata: dict[str, object] | None = {CLIENT_TOOLS_KEY: wire_tools} if wire_tools else None
        extensions = [CLIENT_TOOLS_EXTENSION_URI] if wire_tools else None

        message = model_request_to_a2a_message(
            request,
            context_id=context.variables.get(CONTEXT_ID_VAR_KEY),
            extensions=extensions,
            metadata=metadata,
        )
        return SendMessageRequest(message=message)

    @staticmethod
    def _build_tool_result_followup(
        tool_result: ToolResultEvent | ToolErrorEvent,
        *,
        context_id: str,
        task_id: str,
    ) -> Message:
        return followup_user_message(
            text="",
            context_id=context_id,
            task_id=task_id,
            metadata={TOOL_CALL_RESULT_KEY: tool_result_payload(tool_result)},
        )

    @staticmethod
    def _build_tool_call_response(
        card: AgentCard,
        outcome: StreamOutcome,
        tool_call: ToolCallEvent,
        context: ConversationContext,
    ) -> ModelResponse:
        assert outcome.task is not None
        context.variables[CONTEXT_ID_VAR_KEY] = outcome.task.context_id
        context.variables[TASK_ID_VAR_KEY] = outcome.task.id
        context.variables[PENDING_TOOL_CALL_ID_VAR_KEY] = tool_call.id
        return ModelResponse(
            tool_calls=ToolCallsEvent([tool_call]),
            model=card.name,
            provider="a2a",
            finish_reason="tool_calls",
        )

    def _build_final_response(
        self,
        card: AgentCard,
        outcome: StreamOutcome,
        context: ConversationContext,
    ) -> ModelResponse:
        if outcome.task is None:
            raise A2ANoTaskError()
        state = outcome.task.status.state
        if state == TaskState.TASK_STATE_FAILED:
            raise A2ATaskFailedError(outcome.task)
        if state == TaskState.TASK_STATE_REJECTED:
            raise A2ATaskRejectedError(outcome.task)

        context.variables[CONTEXT_ID_VAR_KEY] = outcome.task.context_id
        context.variables[TASK_ID_VAR_KEY] = outcome.task.id
        context.variables.pop(PENDING_TOOL_CALL_ID_VAR_KEY, None)

        result_metadata = _result_artifact_metadata(outcome.task)
        usage = usage_from_metadata(result_metadata)
        finish_reason = finish_reason_from_metadata(result_metadata) or finish_reason_for(state)
        model = model_from_metadata(result_metadata) or card.name

        text = outcome.text or _final_text(outcome.task)
        return ModelResponse(
            message=ModelMessage(text) if text else None,
            usage=usage,
            model=model,
            provider="a2a",
            finish_reason=finish_reason,
        )

    async def _exchange(
        self,
        client: Client,
        request: SendMessageRequest,
        context: ConversationContext,
        outcome: StreamOutcome,
    ) -> StreamOutcome:
        await drain(client.send_message(request), context, outcome)
        await reconnect(
            client,
            outcome=outcome,
            context=context,
            max_attempts=self._max_reconnects,
            backoff=self._reconnect_backoff,
        )

        # Some servers complete the task without ever streaming an artifact —
        # pull the final task once so we can read accumulated artifacts/metadata.
        if outcome.task is not None and not outcome.input_required and not outcome.text:
            assert self._a2a_client is not None
            outcome.task = await self._fetch_task_with_retry(outcome.task.id)
            if outcome.task.status.state == TaskState.TASK_STATE_AUTH_REQUIRED:
                raise A2AAuthRequiredError(outcome.task)

        return outcome

    async def _fetch_task_with_retry(self, task_id: str) -> Task:
        assert self._a2a_client is not None
        attempts = 0
        last_error: BaseException | None = None
        while attempts <= self._max_reconnects:
            try:
                return await self._a2a_client.get_task(GetTaskRequest(id=task_id))
            except TRANSPORT_ERRORS as exc:
                last_error = exc
                attempts += 1
                if attempts > self._max_reconnects:
                    break
                await asyncio.sleep(self._reconnect_backoff)
        raise A2AReconnectError(attempts=attempts, last_error=last_error)


def _last_model_request(messages: Sequence[BaseEvent]) -> ModelRequest | None:
    for ev in reversed(messages):
        if isinstance(ev, ModelRequest):
            return ev
    return None


def _final_text(task: Task) -> str:
    return "".join(artifact_text(a) for a in task.artifacts if a.name == RESULT_ARTIFACT_NAME)


def _result_artifact_metadata(task: Task) -> dict[str, object] | None:
    for artifact in task.artifacts:
        if artifact.name == RESULT_ARTIFACT_NAME:
            return artifact_metadata_dict(artifact)
    return None
