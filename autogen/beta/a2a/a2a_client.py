# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Any

import httpx
from a2a.client import (
    A2ACardResolver,
    A2AClientHTTPError,
    Client,
    ClientCallInterceptor,
    ClientConfig,
    ClientFactory,
)
from a2a.types import (
    AgentCard,
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskQueryParams,
    TaskState,
)
from fast_depends.library.serializer import SerializerProto

from autogen.beta.config.client import LLMClient
from autogen.beta.context import ConversationContext
from autogen.beta.events import BaseEvent, ModelMessage, ModelRequest, ModelResponse
from autogen.beta.response import ResponseProto
from autogen.beta.tools.schemas import ToolSchema

from .errors import A2AClientToolsNotSupportedError, A2AReconnectError
from .mappers import (
    artifact_text,
    followup_user_message,
    model_request_to_a2a_message,
    task_artifact_update_to_chunks,
    text_from_message,
)
from .utils import CONTEXT_ID_VAR_KEY, PROVIDER_NAME, TASK_ID_VAR_KEY

HttpxClientFactory = Callable[[], httpx.AsyncClient]

_TERMINAL_STATES = frozenset({TaskState.completed, TaskState.canceled, TaskState.failed, TaskState.rejected})

_StreamEvent = tuple[Task, Any] | Message  # noqa: UP007
_StreamIter = AsyncIterator[_StreamEvent]


@dataclass(slots=True)
class _StreamOutcome:
    """Result of consuming one A2A streaming session."""

    text: str = ""
    task: Task | None = None
    input_required: bool = False
    input_prompt: str | None = None


class A2AClient(LLMClient):
    def __init__(
        self,
        url: str,
        *,
        client_factory: HttpxClientFactory | None,
        client_config: ClientConfig | None,
        interceptors: list[ClientCallInterceptor],
        max_reconnects: int,
        polling_interval: float,
        agent_card: AgentCard | None,
    ) -> None:
        self._url = url
        self._client_factory = client_factory
        self._user_client_config = client_config
        self._interceptors = interceptors
        self._max_reconnects = max_reconnects
        self._polling_interval = polling_interval
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
        if list(tools) or response_schema is not None:
            raise A2AClientToolsNotSupportedError()

        request = _last_model_request(messages)
        if request is None:
            raise ValueError("A2AClient requires at least one ModelRequest in messages")

        card, client = await self._ensure_client()
        message = model_request_to_a2a_message(request, context_id=context.variables.get(CONTEXT_ID_VAR_KEY))

        accumulated = ""
        outcome: _StreamOutcome
        while True:
            outcome = await self._exchange(client, message, context, accumulated)
            accumulated = outcome.text

            if outcome.input_required and outcome.task is not None:
                user_text = await context.input(outcome.input_prompt or "")
                message = followup_user_message(user_text, context_id=outcome.task.context_id, task_id=outcome.task.id)
                continue
            break

        if outcome.task is None:
            raise A2AReconnectError(attempts=0)

        context.variables[CONTEXT_ID_VAR_KEY] = outcome.task.context_id
        context.variables[TASK_ID_VAR_KEY] = outcome.task.id

        text = accumulated or _final_text(outcome.task)
        return ModelResponse(
            message=ModelMessage(text) if text else None,
            model=card.name,
            provider=PROVIDER_NAME,
            finish_reason=outcome.task.status.state.value,
        )

    async def aclose(self) -> None:
        """Release the lazily-created `httpx.AsyncClient` (if we own it)."""
        if self._httpx_client is not None and self._client_factory is None:
            await self._httpx_client.aclose()
            self._httpx_client = None
            self._a2a_client = None

    async def _ensure_client(self) -> tuple[AgentCard, Client]:
        async with self._init_lock:
            if self._httpx_client is None:
                self._httpx_client = self._client_factory() if self._client_factory else httpx.AsyncClient()
            if self._agent_card is None:
                self._agent_card = await A2ACardResolver(self._httpx_client, self._url).get_agent_card()
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

    async def _exchange(
        self,
        client: Client,
        message: Message,
        context: ConversationContext,
        seed_text: str,
    ) -> _StreamOutcome:
        """Send one message and consume the resulting stream until terminal/HITL.

        Reconnect is handled here transparently: if the stream drops mid-task
        before terminal state and we know the task id, we issue
        `client.resubscribe(...)` up to `max_reconnects` times.
        """
        outcome = _StreamOutcome(text=seed_text)
        last_error = await self._drain(client.send_message(message), context, outcome)

        attempts = 0
        while _needs_reconnect(outcome) and attempts < self._max_reconnects:
            assert outcome.task is not None
            attempts += 1
            try:
                last_error = await self._drain(client.resubscribe(TaskIdParams(id=outcome.task.id)), context, outcome)
            except (httpx.HTTPError, A2AClientHTTPError) as exc:
                last_error = exc
                await asyncio.sleep(self._polling_interval)

        if _needs_reconnect(outcome):
            raise A2AReconnectError(attempts=attempts, last_error=last_error)

        if outcome.task is not None and not outcome.input_required and not outcome.text:
            assert self._a2a_client is not None
            outcome.task = await self._a2a_client.get_task(TaskQueryParams(id=outcome.task.id))

        return outcome

    async def _drain(
        self,
        stream: _StreamIter,
        context: ConversationContext,
        outcome: _StreamOutcome,
    ) -> BaseException | None:
        """Consume stream events into `outcome` until terminal or input_required.

        Returns the connection error encountered (if any) so the caller can
        decide whether to reconnect.
        """
        try:
            async for event in stream:
                if isinstance(event, Message):
                    continue
                task, update = event
                outcome.task = task
                if isinstance(update, TaskArtifactUpdateEvent):
                    for chunk in task_artifact_update_to_chunks(update):
                        outcome.text += chunk.content
                        await context.send(chunk)
                if task.status.state in _TERMINAL_STATES:
                    return None
                if task.status.state == TaskState.input_required:
                    outcome.input_required = True
                    outcome.input_prompt = _input_prompt(task)
                    return None
            return None
        except (httpx.HTTPError, A2AClientHTTPError) as exc:
            return exc


def _needs_reconnect(outcome: _StreamOutcome) -> bool:
    return outcome.task is not None and outcome.task.status.state not in _TERMINAL_STATES and not outcome.input_required


def _last_model_request(messages: Sequence[BaseEvent]) -> ModelRequest | None:
    for ev in reversed(messages):
        if isinstance(ev, ModelRequest):
            return ev
    return None


def _input_prompt(task: Task) -> str | None:
    msg = task.status.message
    return text_from_message(msg) if msg is not None else None


def _final_text(task: Task) -> str:
    return "".join(artifact_text(a) for a in task.artifacts or ())
