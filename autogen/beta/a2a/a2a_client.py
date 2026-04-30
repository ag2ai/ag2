# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterable, Sequence
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
from a2a.utils.constants import EXTENDED_AGENT_CARD_PATH, PREV_AGENT_CARD_WELL_KNOWN_PATH
from fast_depends.library.serializer import SerializerProto

from autogen.beta.config.client import LLMClient
from autogen.beta.context import ConversationContext
from autogen.beta.events import BaseEvent, ModelMessage, ModelRequest, ModelResponse
from autogen.beta.response import ResponseProto
from autogen.beta.tools.schemas import ToolSchema

from .errors import (
    A2AAuthRequiredError,
    A2AClientToolsNotSupportedError,
    A2ANoTaskError,
    A2AReconnectError,
    A2ATaskFailedError,
    A2ATaskRejectedError,
)
from .mappers import (
    artifact_text,
    followup_user_message,
    model_request_to_a2a_message,
    task_artifact_update_to_chunks,
    text_from_message,
)
from .types import TERMINAL_TASK_STATES, TRANSPORT_ERRORS, HttpxClientFactory, StreamOutcome
from .utils import CONTEXT_ID_VAR_KEY, PROVIDER_NAME, TASK_ID_VAR_KEY


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
        if list(tools) or response_schema is not None:
            raise A2AClientToolsNotSupportedError()

        request = _last_model_request(messages)
        if request is None:
            raise ValueError("A2AClient requires at least one ModelRequest in messages")

        card, client = await self._ensure_client()
        message = model_request_to_a2a_message(request, context_id=context.variables.get(CONTEXT_ID_VAR_KEY))

        accumulated = ""
        outcome: StreamOutcome
        while True:
            outcome = await self._exchange(client, message, context, accumulated)
            accumulated = outcome.text

            if outcome.input_required and outcome.task is not None:
                user_text = await context.input(outcome.input_prompt or "")
                message = followup_user_message(user_text, context_id=outcome.task.context_id, task_id=outcome.task.id)
                continue
            break

        if outcome.task is None:
            raise A2ANoTaskError()

        if outcome.task.status.state == TaskState.failed:
            raise A2ATaskFailedError(outcome.task)
        if outcome.task.status.state == TaskState.rejected:
            raise A2ATaskRejectedError(outcome.task)

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
        """Close the httpx client if we created it ourselves.

        When the user supplies a `client_factory`, lifecycle is theirs.
        The cached `AgentCard` is preserved — instantiate a new `A2AClient`
        if you need to refresh it.
        """
        if self._httpx_client is not None and self._client_factory is None:
            await self._httpx_client.aclose()
        self._httpx_client = None
        self._a2a_client = None

    async def _ensure_client(self) -> tuple[AgentCard, Client]:
        async with self._init_lock:
            if self._httpx_client is None:
                self._httpx_client = self._client_factory() if self._client_factory else httpx.AsyncClient()
            if self._agent_card is None:
                self._agent_card = await self._fetch_card()
            if self._a2a_client is None:
                self._a2a_client = self._build_a2a_client(self._agent_card)
            return self._agent_card, self._a2a_client

    async def _fetch_card(self) -> AgentCard:
        """Resolve the AgentCard, upgrading to the extended one when offered.

        Falls back to the legacy `/.well-known/agent.json` path on 404 — old
        a2a-sdk (0.2.x) servers serve the card there. If the public card
        advertises `supports_authenticated_extended_card`, the spec's
        authenticated endpoint is queried; failures (e.g. unauth) fall back
        to the public card silently.
        """
        if self._httpx_client is None:
            raise RuntimeError("_fetch_card called before httpx client was initialised")
        resolver = A2ACardResolver(self._httpx_client, self._url)
        try:
            card = await resolver.get_agent_card()
        except A2AClientHTTPError as exc:
            if exc.status_code != 404:
                raise
            card = await resolver.get_agent_card(relative_card_path=PREV_AGENT_CARD_WELL_KNOWN_PATH)
        if card.supports_authenticated_extended_card:
            with contextlib.suppress(A2AClientHTTPError):
                card = await resolver.get_agent_card(relative_card_path=EXTENDED_AGENT_CARD_PATH)
        return card

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
    ) -> StreamOutcome:
        """Send one message and bring the task to a terminal/HITL state.

        Two paths depending on `card.capabilities.streaming`:

        - Streaming: SSE stream from `send_message`; on drop, `resubscribe`
          up to `max_reconnects` times.
        - Polling: `send_message` returns the initial task in one shot;
          `get_task` is then polled with `reconnect_backoff` sleep until
          terminal/HITL or `max_reconnects` attempts are exhausted.
        """
        outcome = StreamOutcome(text=seed_text)
        last_error = await self._drain(client.send_message(message), context, outcome)

        streaming = bool(self._agent_card and self._agent_card.capabilities.streaming)
        attempts = 0
        while _needs_reconnect(outcome) and attempts < self._max_reconnects:
            assert outcome.task is not None
            attempts += 1
            try:
                if streaming:
                    last_error = await self._drain(
                        client.resubscribe(TaskIdParams(id=outcome.task.id)), context, outcome
                    )
                else:
                    await asyncio.sleep(self._reconnect_backoff)
                    outcome.task = await client.get_task(TaskQueryParams(id=outcome.task.id))
                    if outcome.task.status.state == TaskState.auth_required:
                        raise A2AAuthRequiredError(outcome.task)
            except TRANSPORT_ERRORS as exc:
                last_error = exc
                await asyncio.sleep(self._reconnect_backoff)

        if _needs_reconnect(outcome):
            raise A2AReconnectError(attempts=attempts, last_error=last_error)

        if outcome.task is not None and not outcome.input_required and not outcome.text:
            assert self._a2a_client is not None
            outcome.task = await self._a2a_client.get_task(TaskQueryParams(id=outcome.task.id))

        return outcome

    async def _drain(
        self,
        stream: AsyncIterator[tuple[Task, Any] | Message],
        context: ConversationContext,
        outcome: StreamOutcome,
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
                if task.status.state == TaskState.auth_required:
                    raise A2AAuthRequiredError(task)
                if task.status.state in TERMINAL_TASK_STATES:
                    return None
                if task.status.state == TaskState.input_required:
                    outcome.input_required = True
                    outcome.input_prompt = _input_prompt(task)
                    return None
            return None
        except TRANSPORT_ERRORS as exc:
            return exc


def _needs_reconnect(outcome: StreamOutcome) -> bool:
    return (
        outcome.task is not None
        and outcome.task.status.state not in TERMINAL_TASK_STATES
        and not outcome.input_required
    )


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
