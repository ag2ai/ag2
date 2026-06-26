# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import httpx
import pytest
from a2a.server.agent_execution import AgentExecutor as A2AAgentExecutorBase
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryPushNotificationConfigStore, TaskUpdater
from a2a.types import Task, TaskState, TaskStatus
from typing_extensions import Self

from ag2 import Agent, Context
from ag2.a2a import A2AConfig, A2AServer, build_card
from ag2.a2a.client import A2AClient
from ag2.a2a.errors import A2ATaskFailedError
from ag2.a2a.testing import make_test_client_factory
from ag2.config.client import LLMClient
from ag2.config.config import ModelConfig
from ag2.events import BaseEvent, ModelMessage, ModelMessageChunk, ModelResponse
from ag2.testing import TestConfig


class _SpyAsyncClient(httpx.AsyncClient):
    aclose_count = 0

    async def aclose(self) -> None:
        type(self).aclose_count += 1
        await super().aclose()


def _make_spy_factory(server: A2AServer, url: str):
    transport = httpx.ASGITransport(app=server.build_jsonrpc(url=url))

    def factory() -> httpx.AsyncClient:
        return _SpyAsyncClient(transport=transport, base_url=url, timeout=30.0)

    return factory


class _AlwaysFailingExecutor(A2AAgentExecutorBase):
    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        msg = request_context.message
        assert msg is not None
        task_id = msg.task_id or uuid4().hex
        context_id = msg.context_id or uuid4().hex
        await event_queue.enqueue_event(
            Task(id=task_id, context_id=context_id, status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED)),
        )
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()
        await updater.failed()

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        return None


@pytest.mark.asyncio
class TestHttpxLifecycle:
    async def test_client_closes_httpx_after_successful_ask(self) -> None:
        _SpyAsyncClient.aclose_count = 0
        server = A2AServer(Agent("server", config=TestConfig("hi")))
        url = "http://test"
        client = Agent("client", config=A2AConfig(card_url=url, httpx_client_factory=_make_spy_factory(server, url)))

        await client.ask("ping")

        assert _SpyAsyncClient.aclose_count >= 1

    async def test_client_closes_httpx_when_task_fails(self) -> None:
        _SpyAsyncClient.aclose_count = 0
        server = A2AServer(
            Agent("stub", config=TestConfig("unused")),
            executor=_AlwaysFailingExecutor(),
        )
        url = "http://test"
        client = Agent(
            "client",
            config=A2AConfig(card_url=url, httpx_client_factory=_make_spy_factory(server, url), streaming=False),
        )

        with pytest.raises(A2ATaskFailedError):
            await client.ask("ping")

        assert _SpyAsyncClient.aclose_count >= 1

    async def test_aclose_is_idempotent(self) -> None:
        _SpyAsyncClient.aclose_count = 0
        client = A2AClient(card_url="http://test")
        client._httpx_client = _SpyAsyncClient(base_url="http://test")

        await client.aclose()
        await client.aclose()

        assert _SpyAsyncClient.aclose_count == 1


class _ChunkingScript(ModelConfig):
    def __init__(self, chunks: Sequence[str]) -> None:
        self._chunks = list(chunks)

    def copy(self) -> Self:
        return self

    def create(self) -> "_ChunkingScriptClient":
        return _ChunkingScriptClient(self._chunks)

    def create_files_client(self) -> None:
        raise NotImplementedError


class _ChunkingScriptClient(LLMClient):
    def __init__(self, chunks: Sequence[str]) -> None:
        self._chunks = list(chunks)

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        **_: Any,
    ) -> ModelResponse:
        for chunk in self._chunks:
            await context.send(ModelMessageChunk(chunk))
        full = "".join(self._chunks)
        return ModelResponse(message=ModelMessage(full))


@pytest.mark.asyncio
async def test_streamed_chunks_not_duplicated_in_final_message() -> None:
    server = A2AServer(Agent("server", config=_ChunkingScript(["he", "llo"])))
    url = "http://test"
    factory = make_test_client_factory(server, url=url)
    client = Agent(
        "client",
        config=A2AConfig(card_url=url, httpx_client_factory=factory, streaming=True),
    )

    reply = await client.ask("ping")

    assert reply.response.content == "hello"


@pytest.mark.asyncio
class TestCardImmutability:
    async def test_build_jsonrpc_does_not_mutate_input_card(self) -> None:
        agent = Agent("server", config=TestConfig("hi"))
        card = build_card(agent, url="http://test", transports=("jsonrpc",))
        before_extended = card.capabilities.extended_agent_card
        before_push = card.capabilities.push_notifications

        server = A2AServer(
            agent,
            extended_card=card,
            push_config_store=InMemoryPushNotificationConfigStore(),
        )
        server.build_jsonrpc(url="http://test", card=card)

        assert card.capabilities.extended_agent_card == before_extended
        assert card.capabilities.push_notifications == before_push

    async def test_build_rest_does_not_mutate_input_card(self) -> None:
        agent = Agent("server", config=TestConfig("hi"))
        card = build_card(agent, url="http://test", transports=("rest",))
        before_extended = card.capabilities.extended_agent_card
        before_push = card.capabilities.push_notifications

        server = A2AServer(
            agent,
            extended_card=card,
            push_config_store=InMemoryPushNotificationConfigStore(),
        )
        server.build_rest(url="http://test", card=card)

        assert card.capabilities.extended_agent_card == before_extended
        assert card.capabilities.push_notifications == before_push

    async def test_build_grpc_does_not_mutate_input_card(self) -> None:
        agent = Agent("server", config=TestConfig("hi"))
        card = build_card(agent, url="http://test", transports=("grpc",), grpc_url="localhost:50051")
        before_extended = card.capabilities.extended_agent_card
        before_push = card.capabilities.push_notifications

        server = A2AServer(
            agent,
            extended_card=card,
            push_config_store=InMemoryPushNotificationConfigStore(),
        )
        server.build_grpc(bind="localhost:0", grpc_url="localhost:50051", card=card)

        assert card.capabilities.extended_agent_card == before_extended
        assert card.capabilities.push_notifications == before_push
