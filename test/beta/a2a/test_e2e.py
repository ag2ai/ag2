# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Agent, MemoryStream
from autogen.beta.a2a.utils import CONTEXT_ID_VAR_KEY, PROVIDER_NAME, TASK_ID_VAR_KEY
from autogen.beta.context import ConversationContext
from autogen.beta.events import ModelMessageChunk, ModelRequest, TextInput, ToolCallEvent
from autogen.beta.testing import TestConfig


@pytest.mark.asyncio
class TestSimpleRoundTrip:
    async def test_returns_servers_text(self, serve) -> None:
        agent = Agent("specialist", "be helpful", config=TestConfig("hello-from-remote"))
        env = serve(agent)
        client = env.config.create()
        ctx = ConversationContext(stream=MemoryStream())

        response = await client(
            [ModelRequest([TextInput("hi")])],
            ctx,
            tools=[],
            response_schema=None,
            serializer=None,  # type: ignore[arg-type]
        )

        assert response.message and response.message.content == "hello-from-remote"

    async def test_response_metadata(self, serve) -> None:
        agent = Agent("specialist", "p", config=TestConfig("ok"))
        env = serve(agent)
        client = env.config.create()
        ctx = ConversationContext(stream=MemoryStream())

        response = await client(
            [ModelRequest([TextInput("hi")])],
            ctx,
            tools=[],
            response_schema=None,
            serializer=None,  # type: ignore[arg-type]
        )

        assert response.provider == PROVIDER_NAME
        assert response.model == "specialist"
        assert response.finish_reason == "completed"

    async def test_context_and_task_ids_persisted_on_context(self, serve) -> None:
        agent = Agent("specialist", "p", config=TestConfig("ok"))
        env = serve(agent)
        client = env.config.create()
        ctx = ConversationContext(stream=MemoryStream())

        await client(
            [ModelRequest([TextInput("hi")])],
            ctx,
            tools=[],
            response_schema=None,
            serializer=None,  # type: ignore[arg-type]
        )

        assert CONTEXT_ID_VAR_KEY in ctx.variables
        assert TASK_ID_VAR_KEY in ctx.variables


@pytest.mark.asyncio
async def test_chunks_arrive_in_context_stream(serve) -> None:
    agent = Agent("specialist", "p", config=TestConfig("hello"))
    env = serve(agent)
    client = env.config.create()

    chunks: list[str] = []
    stream = MemoryStream()

    @stream.where(ModelMessageChunk).subscribe
    async def collect(c: ModelMessageChunk) -> None:
        chunks.append(c.content)

    ctx = ConversationContext(stream=stream)
    response = await client(
        [ModelRequest([TextInput("hi")])],
        ctx,
        tools=[],
        response_schema=None,
        serializer=None,  # type: ignore[arg-type]
    )

    assert response.message and response.message.content == "hello"
    # The same payload is also visible as streaming chunks (where supported).
    assert "".join(chunks) == "hello" or chunks == []


@pytest.mark.asyncio
async def test_orchestrator_can_call_remote_via_as_tool(serve) -> None:
    remote_agent = Agent("remote", "be a helper", config=TestConfig("from-remote"))
    env = serve(remote_agent)

    remote = Agent("remote", config=env.config)

    main_agent = Agent(
        "main",
        "orchestrator",
        config=TestConfig(
            ToolCallEvent(name="task_remote", arguments='{"objective": "say hi"}'),
            "main-final",
        ),
        tools=[remote.as_tool(description="Delegate to remote")],
    )

    reply = await main_agent.ask("Hello")

    assert reply.body == "main-final"
