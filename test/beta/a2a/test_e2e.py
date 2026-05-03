# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Agent, MemoryStream
from autogen.beta.events import ModelMessageChunk, ToolCallEvent
from autogen.beta.testing import TestConfig


@pytest.mark.asyncio
async def test_simple_round_trip_returns_servers_text(serve) -> None:
    server_agent = Agent("specialist", "be helpful", config=TestConfig("hello-from-remote"))
    config = serve(server_agent)

    reply = await Agent("client", "p", config=config).ask("hi")

    assert reply.body == "hello-from-remote"


@pytest.mark.asyncio
async def test_response_metadata_describes_remote_server(serve) -> None:
    config = serve(Agent("specialist", "p", config=TestConfig("ok")))

    reply = await Agent("client", "p", config=config).ask("hi")

    assert (reply.response.provider, reply.response.model, reply.response.finish_reason) == (
        "a2a",
        "specialist",
        "stop",
    )


@pytest.mark.asyncio
async def test_streaming_chunks_arrive_in_client_stream(serve) -> None:
    config = serve(Agent("specialist", "p", config=TestConfig("hello")))

    chunks: list[str] = []
    stream = MemoryStream()

    @stream.where(ModelMessageChunk).subscribe
    async def collect(chunk: ModelMessageChunk) -> None:
        chunks.append(chunk.content)

    reply = await Agent("client", "p", config=config).ask("hi", stream=stream)

    assert reply.body == "hello"
    assert "".join(chunks) == "hello"


@pytest.mark.asyncio
async def test_remote_returns_tool_call_for_client_declared_tool(serve) -> None:
    server_agent = Agent(
        "specialist",
        "p",
        config=TestConfig(
            ToolCallEvent(name="ping", arguments="{}"),
            "done",
        ),
    )
    config = serve(server_agent)

    invoked: list[str] = []
    client_agent = Agent("client", "p", config=config)

    @client_agent.tool
    def ping() -> str:
        invoked.append("called")
        return "pong"

    reply = await client_agent.ask("hi")

    assert reply.body == "done"
    assert invoked == ["called"]
