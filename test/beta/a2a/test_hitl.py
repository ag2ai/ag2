# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from autogen.beta import Agent, Context, MemoryStream
from autogen.beta.context import ConversationContext
from autogen.beta.events import (
    HumanInputRequest,
    HumanMessage,
    ModelRequest,
    TextInput,
    ToolCallEvent,
)
from autogen.beta.testing import TestConfig


def _server_agent_with_confirm() -> Agent:
    agent = Agent(
        "specialist",
        "p",
        config=TestConfig(
            ToolCallEvent(name="confirm", arguments="{}"),
            "after-confirm",
        ),
    )

    @agent.tool
    async def confirm(context: Context) -> str:
        return await context.input("Confirm action?")

    return agent


@pytest.mark.asyncio
class TestHITLForwardingE2E:
    async def test_client_receives_input_request_and_completes(self, serve) -> None:
        env = serve(_server_agent_with_confirm())
        client = env.config.create()

        received_requests: list[str] = []
        stream = MemoryStream()

        @stream.where(HumanInputRequest).subscribe
        async def auto_reply(req: HumanInputRequest, context: Context) -> None:
            received_requests.append(req.content)
            await context.send(HumanMessage("yes-i-confirm", parent_id=req.id))

        ctx = ConversationContext(stream=stream)
        response = await asyncio.wait_for(
            client(
                [ModelRequest([TextInput("please confirm")])],
                ctx,
                tools=[],
                response_schema=None,
                serializer=None,  # type: ignore[arg-type]
            ),
            timeout=5.0,
        )

        assert received_requests == ["Confirm action?"]
        assert response.message is not None
        assert response.message.content == "after-confirm"
        assert response.finish_reason == "completed"

    async def test_local_hitl_hook_is_ignored_when_routed_over_a2a(self, serve) -> None:
        # The server agent has NO hitl_hook — A2A is the only path.
        # The client receives the request and replies; the agent finishes.
        env = serve(_server_agent_with_confirm())
        client = env.config.create()

        stream = MemoryStream()

        @stream.where(HumanInputRequest).subscribe
        async def auto_reply(req: HumanInputRequest, context: Context) -> None:
            await context.send(HumanMessage("ok", parent_id=req.id))

        ctx = ConversationContext(stream=stream)
        response = await asyncio.wait_for(
            client(
                [ModelRequest([TextInput("hi")])],
                ctx,
                tools=[],
                response_schema=None,
                serializer=None,  # type: ignore[arg-type]
            ),
            timeout=5.0,
        )

        assert response.message is not None
        assert response.message.content == "after-confirm"
