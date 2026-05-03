# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from autogen.beta import Agent, Context
from autogen.beta.events import ToolCallEvent
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
async def test_remote_input_request_is_satisfied_by_client_hitl_hook(serve) -> None:
    config = serve(_server_agent_with_confirm())

    async def auto_reply() -> str:
        return "ok"

    client_agent = Agent("client", "p", config=config)
    reply = await asyncio.wait_for(client_agent.ask("hi", hitl_hook=auto_reply), timeout=5.0)

    assert reply.body == "after-confirm"
