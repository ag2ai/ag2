# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from freezegun import freeze_time
from httpx import ASGITransport, AsyncClient

from autogen.remote import HTTPAgentBus, HTTPRemoteAgent
from autogen.remote.errors import RemoteAgentNotFoundError
from autogen.remote.protocol import AgentBusMessage
from autogen.remote.runtime import RemoteService


class TestService(RemoteService):
    def __init__(self, reply: AgentBusMessage | None = None) -> None:
        self.name = "test"
        self.reply = reply

    async def __call__(self, state: AgentBusMessage) -> AgentBusMessage | None:
        return self.reply


@pytest.mark.asyncio()
async def test_client() -> None:
    reply_message = {"role": "assistant", "content": "Hello, world!"}

    remote_app = HTTPAgentBus(
        additional_services=[TestService(AgentBusMessage(messages=[reply_message]))],
    )

    client = AsyncClient(transport=ASGITransport(remote_app))

    remote_agent = HTTPRemoteAgent(
        url="http://localhost:8000",
        name="test",
        client=client,
    )

    _, reply = await remote_agent.a_generate_remote_reply()
    assert reply == reply_message


@pytest.mark.asyncio()
async def test_agent_not_found() -> None:
    client = AsyncClient(transport=ASGITransport(HTTPAgentBus()))

    remote_agent = HTTPRemoteAgent(
        url="http://localhost:8000",
        name="wrong",
        client=client,
    )

    with pytest.raises(RemoteAgentNotFoundError, match="Remote agent wrong not found"):
        await remote_agent.a_generate_remote_reply()


@pytest.mark.asyncio()
@freeze_time(auto_tick_seconds=5)
async def test_long_living_agent_task() -> None:
    class SlowService(TestService):
        async def __call__(self, state: AgentBusMessage) -> AgentBusMessage | None:
            await asyncio.sleep(10)
            return self.reply

    reply_message = {"role": "assistant", "content": "Hello, world!"}
    remote_app = HTTPAgentBus(
        additional_services=[SlowService(AgentBusMessage(messages=[reply_message]))],
        long_polling_interval=1,
    )

    client = AsyncClient(transport=ASGITransport(remote_app))

    remote_agent = HTTPRemoteAgent(
        url="http://localhost:8000",
        name="test",
        client=client,
    )

    _, reply = await remote_agent.a_generate_remote_reply()
    assert reply == reply_message
