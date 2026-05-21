# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict
from httpx import ASGITransport
from starlette.applications import Starlette
from starlette.routing import Mount

from autogen import ConversableAgent
from autogen.a2a import A2aAgentServer, A2aRemoteAgent, HttpxClientFactory
from autogen.testing import TestAgent


@pytest.mark.asyncio()
async def test_sequential_chat() -> None:
    # arrange remote side
    remote_agent1 = ConversableAgent("remote-1")
    remote_agent2 = ConversableAgent("remote-2")

    a2a_asgi_app = Starlette(
        routes=[
            Mount("/one", A2aAgentServer(remote_agent1, url="http://memory/one/").build()),
            Mount("/two", A2aAgentServer(remote_agent2, url="http://memory/two/").build()),
        ]
    )

    a2a_client = HttpxClientFactory(transport=ASGITransport(a2a_asgi_app))

    # arrange local side
    remote_agent1_mirror = A2aRemoteAgent(url="http://memory/one/", name="remote1-mirror", client=a2a_client)
    remote_agent2_mirror = A2aRemoteAgent(url="http://memory/two/", name="remote2-mirror", client=a2a_client)
    local_agent = ConversableAgent("local")

    # act
    with (
        TestAgent(remote_agent1, ["Hi, I am remote agent one!"]),
        TestAgent(remote_agent2, ["Hi, I am remote agent two!"]),
        TestAgent(local_agent, ["Hi, I am local agent!"]),
    ):
        chat_results = await local_agent.a_initiate_chats([
            {
                "recipient": remote_agent1_mirror,
                "message": "Hi agent!",
                "max_turns": 1,
                "chat_id": "some-chat",
            },
            {
                "recipient": remote_agent2_mirror,
                "message": "Hi agent!",
                "max_turns": 1,
                "chat_id": "some-chat2",
            },
        ])

    # assert
    # use IsPartialDict because other fields has no matter for our test
    assert chat_results["some-chat"].chat_history == [
        IsPartialDict({"content": "Hi agent!"}),
        IsPartialDict({"content": "Hi, I am remote agent one!", "name": "remote-1"}),
    ]

    assert chat_results["some-chat2"].chat_history == [
        IsPartialDict({"content": "Hi agent!"}),
        IsPartialDict({"content": "Hi, I am remote agent two!", "name": "remote-2"}),
    ]
