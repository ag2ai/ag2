# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict
from httpx import ASGITransport

from autogen import ConversableAgent
from autogen.remote import HTTPAgentBus, HTTPRemoteAgent, HttpxClientFactory
from autogen.testing import TestAgent


@pytest.mark.asyncio()
async def test_sequential_chat() -> None:
    # arrange remote side
    remote_agent1 = ConversableAgent("remote-1")
    remote_agent2 = ConversableAgent("remote-2")

    asgi_app = HTTPAgentBus(agents=[remote_agent1, remote_agent2])
    client = HttpxClientFactory(transport=ASGITransport(asgi_app))

    # arrange local side
    remote_agent1_mirror = HTTPRemoteAgent(url="http://memory", name="remote-1", client=client)
    remote_agent2_mirror = HTTPRemoteAgent(url="http://memory", name="remote-2", client=client)
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
