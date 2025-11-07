# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest
from dirty_equals import IsPartialDict
from httpx import ASGITransport

from autogen import ConversableAgent, LLMConfig
from autogen.a2a import A2aAgentServer, A2aRemoteAgent, HttpxClientFactory


@pytest.mark.asyncio()
async def test_get_remote_human_input() -> None:
    # create an AssistantAgent instance named "assistant"
    remote_agent = ConversableAgent(
        "remote",
        human_input_mode="ALWAYS",
        llm_config=LLMConfig({"model": "gpt-5", "api_key": "wrong-key"}),
    )

    a2a_asgi_app = A2aAgentServer(remote_agent).build()
    a2a_client = HttpxClientFactory(transport=ASGITransport(a2a_asgi_app))

    # arrange local side
    remote_agent_mirror = A2aRemoteAgent(url="http://memory/", name="remote-mirror", client=a2a_client)
    user_agent = ConversableAgent("user")

    # act
    mock = remote_agent_mirror.a_get_human_input = AsyncMock(return_value="exit")
    result = await user_agent.a_initiate_chat(
        recipient=remote_agent_mirror,
        message="Hi!",
        max_turns=1,
    )

    # assert
    mock.assert_awaited_once()
    assert result.chat_history == [IsPartialDict({"content": "Hi!"})]
