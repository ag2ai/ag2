# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen import ConversableAgent
from autogen.remote.agent_service import AgentService
from autogen.remote.protocol import RequestMessage, ResponseMessage
from autogen.testing import TestAgent


@pytest.mark.asyncio
async def test_smoke() -> None:
    agent = ConversableAgent("test-agent")
    service = AgentService(agent)

    with TestAgent(
        agent,
        ["Hi, I am remote agent!"],
    ):
        result = await service(RequestMessage(messages=[{"content": "Hi agent!"}]))

        assert result == ResponseMessage(
            messages=[
                {
                    "content": "Hi, I am remote agent!",
                    "role": "assistant",
                    "name": "test-agent",
                }
            ]
        )
