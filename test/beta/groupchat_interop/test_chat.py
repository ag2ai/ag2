# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime

import pytest
from dirty_equals import IsPartialDict

from autogen import ConversableAgent
from autogen.beta import Agent, events, testing
from autogen.testing import TestAgent, ToolCall


@pytest.mark.asyncio()
async def test_initiate_chat() -> None:
    # arrange agents
    def get_weekday(date_string: str) -> str:
        return datetime.strptime(date_string, "%Y-%m-%d").strftime("%A")

    beta_agent = Agent(
        "agent",
        config=testing.TestConfig(
            events.ToolCallEvent(name="get_weekday", arguments='{"date_string":"2025-11-07"}'),
            "Friday",
        ),
        tools=[get_weekday],
    ).as_conversable()

    conversable_agent = ConversableAgent("user")

    with (
        TestAgent(
            conversable_agent,
            [ToolCall("get_weekday", date_string="2025-11-07").to_message(), "Friday"],
        ),
        TestAgent(conversable_agent),
    ):
        result = await conversable_agent.a_initiate_chat(
            recipient=beta_agent,
            message="What day is 2025-11-07?",
            max_turns=1,
        )

    # assert
    assert result.chat_history == [
        IsPartialDict({"content": "What day is 2025-11-07?"}),
        IsPartialDict({"content": "Friday"}),
    ]
