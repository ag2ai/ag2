# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Agent
from autogen.beta.events import TextInput
from autogen.beta.testing import TestConfig


@pytest.mark.asyncio()
async def test_agent_ask_accepts_text_input() -> None:
    agent = Agent("", config=TestConfig("result"))

    reply = await agent.ask(TextInput("Hi!"))

    assert reply.body == "result"


@pytest.mark.asyncio()
async def test_turn_ask_accepts_text_input() -> None:
    agent = Agent("", config=TestConfig("first", "second"))

    reply = await agent.ask("Hi!")
    reply = await reply.ask(TextInput("Follow up"))

    assert reply.body == "second"
