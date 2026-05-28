# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta import Agent, observer
from autogen.beta.events import ModelResponse
from autogen.beta.testing import TestConfig


async def check_agent_constructor_with_observers() -> None:
    @observer(ModelResponse)
    def on_response(event: ModelResponse) -> None:
        pass

    Agent(
        "test",
        config=TestConfig(),
        observers=[on_response],
    )


async def check_any_event_observer() -> None:
    @observer()
    def on_response(event: ModelResponse) -> None:
        pass

    Agent(
        "test",
        config=TestConfig(),
        observers=[on_response],
    )


async def check_agent_constructor_with_direct_observers() -> None:
    Agent(
        "test",
        config=TestConfig(),
        observers=[observer(ModelResponse, lambda e: None)],
    )


async def check_agent_ask_with_observers() -> None:
    agent = Agent("test", config=TestConfig())

    await agent.ask(
        "Hi!",
        observers=[observer(ModelResponse, lambda e: None)],
    )


async def check_agent_turn_ask_with_observers() -> None:
    agent = Agent("test", config=TestConfig())

    turn = await agent.ask("Hi!")

    await turn.ask(
        "More!",
        observers=[observer(ModelResponse, lambda e: None)],
    )


async def check_agent_observer_decorator() -> None:
    agent = Agent("test", config=TestConfig())

    @agent.observer(ModelResponse)
    def on_response(event: ModelResponse) -> None:
        pass


async def check_any_event_agent_observer_decorator() -> None:
    agent = Agent("test", config=TestConfig())

    @agent.observer()
    def on_response(event: ModelResponse) -> None:
        pass


async def check_agent_observer_direct() -> None:
    agent = Agent("test", config=TestConfig())

    def on_response(event: ModelResponse) -> None:
        pass

    agent.observer(ModelResponse, on_response)
