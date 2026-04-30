# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard
from starlette.applications import Starlette

from autogen.beta import Agent
from autogen.beta.a2a import A2AServer, AgentExecutor
from autogen.beta.testing import TestConfig


@pytest.fixture
def agent() -> Agent:
    return Agent("specialist", "be helpful", config=TestConfig("ok"))


def _make_card(
    *, name: str = "override", url: str = "http://elsewhere", version: str = "2.0.0", streaming: bool = True
) -> AgentCard:
    return AgentCard(
        name=name,
        description="custom",
        url=url,
        version=version,
        capabilities=AgentCapabilities(streaming=streaming),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
    )


class TestA2AServer:
    def test_default_card_built_from_agent(self, agent: Agent) -> None:
        server = A2AServer(agent, url="http://localhost:8000")

        assert server.card.name == "specialist"
        assert server.card.url == "http://localhost:8000"

    def test_custom_card_is_used(self, agent: Agent) -> None:
        custom = _make_card()

        server = A2AServer(agent, card=custom)

        assert server.card is custom

    def test_executor_wraps_agent(self, agent: Agent) -> None:
        server = A2AServer(agent)

        assert isinstance(server.executor, AgentExecutor)

    def test_build_asgi_returns_starlette_app(self, agent: Agent) -> None:
        server = A2AServer(agent)

        app = server.build_asgi()

        assert isinstance(app, Starlette)

    def test_custom_task_store_passes_through(self, agent: Agent) -> None:
        store = InMemoryTaskStore()
        server = A2AServer(agent, task_store=store)

        app = server.build_asgi()

        assert isinstance(app, Starlette)

    def test_extended_card_auto_sets_public_supports_flag(self, agent: Agent) -> None:
        extended = _make_card(name="specialist-extended", url="http://localhost:8000", version="0.1.0")

        server = A2AServer(agent, extended_card=extended)

        assert server.card.supports_authenticated_extended_card is True
        assert server.extended_card is extended

    def test_extended_card_optional(self, agent: Agent) -> None:
        server = A2AServer(agent)

        assert server.extended_card is None
        assert not server.card.supports_authenticated_extended_card

    def test_extended_card_with_user_card_lacking_flag_warns(self, agent: Agent) -> None:
        public = _make_card(name="public", url="http://test", version="0.1.0")
        extended = _make_card(name="extended", url="http://test", version="0.1.0")

        with pytest.warns(RuntimeWarning, match="extended_card was provided"):
            A2AServer(agent, card=public, extended_card=extended)
