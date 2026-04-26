# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard
from starlette.applications import Starlette

from autogen.beta import Agent
from autogen.beta.a2a import A2AServer, AgentExecutor
from autogen.beta.testing import TestConfig


def _agent() -> Agent:
    return Agent("specialist", "be helpful", config=TestConfig("ok"))


def test_default_card_built_from_agent() -> None:
    server = A2AServer(_agent(), url="http://localhost:8000")

    assert server.card.name == "specialist"
    assert server.card.url == "http://localhost:8000"


def test_custom_card_is_used() -> None:
    custom = AgentCard(
        name="override",
        description="custom",
        url="http://elsewhere",
        version="2.0.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
    )

    server = A2AServer(_agent(), card=custom)

    assert server.card is custom


def test_executor_wraps_agent() -> None:
    server = A2AServer(_agent())

    assert isinstance(server.executor, AgentExecutor)


def test_build_asgi_returns_starlette_app() -> None:
    server = A2AServer(_agent())

    app = server.build_asgi()

    assert isinstance(app, Starlette)


def test_custom_task_store_passes_through() -> None:
    store = InMemoryTaskStore()
    server = A2AServer(_agent(), task_store=store)

    app = server.build_asgi()  # should not raise

    assert isinstance(app, Starlette)
