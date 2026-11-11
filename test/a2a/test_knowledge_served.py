# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""An agent with knowledge works the same when served over A2A as when called directly."""

import pytest

from ag2 import Agent, KnowledgeConfig
from ag2.a2a import A2AConfig, A2AServer
from ag2.a2a.testing import make_test_client_factory
from ag2.knowledge import KnowledgePlugin, MemoryKnowledgeStore
from ag2.testing import TestConfig

SERVER_URL = "http://test"


def _client_for(served: Agent) -> Agent:
    return Agent(
        "client-agent",
        config=A2AConfig(
            card_url=SERVER_URL,
            httpx_client_factory=make_test_client_factory(A2AServer(served), url=SERVER_URL),
            streaming=False,
        ),
    )


@pytest.mark.asyncio
async def test_served_agent_with_knowledge_plugin_answers_and_initializes_its_store() -> None:
    store = MemoryKnowledgeStore()
    librarian = Agent("librarian", config=TestConfig("served"), plugins=[KnowledgePlugin(store)])

    reply = await _client_for(librarian).ask("hi")

    assert reply.body == "served"
    assert await store.read("/.initialized") == "librarian"


@pytest.mark.asyncio
@pytest.mark.filterwarnings(r"ignore:Agent\(knowledge=:DeprecationWarning")
async def test_served_agent_with_legacy_knowledge_answers_and_initializes_its_store() -> None:
    store = MemoryKnowledgeStore()
    librarian = Agent("librarian", config=TestConfig("served"), knowledge=KnowledgeConfig(store))

    reply = await _client_for(librarian).ask("hi")

    assert reply.body == "served"
    assert await store.read("/.initialized") == "librarian"
