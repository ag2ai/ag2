# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from a2a.server.tasks import InMemoryTaskStore, TaskStore

from ag2 import Agent
from ag2.a2a import A2AConfig, A2AServer
from ag2.a2a.tasks import list_tasks
from ag2.a2a.testing import make_test_client_factory
from ag2.testing import TestConfig

URL = "http://test"


def test_default_task_store_is_materialised_eagerly() -> None:
    server = A2AServer(Agent("a", config=TestConfig("hi")))

    assert isinstance(server.task_store, TaskStore)


def test_user_task_store_is_preserved() -> None:
    custom = InMemoryTaskStore()

    server = A2AServer(Agent("a", config=TestConfig("hi")), task_store=custom)

    assert server.task_store is custom


def test_building_transports_does_not_swap_the_store() -> None:
    # `build_grpc` is left out on purpose: it binds a real port that nothing
    # here would close. The HTTP builders make the same point.
    server = A2AServer(Agent("a", config=TestConfig("hi")))
    store = server.task_store

    server.build_jsonrpc(url=URL)
    server.build_rest(url=URL)

    assert server.task_store is store


@pytest.mark.asyncio
async def test_a_task_created_through_one_app_is_visible_through_another() -> None:
    # The point of materialising the store eagerly. Each factory call builds
    # its own ASGI app off the same server; if the builders defaulted to a
    # store apiece, the second app would report no tasks at all. Asserting on
    # `server.task_store` alone cannot see that — the wiring has to be
    # observed through a second app.
    server = A2AServer(Agent("server", config=TestConfig("hi")), task_store=InMemoryTaskStore())
    first = A2AConfig(card_url=URL, httpx_client_factory=make_test_client_factory(server, url=URL), streaming=False)
    second = A2AConfig(card_url=URL, httpx_client_factory=make_test_client_factory(server, url=URL), streaming=False)

    await Agent("client", config=first).ask("ping")

    [through_first] = (await list_tasks(first)).tasks
    [through_second] = (await list_tasks(second)).tasks
    assert through_second.id == through_first.id
