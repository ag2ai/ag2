# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from a2a.server.tasks import InMemoryTaskStore, TaskStore

from ag2 import Agent
from ag2.a2a import A2AServer
from ag2.a2a.tasks import list_tasks
from ag2.testing import TestConfig

from ._helpers import make_pair


def test_default_task_store_is_materialised_eagerly() -> None:
    server = A2AServer(Agent("a", config=TestConfig("hi")))
    assert isinstance(server.task_store, TaskStore)


def test_user_task_store_is_preserved() -> None:
    custom = InMemoryTaskStore()
    server = A2AServer(Agent("a", config=TestConfig("hi")), task_store=custom)
    assert server.task_store is custom


@pytest.mark.asyncio
async def test_task_store_persists_across_build_calls() -> None:
    pair = make_pair("hi", streaming=False, task_store=InMemoryTaskStore())

    await pair.client.ask("ping")
    tasks_before = (await list_tasks(pair.client.config)).tasks
    assert len(tasks_before) >= 1

    store_before = pair.server.task_store
    pair.server.build_jsonrpc(url="http://test")
    assert pair.server.task_store is store_before

    tasks_after = (await list_tasks(pair.client.config)).tasks
    assert {t.id for t in tasks_before} == {t.id for t in tasks_after}
