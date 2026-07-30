# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from a2a.server.tasks import InMemoryPushNotificationConfigStore, InMemoryTaskStore
from a2a.types import TaskState

from ag2.a2a import A2AConfig
from ag2.a2a.push import (
    A2APushAuthentication,
    A2APushConfig,
    create_push_notification_config,
    delete_push_notification_config,
    get_push_notification_config,
    list_push_notification_configs,
)
from ag2.a2a.tasks import cancel_task, get_task, list_tasks
from ag2.exceptions import HumanInputNotProvidedError

from ._helpers import PromptThenAckExecutor, a2a_config, make_executor_pair, make_pair


def _push_config() -> A2APushConfig:
    return A2APushConfig(
        url="https://hooks.example.com/a2a",
        token="secret",
        authentication=A2APushAuthentication(scheme="bearer", credentials="abc"),
    )


async def _completed_task_with_push_store() -> tuple[A2AConfig, str]:
    """Drive one ask to completion on a server that also stores push configs."""
    pair = make_pair(
        "hi",
        streaming=False,
        task_store=InMemoryTaskStore(),
        push_config_store=InMemoryPushNotificationConfigStore(),
    )
    await pair.client.ask("ping")
    config = a2a_config(pair.client)
    [task] = (await list_tasks(config)).tasks
    return config, task.id


@pytest.mark.asyncio
class TestTaskAdmin:
    async def test_completed_task_is_listed_and_fetchable_by_id(self) -> None:
        pair = make_pair("hi", streaming=False, task_store=InMemoryTaskStore())
        await pair.client.ask("ping")

        [listed] = (await list_tasks(a2a_config(pair.client))).tasks
        task = await get_task(a2a_config(pair.client), listed.id)

        assert task.id == listed.id
        assert task.status.state == TaskState.TASK_STATE_COMPLETED

    async def test_list_tasks_reports_a_single_complete_page(self) -> None:
        pair = make_pair("hi", streaming=False, task_store=InMemoryTaskStore())
        await pair.client.ask("ping")

        listed = await list_tasks(a2a_config(pair.client))

        # One ask -> exactly one task, and an empty next-page token means the
        # page held everything there was.
        assert (len(listed.tasks), listed.total_size, listed.next_page_token) == (1, 1, "")
        assert listed.page_size >= listed.total_size

    async def test_cancel_active_task_marks_it_cancelled(self) -> None:
        pair = make_executor_pair(
            PromptThenAckExecutor(prompt="What's your name?"),
            streaming=False,
            task_store=InMemoryTaskStore(),
        )

        # No hitl_hook, so the task is abandoned mid-flight and stays active.
        with pytest.raises(HumanInputNotProvidedError):
            await pair.client.ask("hello")

        [task] = (await list_tasks(a2a_config(pair.client))).tasks
        await cancel_task(a2a_config(pair.client), task.id)

        cancelled = await get_task(a2a_config(pair.client), task.id)
        assert cancelled.status.state == TaskState.TASK_STATE_CANCELED


@pytest.mark.asyncio
class TestPushNotificationConfigs:
    async def test_create_returns_the_config_with_a_server_issued_id(self) -> None:
        config, task_id = await _completed_task_with_push_store()
        push = _push_config()

        created = await create_push_notification_config(config, task_id, push)

        # Everything sent comes back untouched — the id is the server's only addition.
        assert created.id
        assert created == replace(push, id=created.id)

    async def test_created_config_round_trips_through_get(self) -> None:
        config, task_id = await _completed_task_with_push_store()

        created = await create_push_notification_config(config, task_id, _push_config())
        assert created.id is not None, "create must return a server-issued id to fetch by"

        fetched = await get_push_notification_config(config, task_id, created.id)

        assert fetched == created

    async def test_created_config_is_the_only_one_listed(self) -> None:
        config, task_id = await _completed_task_with_push_store()

        created = await create_push_notification_config(config, task_id, _push_config())

        assert await list_push_notification_configs(config, task_id) == [created]

    async def test_deleting_leaves_no_configs_behind(self) -> None:
        config, task_id = await _completed_task_with_push_store()
        created = await create_push_notification_config(config, task_id, _push_config())
        assert created.id is not None, "create must return a server-issued id to delete by"

        await delete_push_notification_config(config, task_id, created.id)

        assert await list_push_notification_configs(config, task_id) == []
