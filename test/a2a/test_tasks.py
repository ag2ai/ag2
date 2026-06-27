# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from a2a.server.tasks import InMemoryPushNotificationConfigStore, InMemoryTaskStore
from a2a.types import TaskState

from ag2.a2a.push import (
    A2APushAuthentication,
    A2APushConfig,
    create_push_notification_config,
    delete_push_notification_config,
    get_push_notification_config,
    list_push_notification_configs,
)
from ag2.a2a.tasks import ListedTasks, cancel_task, get_task, list_tasks
from ag2.exceptions import HumanInputNotProvidedError

from ._helpers import PromptThenAckExecutor, make_executor_pair, make_pair


@pytest.mark.asyncio
class TestAdmin:
    async def test_get_task_returns_completed_task(self) -> None:
        pair = make_pair(
            "hi",
            streaming=False,
            task_store=InMemoryTaskStore(),
        )
        await pair.client.ask("ping")

        listed = await list_tasks(pair.client.config)
        task = await get_task(pair.client.config, listed.tasks[0].id)

        assert task.id == listed.tasks[0].id

    async def test_list_tasks_returns_listed_tasks_dataclass(self) -> None:
        pair = make_pair(
            "hi",
            streaming=False,
            task_store=InMemoryTaskStore(),
        )
        await pair.client.ask("ping")

        listed = await list_tasks(pair.client.config)

        assert isinstance(listed, ListedTasks)
        assert len(listed.tasks) >= 1
        assert isinstance(listed.next_page_token, str)
        assert isinstance(listed.page_size, int)
        assert isinstance(listed.total_size, int)

    async def test_cancel_active_task_marks_it_cancelled(self) -> None:
        executor = PromptThenAckExecutor(prompt="What's your name?")
        pair = make_executor_pair(
            executor,
            streaming=False,
            task_store=InMemoryTaskStore(),
        )

        with pytest.raises(HumanInputNotProvidedError):
            await pair.client.ask("hello")

        [task] = (await list_tasks(pair.client.config)).tasks
        await cancel_task(pair.client.config, task.id)

        cancelled = await get_task(pair.client.config, task.id)
        assert cancelled.status.state == TaskState.TASK_STATE_CANCELED


@pytest.mark.asyncio
class TestPushCRUD:
    async def test_create_get_list_delete(self) -> None:
        pair = make_pair(
            "hi",
            streaming=False,
            task_store=InMemoryTaskStore(),
            push_config_store=InMemoryPushNotificationConfigStore(),
        )
        await pair.client.ask("ping")
        [task] = (await list_tasks(pair.client.config)).tasks

        push = A2APushConfig(
            url="https://hooks.example.com/a2a",
            token="secret",
            authentication=A2APushAuthentication(scheme="bearer", credentials="abc"),
        )
        created = await create_push_notification_config(pair.client.config, task.id, push)
        assert created.url == push.url
        assert created.id is not None

        configs = await list_push_notification_configs(pair.client.config, task.id)
        assert any(c.url == push.url for c in configs)

        fetched = await get_push_notification_config(pair.client.config, task.id, created.id)
        assert fetched.url == push.url

        await delete_push_notification_config(pair.client.config, task.id, created.id)

        after_delete = await list_push_notification_configs(pair.client.config, task.id)
        assert all(c.id != created.id for c in after_delete)
