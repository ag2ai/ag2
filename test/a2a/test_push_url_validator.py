# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

import pytest
from a2a.server.context import ServerCallContext
from a2a.server.tasks import InMemoryPushNotificationConfigStore, InMemoryTaskStore
from a2a.types import Task, TaskState, TaskStatus
from a2a.utils.errors import InvalidParamsError

from ag2 import Agent
from ag2.a2a import A2AConfig, A2AServer, build_card
from ag2.a2a.push import A2APushConfig, create_push_notification_config, list_push_notification_configs
from ag2.a2a.testing import make_test_client_factory, make_test_rest_client_factory, pick_free_port
from ag2.a2a.transports import TransportName
from ag2.testing import TestConfig


@asynccontextmanager
async def push_server(
    transport: TransportName, validator: Callable[[str], Awaitable[bool]] | None = None
) -> AsyncGenerator[A2AConfig]:
    store = InMemoryTaskStore()
    await store.save(
        Task(id="task", context_id="context", status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED)),
        ServerCallContext(),
    )
    agent = Agent("server", config=TestConfig("unused"))
    options: dict[str, Any] = {"push_url_validator": validator} if validator is not None else {}
    server = A2AServer(
        agent,
        task_store=store,
        push_config_store=InMemoryPushNotificationConfigStore(),
        **options,
    )
    if transport == "grpc":
        url = f"127.0.0.1:{pick_free_port()}"
        card = build_card(agent, url=url, transports=("grpc",), grpc_url=url, push_notifications=True)
        listener = server.build_grpc(bind=url, grpc_url=url, card=card)
        await listener.start()
        try:
            yield A2AConfig(card_url=url, preset_card=card, prefer="grpc")
        finally:
            await listener.stop(grace=0)
    else:
        factory = make_test_rest_client_factory if transport == "rest" else make_test_client_factory
        yield A2AConfig(card_url="http://test", httpx_client_factory=factory(server), prefer=transport)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["jsonrpc", "rest", "grpc"])
async def test_push_url_validator_rejects_without_storing_and_accepts_allowed_url(transport: TransportName) -> None:
    seen: list[str] = []

    async def validate(url: str) -> bool:
        seen.append(url)
        return url == "https://hooks.example.com/a2a"

    async with push_server(transport, validate) as config:
        with pytest.raises(InvalidParamsError):
            await create_push_notification_config(config, "task", A2APushConfig(url="http://127.0.0.1/private"))
        assert await list_push_notification_configs(config, "task") == []

        push = A2APushConfig(url="https://hooks.example.com/a2a", token="secret")
        created = await create_push_notification_config(config, "task", push)
        assert created.id
        assert created == replace(push, id=created.id)
        assert await list_push_notification_configs(config, "task") == [created]
    assert seen == ["http://127.0.0.1/private", "https://hooks.example.com/a2a"]


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["jsonrpc", "rest", "grpc"])
async def test_omitting_push_url_validator_keeps_existing_behavior(transport: TransportName) -> None:
    async with push_server(transport) as config:
        push = A2APushConfig(url="http://127.0.0.1/private")
        created = await create_push_notification_config(config, "task", push)
        assert created == replace(push, id=created.id)
        assert await list_push_notification_configs(config, "task") == [created]
