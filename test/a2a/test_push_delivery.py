# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from uuid import uuid4

import pytest
from a2a.server.tasks import InMemoryPushNotificationConfigStore
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from ag2 import Agent
from ag2.a2a import A2AServer
from ag2.a2a.testing import make_test_client_factory
from ag2.testing import TestConfig

from ._http_server import serve_over_http


class _Webhook:
    def __init__(self) -> None:
        self.tokens: list[str | None] = []
        self.received = asyncio.Event()

    async def receive(self, request: Request) -> Response:
        self.tokens.append(request.headers.get("x-a2a-notification-token"))
        self.received.set()
        return Response(status_code=200)


@pytest.mark.asyncio
async def test_server_with_a_push_store_delivers_to_the_registered_webhook() -> None:
    # Spec §4.3: a server that advertises push notifications attempts delivery; a store alone advertises them.
    webhook = _Webhook()
    server = A2AServer(
        Agent("server", config=TestConfig("hello")), push_config_store=InMemoryPushNotificationConfigStore()
    )
    receiver = Starlette(routes=[Route("/hook", webhook.receive, methods=["POST"])])

    async with serve_over_http(lambda url: receiver) as hook_url, make_test_client_factory(server)() as http:
        card = (await http.get("/.well-known/agent-card.json")).json()
        response = await http.post(
            "/",
            headers={"A2A-Version": "1.0"},
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "SendMessage",
                "params": {
                    "message": {"messageId": uuid4().hex, "role": "ROLE_USER", "parts": [{"text": "hi"}]},
                    "configuration": {"taskPushNotificationConfig": {"url": f"{hook_url}/hook", "token": "tok"}},
                },
            },
        )
        await asyncio.wait_for(webhook.received.wait(), 5)

    assert card["capabilities"]["pushNotifications"] is True
    assert response.json()["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert set(webhook.tokens) == {"tok"}
