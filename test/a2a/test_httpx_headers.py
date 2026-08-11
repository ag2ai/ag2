# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx
import pytest

from ag2 import Agent
from ag2.a2a import A2AConfig, A2AServer
from ag2.a2a.testing import make_test_client_factory
from ag2.a2a.transports._common import DEFAULT_AGENT_CARD_PATH
from ag2.testing import TestConfig

from ._http_server import serve_over_http


class _HeaderSpy:
    """ASGI wrapper recording ``header``'s value on every request, by path."""

    def __init__(self, app: object, header: str) -> None:
        self.app = app
        self.header = header.lower().encode()
        self.seen: dict[str, str | None] = {}

    async def __call__(self, scope: dict[str, Any], receive: object, send: object) -> None:
        if scope["type"] == "http":
            raw = dict(scope["headers"]).get(self.header)
            self.seen[scope["path"]] = raw.decode() if raw is not None else None
        await self.app(scope, receive, send)  # type: ignore[operator]


@pytest.mark.asyncio
async def test_configured_headers_reach_the_server() -> None:
    # Needs a real socket rather than the usual in-process factory: passing
    # httpx_client_factory is exactly what makes A2AConfig skip building the
    # client that would carry these headers.
    agent = Agent("server", config=TestConfig("hi"))
    server = A2AServer(agent)
    spy: _HeaderSpy | None = None

    def build_app(url: str) -> object:
        nonlocal spy
        spy = _HeaderSpy(server.build_jsonrpc(url=url), "X-Tenant")
        return spy

    async with serve_over_http(build_app) as url:
        client = Agent("client", config=A2AConfig(card_url=url, headers={"X-Tenant": "alpha"}))

        reply = await client.ask("ping")

    assert reply.body == "hi"
    assert spy is not None
    # The card fetch carries the headers too, not just the RPC call.
    assert DEFAULT_AGENT_CARD_PATH in spy.seen
    assert set(spy.seen.values()) == {"alpha"}


@pytest.mark.asyncio
async def test_a_supplied_client_is_used_as_is_and_left_unmutated() -> None:
    # AG2 must not stamp its headers onto a client it does not own; it warns
    # instead, so the caller learns the headers were ignored.
    server = A2AServer(Agent("server", config=TestConfig("hi")))
    build = make_test_client_factory(server, url="http://test")
    handed_out: list[httpx.AsyncClient] = []

    def factory() -> httpx.AsyncClient:
        client = build()
        handed_out.append(client)
        return client

    client = Agent(
        "client",
        config=A2AConfig(card_url="http://test", headers={"X-Tenant": "alpha"}, httpx_client_factory=factory),
    )

    with pytest.warns(RuntimeWarning, match="headers"):
        reply = await client.ask("ping")

    assert reply.body == "hi"
    assert [c.headers.get("X-Tenant") for c in handed_out] == [None]
