# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import pytest
from mcp.client.extension import advertise
from mcp.types import DiscoverResult
from mcp_types.version import LATEST_MODERN_VERSION

from ag2 import Agent
from ag2.mcp import MCPServer, mcp_tool
from ag2.mcp.extensions import client_extension
from ag2.mcp.testing import connect, connect_modern
from ag2.mcp.tools import MCPRequestContext
from ag2.testing import TestConfig

_ID = "com.example/thing"


def _agent() -> Agent:
    return Agent("g", config=TestConfig("hi"))


@mcp_tool
async def read_thing(ctx: MCPRequestContext) -> str:
    """Report what the client advertised for the example extension."""
    return json.dumps(client_extension(ctx, _ID))


def _handshake_ad(settings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """What ``connect`` forwards to ``ClientSession(extensions=...)``."""
    return {_ID: settings}


async def _discovered(session: Any) -> DiscoverResult:
    """The server's own modern-era capability announcement.

    ``connect_modern`` adopts a synthesized handshake (the modern era has none),
    so the session's cached capabilities are the client's assumption rather than
    the server's statement. ``server/discover`` is what actually asks.
    """
    return DiscoverResult.model_validate(await session.send_discover(LATEST_MODERN_VERSION))


@pytest.mark.asyncio
class TestServerAdvertisesExtensions:
    async def test_modern_client_receives_the_extension_map(self) -> None:
        server = MCPServer(_agent(), extensions={_ID: {"level": 2}})

        async with connect_modern(server) as session:
            discovered = await _discovered(session)

        assert discovered.capabilities.extensions == {_ID: {"level": 2}}

    async def test_handshake_client_does_not_receive_it_and_that_is_expected(self) -> None:
        # Not a bug to fix: ``ServerCapabilities.extensions`` does not exist in the
        # 2025-11-25 wire schema, so handshake-era serialization drops the field.
        # Advertising is modern-era only; reading a client's advertisement is not.
        server = MCPServer(_agent(), extensions={_ID: {"level": 2}})

        async with connect(server) as session:
            capabilities = session.server_capabilities

        assert capabilities is not None
        assert capabilities.extensions is None

    async def test_invalid_identifier_raises_at_construction(self) -> None:
        # A reverse-DNS prefix is mandatory; "thing" has none.
        with pytest.raises(TypeError, match="vendor-prefix/name"):
            MCPServer(_agent(), extensions={"thing": {}})

    async def test_no_extensions_advertises_nothing(self) -> None:
        async with connect_modern(MCPServer(_agent())) as session:
            discovered = await _discovered(session)

        assert discovered.capabilities.extensions is None


@pytest.mark.asyncio
class TestServerReadsClientExtensions:
    async def test_handshake_client_advertisement_is_readable(self) -> None:
        server = MCPServer(_agent(), tools=[read_thing])

        async with connect(server, extensions=_handshake_ad({"level": 1})) as session:
            result = await session.call_tool("read_thing", {})

        assert json.loads(result.content[0].text) == {"level": 1}  # type: ignore[union-attr]

    async def test_modern_client_advertisement_is_readable(self) -> None:
        server = MCPServer(_agent(), tools=[read_thing])

        async with connect_modern(server, extensions=[advertise(_ID, {"level": 1})]) as session:
            result = await session.call_tool("read_thing", {})

        assert json.loads(result.content[0].text) == {"level": 1}  # type: ignore[union-attr]

    async def test_empty_settings_are_distinguishable_from_no_advertisement(self) -> None:
        server = MCPServer(_agent(), tools=[read_thing])

        async with connect(server, extensions=_handshake_ad({})) as session:
            supported = await session.call_tool("read_thing", {})
        async with connect(server) as session:
            silent = await session.call_tool("read_thing", {})

        assert json.loads(supported.content[0].text) == {}  # type: ignore[union-attr]
        assert json.loads(silent.content[0].text) is None  # type: ignore[union-attr]

    async def test_reading_outside_a_request_yields_none(self) -> None:
        assert client_extension(None, _ID) is None
