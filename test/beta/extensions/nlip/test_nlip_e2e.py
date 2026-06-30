# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Agent
from ag2.events import ModelRequest, TextInput
from ag2.extensions.nlip import NlipConfig, NlipServer
from ag2.extensions.nlip.testing import make_test_client_factory
from ag2.testing import TestConfig, TrackingConfig


def _make_pair(*events: str, server_url: str = "http://test") -> tuple[NlipServer, Agent, TrackingConfig]:
    tracking = TrackingConfig(TestConfig(*events))
    server_agent = Agent("server-agent", config=tracking)
    server = NlipServer(server_agent)

    factory = make_test_client_factory(server, url=server_url)
    client = Agent(
        "client-agent",
        config=NlipConfig(url=server_url, httpx_client_factory=factory),
    )
    return server, client, tracking


@pytest.mark.asyncio
class TestE2E:
    async def test_single_turn_round_trip(self) -> None:
        _, client, _ = _make_pair("hello world")

        reply = await client.ask("ping")

        assert reply.response.content == "hello world"

    async def test_server_sees_user_input(self) -> None:
        _, client, tracking = _make_pair("ack")

        await client.ask("hello server")

        tracking.mock.assert_called_with(ModelRequest([TextInput("hello server")]))
