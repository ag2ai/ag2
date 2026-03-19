# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the debug integration in Agent.ask() (lines 272-286 of agent.py)."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autogen.beta import Agent
from autogen.beta.testing import TestConfig


@pytest.mark.asyncio()
async def test_agent_ask_with_debug_server_url() -> None:
    """When AG2_DEBUG_SERVER_URL is set, Agent.ask() should set up debug session."""
    config = TestConfig("hello world")
    agent = Agent("test-agent", prompt="Be helpful.", config=config)

    mock_debug_client = MagicMock()
    mock_debug_client.register_session = AsyncMock()
    mock_debug_client.send_event = AsyncMock()

    mock_replay = AsyncMock()

    with patch.dict(os.environ, {"AG2_DEBUG_SERVER_URL": "http://localhost:8765"}):
        with (
            patch(
                "autogen.beta.debug.client.get_server",
                return_value=mock_debug_client,
            ) as mock_get_server,
            patch.object(
                type(mock_debug_client).__name__,
                "register_session",
                new=AsyncMock(),
                create=True,
            ) if False else patch(  # noqa: SIM210
                "autogen.beta.debug.session.DebugSession.replay_events",
                mock_replay,
            ),
        ):
            reply = await agent.ask("hello")

    assert reply.content == "hello world"
    mock_get_server.assert_called_once_with("http://localhost:8765")
    mock_debug_client.register_session.assert_awaited_once()
    mock_replay.assert_awaited_once()


@pytest.mark.asyncio()
async def test_agent_ask_without_debug_server_url() -> None:
    """When AG2_DEBUG_SERVER_URL is NOT set, no debug setup should happen."""
    config = TestConfig("response")
    agent = Agent("test-agent", prompt="Be helpful.", config=config)

    # Ensure the env var is not set
    env = os.environ.copy()
    env.pop("AG2_DEBUG_SERVER_URL", None)

    with patch.dict(os.environ, env, clear=True):
        with patch("autogen.beta.debug.client.get_server") as mock_get_server:
            reply = await agent.ask("hello")

    assert reply.content == "response"
    mock_get_server.assert_not_called()
