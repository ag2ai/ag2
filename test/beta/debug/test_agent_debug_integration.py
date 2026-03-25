# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the debug integration via DebugMiddleware."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autogen.beta import Agent
from autogen.beta.debug.client import DebugClient
from autogen.beta.debug.middleware import DebugMiddleware
from autogen.beta.debug.session import DEBUG_SESSION_VAR, DebugSession
from autogen.beta.testing import TestConfig


def _mock_client() -> MagicMock:
    client = MagicMock(spec=DebugClient)
    client.send_event = AsyncMock()
    client.register_stream = AsyncMock()
    client.register_session = AsyncMock()
    client.add_stream_to_session = AsyncMock()
    client.end_session = AsyncMock()
    return client


@pytest.mark.asyncio()
async def test_middleware_with_env_var_creates_auto_session() -> None:
    """When AG2_DEBUG_SERVER_URL is set, DebugMiddleware should auto-create and close a session."""
    config = TestConfig("hello world")
    agent = Agent("test-agent", prompt="Be helpful.", config=config)

    mock_client = _mock_client()

    with (
        patch.dict(os.environ, {"AG2_DEBUG_SERVER_URL": "http://localhost:8765"}),
        patch("autogen.beta.debug.session.DebugClient", return_value=mock_client),
    ):
        reply = await agent.ask("hello", middleware=[DebugMiddleware])

    assert reply.body == "hello world"
    # Should have recorded events via send_event
    assert mock_client.send_event.await_count >= 1
    # Auto-created session should be closed after ask()
    mock_client.end_session.assert_awaited_once()


@pytest.mark.asyncio()
async def test_middleware_with_explicit_session_does_not_close() -> None:
    """When a session is passed via variables, middleware should NOT close it."""
    config = TestConfig("hello world")
    agent = Agent("test-agent", prompt="Be helpful.", config=config)

    session = DebugSession(name="my-run", url="http://localhost:8765")
    mock_client = _mock_client()
    session._client = mock_client

    reply = await agent.ask(
        "hello",
        variables={DEBUG_SESSION_VAR: session},
        middleware=[DebugMiddleware],
    )

    assert reply.body == "hello world"
    assert mock_client.send_event.await_count >= 1
    # Caller-provided session should NOT be auto-closed
    mock_client.end_session.assert_not_awaited()


@pytest.mark.asyncio()
async def test_middleware_disabled_without_env_var() -> None:
    """When AG2_DEBUG_SERVER_URL is NOT set and no session provided, middleware is a no-op."""
    config = TestConfig("response")
    agent = Agent("test-agent", prompt="Be helpful.", config=config)

    env = os.environ.copy()
    env.pop("AG2_DEBUG_SERVER_URL", None)

    with patch.dict(os.environ, env, clear=True):
        reply = await agent.ask("hello", middleware=[DebugMiddleware])

    assert reply.body == "response"
