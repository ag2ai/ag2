# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the debug integration in Agent.ask()."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from autogen.beta import Agent
from autogen.beta.testing import TestConfig


@pytest.mark.asyncio()
async def test_agent_ask_with_debug_server_url() -> None:
    """When AG2_DEBUG_SERVER_URL is set, Agent.ask() should set up a debug session."""
    config = TestConfig("hello world")
    agent = Agent("test-agent", prompt="Be helpful.", config=config)

    mock_attach = AsyncMock()
    mock_close = AsyncMock()

    with (
        patch.dict(os.environ, {"AG2_DEBUG_SERVER_URL": "http://localhost:8765"}),
        patch("autogen.beta.debug.session.DebugSession._attach", mock_attach),
        patch("autogen.beta.debug.session.DebugSession.close", mock_close),
    ):
        reply = await agent.ask("hello")

    assert reply.content == "hello world"
    mock_attach.assert_awaited_once()
    # Auto-created session should be closed after ask()
    mock_close.assert_awaited_once()


@pytest.mark.asyncio()
async def test_agent_ask_with_explicit_session_via_variables() -> None:
    """When a session is passed via variables, ask() should attach it but NOT close it."""
    from autogen.beta.debug.session import DEBUG_SESSION_VAR, DebugSession

    config = TestConfig("hello world")
    agent = Agent("test-agent", prompt="Be helpful.", config=config)

    mock_attach = AsyncMock()
    mock_close = AsyncMock()

    session = DebugSession(name="my-run", url="http://localhost:8765")

    with (
        patch.dict(os.environ, {"AG2_DEBUG_SERVER_URL": "http://localhost:8765"}),
        patch.object(session, "_attach", mock_attach),
        patch.object(session, "close", mock_close),
    ):
        reply = await agent.ask("hello", variables={DEBUG_SESSION_VAR: session})

    assert reply.content == "hello world"
    mock_attach.assert_awaited_once()
    # Caller-provided session should NOT be auto-closed
    mock_close.assert_not_awaited()


@pytest.mark.asyncio()
async def test_agent_ask_without_debug_server_url() -> None:
    """When AG2_DEBUG_SERVER_URL is NOT set, no debug setup should happen."""
    config = TestConfig("response")
    agent = Agent("test-agent", prompt="Be helpful.", config=config)

    # Ensure the env var is not set
    env = os.environ.copy()
    env.pop("AG2_DEBUG_SERVER_URL", None)

    with (
        patch.dict(os.environ, env, clear=True),
        patch("autogen.beta.debug.session.DebugSession._attach") as mock_attach,
    ):
        reply = await agent.ask("hello")

    assert reply.content == "response"
    mock_attach.assert_not_called()
