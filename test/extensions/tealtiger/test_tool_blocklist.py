# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for TealTiger tool blocklist policy.

The blocklist is the complement of the allowlist: every tool is permitted
except those matching a blocked pattern. Patterns match tool names via
``fnmatch``, so both exact names and globs (``delete_*``) are supported.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ag2.events import ToolCallEvent, ToolErrorEvent
from ag2.extensions.tealtiger import GovernanceMode, GovernancePolicy, TealTigerMiddleware
from ag2.utils import AGENT_CONTEXT_DEPENDENCY_KEY


def _make_context(agent_name: str = "assistant") -> MagicMock:
    """Create a mock Context with agent dependency."""
    ctx = MagicMock()
    agent = MagicMock()
    agent.name = agent_name
    ctx.dependencies = {AGENT_CONTEXT_DEPENDENCY_KEY: agent}
    return ctx


def _make_tool_event(name: str = "search", arguments: dict | None = None) -> MagicMock:
    """Create a mock ToolCallEvent with serialized_arguments."""
    event = MagicMock(spec=ToolCallEvent)
    event.name = name
    args = arguments or {}
    event.serialized_arguments = args
    event.arguments = json.dumps(args)
    event.call_id = "call-123"
    return event


@pytest.mark.asyncio
class TestToolBlocklist:
    async def test_non_blocked_tool_passes(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_blocklist(["delete_all", "shell"])],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="search")
        call_next = AsyncMock(return_value=MagicMock())

        result = await per_turn.on_tool_execution(call_next, event, ctx)

        call_next.assert_awaited_once()
        assert not isinstance(result, ToolErrorEvent)

    async def test_blocked_tool_returns_error(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_blocklist(["delete_all"])],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="delete_all")
        call_next = AsyncMock()

        result = await per_turn.on_tool_execution(call_next, event, ctx)

        assert isinstance(result, ToolErrorEvent)
        assert "GOVERNANCE DENIED" in str(result.error)
        assert "TOOL_BLOCKED" in str(result.error)
        call_next.assert_not_awaited()

    async def test_glob_pattern_blocks_matching_tools(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_blocklist(["delete_*"])],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="delete_database")
        call_next = AsyncMock()

        result = await per_turn.on_tool_execution(call_next, event, ctx)

        assert isinstance(result, ToolErrorEvent)
        assert "TOOL_BLOCKED" in str(result.error)
        call_next.assert_not_awaited()

    async def test_glob_pattern_allows_non_matching_tools(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_blocklist(["delete_*"])],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="read_file")
        call_next = AsyncMock(return_value=MagicMock())

        result = await per_turn.on_tool_execution(call_next, event, ctx)

        call_next.assert_awaited_once()
        assert not isinstance(result, ToolErrorEvent)

    async def test_blocked_decision_recorded(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_blocklist(["shell"])],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="shell")
        call_next = AsyncMock()

        await per_turn.on_tool_execution(call_next, event, ctx)

        last = mw.decisions[-1]
        assert last.action == "DENY"
        assert "TOOL_BLOCKED" in last.reason_codes
        assert mw.deny_count == 1

    async def test_observe_mode_allows_blocked_tool(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_blocklist(["delete_all"])],
            mode=GovernanceMode.OBSERVE,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="delete_all")
        call_next = AsyncMock(return_value=MagicMock())

        result = await per_turn.on_tool_execution(call_next, event, ctx)

        call_next.assert_awaited_once()
        assert not isinstance(result, ToolErrorEvent)

    async def test_monitor_mode_records_but_allows_blocked_tool(self):
        mw = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_blocklist(["delete_all"])],
            mode=GovernanceMode.MONITOR,
        )
        ctx = _make_context()
        per_turn = mw(MagicMock(), ctx)
        event = _make_tool_event(name="delete_all")
        call_next = AsyncMock(return_value=MagicMock())

        result = await per_turn.on_tool_execution(call_next, event, ctx)

        call_next.assert_awaited_once()
        assert not isinstance(result, ToolErrorEvent)
        assert mw.decisions[-1].action == "DENY"

    async def test_allowlist_and_blocklist_combine(self):
        # Allowlist permits read_*, blocklist carves out read_secrets.
        mw = TealTigerMiddleware(
            policies=[
                GovernancePolicy.tool_allowlist(["read_*"]),
                GovernancePolicy.tool_blocklist(["read_secrets"]),
            ],
            mode=GovernanceMode.ENFORCE,
        )
        ctx = _make_context()

        allowed = mw(MagicMock(), ctx)
        allowed_event = _make_tool_event(name="read_file")
        allowed_next = AsyncMock(return_value=MagicMock())
        allowed_result = await allowed.on_tool_execution(allowed_next, allowed_event, ctx)
        allowed_next.assert_awaited_once()
        assert not isinstance(allowed_result, ToolErrorEvent)

        blocked = mw(MagicMock(), ctx)
        blocked_event = _make_tool_event(name="read_secrets")
        blocked_next = AsyncMock()
        blocked_result = await blocked.on_tool_execution(blocked_next, blocked_event, ctx)
        assert isinstance(blocked_result, ToolErrorEvent)
        assert "TOOL_BLOCKED" in str(blocked_result.error)
        blocked_next.assert_not_awaited()


def test_empty_blocklist_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        GovernancePolicy.tool_blocklist([])
