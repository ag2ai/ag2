# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest

from autogen.beta.events import BaseEvent, ToolCallEvent, ToolErrorEvent, ToolResultEvent
from autogen.beta.middleware.builtin.tool_policy import (
    ToolPolicyConfig,
    ToolPolicyMiddleware,
    _ToolPolicy,
    _make_tool_error,
)
from autogen.beta.tools import ToolResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event() -> BaseEvent:
    return mock.MagicMock(spec=BaseEvent)


def _make_context() -> mock.MagicMock:
    return mock.MagicMock()


def _make_tool_call(name: str = "search", call_id: str = "call_001") -> ToolCallEvent:
    tc = mock.MagicMock(spec=ToolCallEvent)
    tc.name = name
    tc.id = call_id
    return tc


# ---------------------------------------------------------------------------
# TestToolPolicy
# ---------------------------------------------------------------------------


class TestToolPolicy:
    def test_blocks_disallowed(self) -> None:
        # Given a policy with "delete_all" blocked
        config = ToolPolicyConfig(blocked_tools=["delete_all"])
        policy = _ToolPolicy(config)

        # When checking the blocked tool
        allowed, reason = policy.check("delete_all")

        # Then the tool is denied and the reason mentions "blocked"
        assert not allowed
        assert "blocked" in reason

    def test_allows_normal(self) -> None:
        # Given a policy with no restrictions
        config = ToolPolicyConfig()
        policy = _ToolPolicy(config)

        # When checking any tool name
        allowed, reason = policy.check("search")

        # Then the tool passes through
        assert allowed
        assert reason == ""

    def test_allowlist_blocks_unlisted(self) -> None:
        # Given a policy with an explicit allowlist
        config = ToolPolicyConfig(allowed_tools=["search", "calc"])
        policy = _ToolPolicy(config)

        # When checking a tool not in the allowlist
        allowed, reason = policy.check("delete_all")

        # Then the tool is denied
        assert not allowed
        assert reason != ""

    def test_allowlist_permits_listed(self) -> None:
        # Given a policy with an explicit allowlist
        config = ToolPolicyConfig(allowed_tools=["search"])
        policy = _ToolPolicy(config)

        # When checking a tool that is on the allowlist
        allowed, reason = policy.check("search")

        # Then the tool is allowed
        assert allowed
        assert reason == ""


# ---------------------------------------------------------------------------
# TestToolPolicyConfig
# ---------------------------------------------------------------------------


class TestToolPolicyConfig:
    def test_blocked_tools_frozen_after_init(self) -> None:
        # Given a config constructed with a mutable list
        config = ToolPolicyConfig(blocked_tools=["delete_all"])

        # Then the stored value is a tuple (immutable)
        assert isinstance(config.blocked_tools, tuple)

    def test_allowed_tools_frozen_after_init(self) -> None:
        # Given a config constructed with a mutable allowed list
        config = ToolPolicyConfig(allowed_tools=["search", "calc"])

        # Then the stored value is a tuple (immutable)
        assert isinstance(config.allowed_tools, tuple)

    def test_allowed_tools_none_means_no_restriction(self) -> None:
        # Given a config with allowed_tools=None (default)
        config = ToolPolicyConfig()

        # Then allowed_tools is None, meaning no allowlist restriction
        assert config.allowed_tools is None


# ---------------------------------------------------------------------------
# TestToolPolicyAdversarial
# ---------------------------------------------------------------------------


class TestToolPolicyAdversarial:
    def test_empty_allowlist_blocks_all(self) -> None:
        # Given an empty allowlist -- deny all tools
        config = ToolPolicyConfig(allowed_tools=[])
        policy = _ToolPolicy(config)

        # When checking any tool
        allowed, _ = policy.check("search")

        # Then the tool is denied and internal allowed set is empty frozenset
        assert not allowed
        assert policy._allowed == frozenset()

    def test_blocklist_overrides_allowlist(self) -> None:
        # Given a tool that appears in both blocked and allowed lists
        config = ToolPolicyConfig(blocked_tools=["search"], allowed_tools=["search", "calc"])
        policy = _ToolPolicy(config)

        # When checking the conflicting tool
        allowed, reason = policy.check("search")

        # Then blocked wins
        assert not allowed
        assert "blocked" in reason

    def test_tool_error_content_matches_reason(self) -> None:
        # Given a tool call that will be blocked
        event = _make_tool_call(name="delete_all")
        reason = "tool 'delete_all' is blocked"

        # When building the error event
        error_event = _make_tool_error(event, reason)

        # Then the content equals the reason string, not "NoneType: None"
        assert error_event.content == reason
        assert "NoneType" not in error_event.content


# ---------------------------------------------------------------------------
# TestToolPolicyStats
# ---------------------------------------------------------------------------


class TestToolPolicyStats:
    @pytest.mark.asyncio()
    async def test_call_and_blocked_counters(self) -> None:
        # Given a middleware factory with a blocklist
        config = ToolPolicyConfig(blocked_tools=["bad_tool"])
        factory = ToolPolicyMiddleware(config)

        ctx = _make_context()
        initial_event = _make_event()

        # Prepare a real ToolResultEvent to return from call_next
        good_call = ToolCallEvent(id="c1", name="good_tool")
        good_result = ToolResultEvent(parent_id="c1", name="good_tool", result=ToolResult("ok"))

        async def call_next(event: ToolCallEvent, context: mock.MagicMock) -> ToolResultEvent:
            return good_result

        # When processing one allowed call and one blocked call
        instance_allow = factory(initial_event, ctx)
        await instance_allow.on_tool_execution(call_next, good_call, ctx)

        blocked_call = ToolCallEvent(id="c2", name="bad_tool")
        instance_block = factory(initial_event, ctx)
        result = await instance_block.on_tool_execution(call_next, blocked_call, ctx)

        # Then counters reflect the activity
        assert factory.total_tool_calls == 1
        assert factory.total_blocked == 1
        assert isinstance(result, ToolErrorEvent)
