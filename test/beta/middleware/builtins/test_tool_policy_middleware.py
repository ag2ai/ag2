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
# TestToolPolicyMiddleware
# ---------------------------------------------------------------------------


class TestToolPolicyMiddleware:
    def test_flat_constructor(self) -> None:
        # Given arguments passed directly (no explicit config object)
        mw = ToolPolicyMiddleware(
            blocked_tools=["delete_all"],
            allowed_tools=["search"],
        )

        # Then the internal config is built from those arguments
        assert mw._config.blocked_tools == ("delete_all",)
        assert mw._config.allowed_tools == ("search",)

    def test_default_constructor(self) -> None:
        # Given no arguments
        mw = ToolPolicyMiddleware()

        # Then defaults are empty blocklist and None allowlist (no restriction)
        assert mw._config.blocked_tools == ()
        assert mw._config.allowed_tools is None

    @pytest.mark.asyncio()
    async def test_on_blocked_callback_fires_on_denial(self) -> None:
        # Given a middleware with a blocklist and an on_blocked callback
        recorded: list[tuple[str, str]] = []

        def audit(call: ToolCallEvent, reason: str) -> None:
            recorded.append((call.name, reason))

        factory = ToolPolicyMiddleware(blocked_tools=["bad_tool"], on_blocked=audit)

        ctx = _make_context()
        initial_event = _make_event()

        async def call_next(event: ToolCallEvent, context: mock.MagicMock) -> ToolResultEvent:
            return ToolResultEvent(parent_id=event.id, name=event.name, result=ToolResult("ok"))

        # When a blocked call is denied
        blocked_call = ToolCallEvent(id="c1", name="bad_tool")
        instance = factory(initial_event, ctx)
        result = await instance.on_tool_execution(call_next, blocked_call, ctx)

        # Then the callback was invoked with the call and reason, and a ToolErrorEvent was returned
        assert len(recorded) == 1
        assert recorded[0][0] == "bad_tool"
        assert "blocked" in recorded[0][1]
        assert isinstance(result, ToolErrorEvent)

    @pytest.mark.asyncio()
    async def test_on_blocked_not_called_when_allowed(self) -> None:
        # Given a middleware with a callback and an allowed call
        recorded: list[tuple[str, str]] = []

        def audit(call: ToolCallEvent, reason: str) -> None:
            recorded.append((call.name, reason))

        factory = ToolPolicyMiddleware(allowed_tools=["good_tool"], on_blocked=audit)

        ctx = _make_context()
        initial_event = _make_event()
        good_call = ToolCallEvent(id="c1", name="good_tool")
        good_result = ToolResultEvent(parent_id="c1", name="good_tool", result=ToolResult("ok"))

        async def call_next(event: ToolCallEvent, context: mock.MagicMock) -> ToolResultEvent:
            return good_result

        # When the call passes the policy
        instance = factory(initial_event, ctx)
        result = await instance.on_tool_execution(call_next, good_call, ctx)

        # Then the callback was not invoked and the downstream result was returned
        assert recorded == []
        assert result is good_result

    @pytest.mark.asyncio()
    async def test_update_policy_takes_effect_on_next_call(self) -> None:
        # Given a middleware that initially allows "search" only
        factory = ToolPolicyMiddleware(allowed_tools=["search"])
        ctx = _make_context()
        initial_event = _make_event()

        async def call_next(event: ToolCallEvent, context: mock.MagicMock) -> ToolResultEvent:
            return ToolResultEvent(parent_id=event.id, name=event.name, result=ToolResult("ok"))

        # When "shell_exec" is attempted before the update -- denied
        instance_before = factory(initial_event, ctx)
        result_before = await instance_before.on_tool_execution(
            call_next, ToolCallEvent(id="c1", name="shell_exec"), ctx
        )
        assert isinstance(result_before, ToolErrorEvent)

        # And the policy is replaced at runtime to allow shell_exec
        factory.update_policy(ToolPolicyConfig(allowed_tools=["search", "shell_exec"]))

        # Then a new instance picks up the new policy and permits the call
        instance_after = factory(initial_event, ctx)
        result_after = await instance_after.on_tool_execution(call_next, ToolCallEvent(id="c2", name="shell_exec"), ctx)
        assert not isinstance(result_after, ToolErrorEvent)
        assert factory.config.allowed_tools == ("search", "shell_exec")

    @pytest.mark.asyncio()
    async def test_update_policy_in_flight_instance_uses_snapshot(self) -> None:
        # Given a factory that allows "search" and an instance constructed under that policy
        factory = ToolPolicyMiddleware(allowed_tools=["search"])
        ctx = _make_context()
        instance = factory(_make_event(), ctx)  # snapshot taken here

        async def call_next(event: ToolCallEvent, context: mock.MagicMock) -> ToolResultEvent:
            return ToolResultEvent(parent_id=event.id, name=event.name, result=ToolResult("ok"))

        # When the factory policy is swapped to allow everything mid-flight
        factory.update_policy(ToolPolicyConfig())

        # Then the previously constructed instance still enforces the old snapshot
        result = await instance.on_tool_execution(call_next, ToolCallEvent(id="c1", name="shell_exec"), ctx)
        assert isinstance(result, ToolErrorEvent)

    def test_update_policy_clears_restrictions(self) -> None:
        # Given a middleware with both blocked and allowed lists
        factory = ToolPolicyMiddleware(blocked_tools=["x"], allowed_tools=["y"])

        # When the policy is replaced with an empty config
        factory.update_policy(ToolPolicyConfig())

        # Then check() permits any tool name
        allowed, reason = factory._policy.check("anything")
        assert allowed
        assert reason == ""
        assert factory.config.blocked_tools == ()
        assert factory.config.allowed_tools is None

    def test_config_property_returns_active_config(self) -> None:
        # Given a middleware
        factory = ToolPolicyMiddleware(blocked_tools=["delete_all"])

        # Then config property exposes the active ToolPolicyConfig
        assert isinstance(factory.config, ToolPolicyConfig)
        assert factory.config.blocked_tools == ("delete_all",)

    @pytest.mark.asyncio()
    async def test_no_callback_means_no_error(self) -> None:
        # Given a middleware without an on_blocked callback
        factory = ToolPolicyMiddleware(blocked_tools=["bad_tool"])

        ctx = _make_context()
        initial_event = _make_event()

        async def call_next(event: ToolCallEvent, context: mock.MagicMock) -> ToolResultEvent:
            return ToolResultEvent(parent_id=event.id, name=event.name, result=ToolResult("ok"))

        # When a blocked call is denied without a callback
        blocked_call = ToolCallEvent(id="c1", name="bad_tool")
        instance = factory(initial_event, ctx)
        result = await instance.on_tool_execution(call_next, blocked_call, ctx)

        # Then denial still works, just without observation
        assert isinstance(result, ToolErrorEvent)
