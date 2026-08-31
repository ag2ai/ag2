# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for TealTiger per-tool argument validation.

`arg_validation` constrains the arguments a matched tool may receive — max/min
length, type, blocked terms, blocked regex patterns, and allowed values — to
defend against dangerous values such as SQL injection and path traversal.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ag2.events import ToolCallEvent, ToolErrorEvent
from ag2.extensions.tealtiger import GovernancePolicy, TealTigerMiddleware
from ag2.utils import AGENT_CONTEXT_DEPENDENCY_KEY


def _make_context(agent_name: str = "assistant") -> MagicMock:
    """Create a mock Context with agent dependency."""
    ctx = MagicMock()
    agent = MagicMock()
    agent.name = agent_name
    ctx.dependencies = {AGENT_CONTEXT_DEPENDENCY_KEY: agent}
    return ctx


def _make_tool_event(name: str = "sql_query", arguments: dict | None = None) -> MagicMock:
    """Create a mock ToolCallEvent with serialized_arguments."""
    event = MagicMock(spec=ToolCallEvent)
    event.name = name
    args = arguments or {}
    event.serialized_arguments = args
    event.arguments = json.dumps(args)
    event.call_id = "call-123"
    return event


async def _run(policy: GovernancePolicy, tool_name: str, arguments: dict, mode: str = "ENFORCE"):
    """Run one tool call through the middleware, return (result, middleware)."""
    mw = TealTigerMiddleware(policies=[policy], mode=mode)
    ctx = _make_context()
    per_turn = mw(MagicMock(), ctx)
    event = _make_tool_event(name=tool_name, arguments=arguments)
    call_next = AsyncMock(return_value=MagicMock())
    result = await per_turn.on_tool_execution(call_next, event, ctx)
    return result, mw


@pytest.mark.asyncio
class TestArgValidation:
    async def test_within_constraints_passes(self):
        policy = GovernancePolicy.arg_validation("sql_query", {"query": {"max_length": 100}})
        result, mw = await _run(policy, "sql_query", {"query": "SELECT 1"})
        assert not isinstance(result, ToolErrorEvent)
        assert mw.decisions[-1].action == "ALLOW"

    async def test_max_length_denied(self):
        policy = GovernancePolicy.arg_validation("sql_query", {"query": {"max_length": 10}})
        result, mw = await _run(policy, "sql_query", {"query": "SELECT * FROM a very long table name"})
        assert isinstance(result, ToolErrorEvent)
        assert "ARG_VALIDATION:query:max_length" in str(result.error)

    async def test_min_length_denied(self):
        policy = GovernancePolicy.arg_validation("sql_query", {"query": {"min_length": 5}})
        result, _ = await _run(policy, "sql_query", {"query": "hi"})
        assert isinstance(result, ToolErrorEvent)
        assert "ARG_VALIDATION:query:min_length" in str(result.error)

    async def test_blocked_terms_denied_case_insensitive(self):
        policy = GovernancePolicy.arg_validation("sql_query", {"query": {"blocked_terms": ["DROP", "DELETE"]}})
        result, _ = await _run(policy, "sql_query", {"query": "drop table users"})
        assert isinstance(result, ToolErrorEvent)
        assert "ARG_VALIDATION:query:blocked_terms" in str(result.error)

    async def test_blocked_patterns_denies_path_traversal(self):
        policy = GovernancePolicy.arg_validation("read_file", {"path": {"blocked_patterns": [r"\.\.[\\/]"]}})
        result, _ = await _run(policy, "read_file", {"path": "../../etc/passwd"})
        assert isinstance(result, ToolErrorEvent)
        assert "ARG_VALIDATION:path:blocked_patterns" in str(result.error)

    async def test_blocked_patterns_allows_safe_path(self):
        policy = GovernancePolicy.arg_validation("read_file", {"path": {"blocked_patterns": [r"\.\.[\\/]"]}})
        result, _ = await _run(policy, "read_file", {"path": "data/report.txt"})
        assert not isinstance(result, ToolErrorEvent)

    async def test_type_check_rejects_wrong_type(self):
        policy = GovernancePolicy.arg_validation("calc", {"n": {"type": "int"}})
        result, _ = await _run(policy, "calc", {"n": "not-a-number"})
        assert isinstance(result, ToolErrorEvent)
        assert "ARG_VALIDATION:n:type" in str(result.error)

    async def test_type_check_rejects_bool_for_int(self):
        # bool is a subclass of int; the check must still reject it.
        policy = GovernancePolicy.arg_validation("calc", {"n": {"type": "int"}})
        result, _ = await _run(policy, "calc", {"n": True})
        assert isinstance(result, ToolErrorEvent)
        assert "ARG_VALIDATION:n:type" in str(result.error)

    async def test_type_check_accepts_correct_type(self):
        policy = GovernancePolicy.arg_validation("calc", {"n": {"type": "int"}})
        result, _ = await _run(policy, "calc", {"n": 42})
        assert not isinstance(result, ToolErrorEvent)

    async def test_allowed_values_denied(self):
        policy = GovernancePolicy.arg_validation("set_mode", {"mode": {"allowed_values": ["read", "write"]}})
        result, _ = await _run(policy, "set_mode", {"mode": "admin"})
        assert isinstance(result, ToolErrorEvent)
        assert "ARG_VALIDATION:mode:allowed_values" in str(result.error)

    async def test_unconstrained_argument_ignored(self):
        policy = GovernancePolicy.arg_validation("sql_query", {"query": {"max_length": 10}})
        result, _ = await _run(policy, "sql_query", {"query": "SELECT 1", "note": "x" * 500})
        assert not isinstance(result, ToolErrorEvent)

    async def test_absent_constrained_argument_skipped(self):
        policy = GovernancePolicy.arg_validation("sql_query", {"query": {"max_length": 5}})
        result, _ = await _run(policy, "sql_query", {"other": "value"})
        assert not isinstance(result, ToolErrorEvent)

    async def test_only_matching_tool_is_validated(self):
        policy = GovernancePolicy.arg_validation("sql_*", {"query": {"blocked_terms": ["DROP"]}})
        # A tool that does not match the pattern is unaffected.
        result, _ = await _run(policy, "search", {"query": "DROP everything"})
        assert not isinstance(result, ToolErrorEvent)

    async def test_matching_tool_glob_is_validated(self):
        policy = GovernancePolicy.arg_validation("sql_*", {"query": {"blocked_terms": ["DROP"]}})
        result, _ = await _run(policy, "sql_exec", {"query": "DROP TABLE t"})
        assert isinstance(result, ToolErrorEvent)
        assert "ARG_VALIDATION:query:blocked_terms" in str(result.error)

    async def test_decision_recorded_with_risk_score(self):
        policy = GovernancePolicy.arg_validation("sql_query", {"query": {"blocked_terms": ["DROP"]}})
        _, mw = await _run(policy, "sql_query", {"query": "DROP TABLE t"})
        last = mw.decisions[-1]
        assert last.action == "DENY"
        assert last.risk_score == 85
        assert mw.deny_count == 1

    async def test_observe_mode_allows_violation(self):
        policy = GovernancePolicy.arg_validation("sql_query", {"query": {"blocked_terms": ["DROP"]}})
        result, _ = await _run(policy, "sql_query", {"query": "DROP TABLE t"}, mode="OBSERVE")
        assert not isinstance(result, ToolErrorEvent)

    async def test_monitor_mode_records_but_allows(self):
        policy = GovernancePolicy.arg_validation("sql_query", {"query": {"blocked_terms": ["DROP"]}})
        result, mw = await _run(policy, "sql_query", {"query": "DROP TABLE t"}, mode="MONITOR")
        assert not isinstance(result, ToolErrorEvent)
        assert mw.decisions[-1].action == "DENY"


class TestArgValidationConstruction:
    def test_empty_tool_raises(self):
        with pytest.raises(ValueError, match="`tool` must not be empty"):
            GovernancePolicy.arg_validation("", {"query": {"max_length": 10}})

    def test_empty_constraints_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            GovernancePolicy.arg_validation("sql_query", {})

    def test_non_dict_spec_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            GovernancePolicy.arg_validation("sql_query", {"query": ["max_length", 10]})

    def test_unknown_check_raises(self):
        with pytest.raises(ValueError, match="Unknown constraint"):
            GovernancePolicy.arg_validation("sql_query", {"query": {"maxlength": 10}})

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported type"):
            GovernancePolicy.arg_validation("calc", {"n": {"type": "complex"}})
