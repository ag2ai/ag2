# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for TealTiger per-tool argument validation.

Where the allow/blocklists govern *which* tools run, `arg_validation` governs
*what* a matched tool is called with — max/min length, type, blocked terms,
blocked regex patterns, and allowed values — to defend against dangerous values
such as SQL injection and path traversal.

Each case scripts a real agent's turn with ``TestConfig``, so the policy runs
where it does in production and the argument values come in the way a model
would actually send them: as the JSON string on a ``ToolCallEvent``. Every tool
appends to ``ran`` before returning, so "the call was blocked" is observed as
the tool body never having executed. In ENFORCE mode a denial fails the turn,
so ``ask`` raises.
"""

import json
from typing import Any

import pytest

from ag2 import Agent
from ag2.events import ToolCallEvent
from ag2.extensions.tealtiger import GovernanceMode, GovernancePolicy, TealTigerMiddleware
from ag2.extensions.tealtiger.types import ARG_CHECKS
from ag2.testing import TestConfig


def _call(tool_name: str, **arguments: Any) -> ToolCallEvent:
    """The tool call a model would emit for these arguments."""
    return ToolCallEvent(name=tool_name, arguments=json.dumps(arguments))


@pytest.mark.asyncio
class TestArgumentConstraintsAreEnforced:
    async def test_an_argument_within_its_constraints_runs(self):
        ran: list[str] = []

        def sql_query(query: str) -> str:
            ran.append(query)
            return "3 rows"

        governance = TealTigerMiddleware(
            policies=[
                GovernancePolicy.arg_validation("sql_query", {"query": {"max_length": 500, "blocked_terms": ["DROP"]}})
            ],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("sql_query", query="SELECT name FROM users"), "Done."),
            tools=[sql_query],
            middleware=[governance],
        )

        reply = await agent.ask("who is in the table")

        assert reply.body == "Done."
        assert ran == ["SELECT name FROM users"]
        assert governance.deny_count == 0

    async def test_a_blocked_term_stops_the_call(self):
        ran: list[str] = []

        def sql_query(query: str) -> str:
            ran.append(query)
            return "dropped"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.arg_validation("sql_query", {"query": {"blocked_terms": ["DROP", ";--"]}})],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("sql_query", query="drop table users"), "Done."),
            tools=[sql_query],
            middleware=[governance],
        )

        # Blocked terms match case-insensitively, so lowercase `drop` is caught by `DROP`.
        with pytest.raises(Exception, match=r"\[GOVERNANCE DENIED\].*ARG_VALIDATION:query:blocked_terms"):
            await agent.ask("clean up the table")

        assert ran == []

    async def test_a_blocked_pattern_stops_path_traversal(self):
        ran: list[str] = []

        def read_file(path: str) -> str:
            ran.append(path)
            return "root:x:0:0"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.arg_validation("read_file", {"path": {"blocked_patterns": [r"\.\.[\\/]"]}})],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("read_file", path="../../etc/passwd"), "Done."),
            tools=[read_file],
            middleware=[governance],
        )

        with pytest.raises(Exception, match="ARG_VALIDATION:path:blocked_patterns"):
            await agent.ask("read the file")

        assert ran == []

    async def test_a_pattern_that_does_not_match_lets_the_call_through(self):
        ran: list[str] = []

        def read_file(path: str) -> str:
            ran.append(path)
            return "file contents"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.arg_validation("read_file", {"path": {"blocked_patterns": [r"\.\.[\\/]"]}})],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("read_file", path="reports/q3.txt"), "Done."),
            tools=[read_file],
            middleware=[governance],
        )

        reply = await agent.ask("read the report")

        assert reply.body == "Done."
        assert ran == ["reports/q3.txt"]

    async def test_an_oversized_argument_stops_the_call(self):
        ran: list[str] = []

        def sql_query(query: str) -> str:
            ran.append(query)
            return "rows"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.arg_validation("sql_query", {"query": {"max_length": 20}})],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("sql_query", query="SELECT * FROM a_very_wide_table_indeed"), "Done."),
            tools=[sql_query],
            middleware=[governance],
        )

        with pytest.raises(Exception, match="ARG_VALIDATION:query:max_length"):
            await agent.ask("run the query")

        assert ran == []

    async def test_a_value_outside_allowed_values_stops_the_call(self):
        ran: list[str] = []

        def open_file(mode: str) -> str:
            ran.append(mode)
            return "opened"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.arg_validation("open_file", {"mode": {"allowed_values": ["read", "list"]}})],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("open_file", mode="write"), "Done."),
            tools=[open_file],
            middleware=[governance],
        )

        with pytest.raises(Exception, match="ARG_VALIDATION:mode:allowed_values"):
            await agent.ask("open it for writing")

        assert ran == []

    async def test_a_bool_does_not_satisfy_an_int_constraint(self):
        # bool is a subclass of int in Python, so `True` would pass a naive
        # isinstance check and reach a tool expecting a real count.
        ran: list[bool] = []

        def fetch(limit: int) -> str:
            ran.append(limit)
            return "fetched"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.arg_validation("fetch", {"limit": {"type": "int"}})],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("fetch", limit=True), "Done."),
            tools=[fetch],
            middleware=[governance],
        )

        with pytest.raises(Exception, match="ARG_VALIDATION:limit:type"):
            await agent.ask("fetch some")

        assert ran == []

    async def test_a_real_int_satisfies_an_int_constraint(self):
        ran: list[int] = []

        def fetch(limit: int) -> str:
            ran.append(limit)
            return "fetched"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.arg_validation("fetch", {"limit": {"type": "int"}})],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("fetch", limit=10), "Done."),
            tools=[fetch],
            middleware=[governance],
        )

        reply = await agent.ask("fetch some")

        assert reply.body == "Done."
        assert ran == [10]


@pytest.mark.asyncio
class TestWhatTheConstraintsApplyTo:
    async def test_a_glob_covers_every_matching_tool(self):
        ran: list[str] = []

        def sql_exec(query: str) -> str:
            ran.append(query)
            return "executed"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.arg_validation("sql_*", {"query": {"blocked_terms": ["DROP"]}})],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("sql_exec", query="DROP TABLE users"), "Done."),
            tools=[sql_exec],
            middleware=[governance],
        )

        with pytest.raises(Exception, match="ARG_VALIDATION:query:blocked_terms"):
            await agent.ask("run it")

        assert ran == []

    async def test_a_tool_the_pattern_does_not_match_is_left_alone(self):
        ran: list[str] = []

        def search(query: str) -> str:
            ran.append(query)
            return "found"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.arg_validation("sql_*", {"query": {"blocked_terms": ["DROP"]}})],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("search", query="DROP TABLE users"), "Done."),
            tools=[search],
            middleware=[governance],
        )

        reply = await agent.ask("look it up")

        assert reply.body == "Done."
        assert ran == ["DROP TABLE users"]

    async def test_an_unconstrained_argument_passes_through(self):
        ran: list[tuple[str, str]] = []

        def sql_query(query: str, label: str) -> str:
            ran.append((query, label))
            return "rows"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.arg_validation("sql_query", {"query": {"blocked_terms": ["DROP"]}})],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("sql_query", query="SELECT 1", label="DROP this label"), "Done."),
            tools=[sql_query],
            middleware=[governance],
        )

        reply = await agent.ask("run it")

        # `label` is not in `constraints`, so its `DROP` is none of the policy's business.
        assert reply.body == "Done."
        assert ran == [("SELECT 1", "DROP this label")]

    async def test_a_constrained_argument_absent_from_the_call_is_skipped(self):
        ran: list[str] = []

        def sql_query(query: str) -> str:
            ran.append(query)
            return "rows"

        governance = TealTigerMiddleware(
            policies=[
                GovernancePolicy.arg_validation(
                    "sql_query",
                    {"query": {"blocked_terms": ["DROP"]}, "schema": {"allowed_values": ["public"]}},
                )
            ],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("sql_query", query="SELECT 1"), "Done."),
            tools=[sql_query],
            middleware=[governance],
        )

        reply = await agent.ask("run it")

        # `schema` was never sent, so its `allowed_values` has nothing to judge.
        assert reply.body == "Done."
        assert ran == ["SELECT 1"]

    async def test_arguments_that_are_not_a_mapping_are_scanned_as_a_whole(self):
        # A model can emit a JSON array rather than an object, leaving no argument
        # names to read. The policy stays fail-closed: the term checks run over the
        # whole serialized call, reported against `*` because no single argument
        # can honestly be blamed.
        ran: list[Any] = []

        def sql_query(query: str) -> str:
            ran.append(query)
            return "rows"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.arg_validation("sql_query", {"query": {"blocked_terms": ["DROP"]}})],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(ToolCallEvent(name="sql_query", arguments='["DROP TABLE users"]'), "Done."),
            tools=[sql_query],
            middleware=[governance],
        )

        with pytest.raises(Exception, match=r"ARG_VALIDATION:\*:blocked_terms"):
            await agent.ask("run it")

        assert ran == []


# One violating (spec, value) pair per check the vocabulary advertises. The
# equality test below fails if a check is ever added to `ARG_CHECKS` without a
# case here, and each case proves the evaluator actually wires that check up.
_VIOLATIONS: dict[str, tuple[dict[str, Any], Any]] = {
    "max_length": ({"max_length": 5}, "far too long"),
    "min_length": ({"min_length": 5}, "hi"),
    "type": ({"type": "int"}, "not a number"),
    "blocked_terms": ({"blocked_terms": ["DROP"]}, "drop table users"),
    "blocked_patterns": ({"blocked_patterns": [r"\.\.[\\/]"]}, "../etc/passwd"),
    "allowed_values": ({"allowed_values": ["read"]}, "write"),
}


def test_every_advertised_check_has_a_case():
    assert set(_VIOLATIONS) == set(ARG_CHECKS)


@pytest.mark.asyncio
@pytest.mark.parametrize("check", sorted(_VIOLATIONS))
async def test_every_advertised_check_can_deny(check: str):
    spec, value = _VIOLATIONS[check]
    ran: list[str] = []

    def probe(value: str) -> str:
        ran.append(value)
        return "probed"

    governance = TealTigerMiddleware(
        policies=[GovernancePolicy.arg_validation("probe", {"value": spec})],
        mode=GovernanceMode.ENFORCE,
    )
    agent = Agent(
        "assistant",
        config=TestConfig(_call("probe", value=value), "Done."),
        tools=[probe],
        middleware=[governance],
    )

    with pytest.raises(Exception, match=f"ARG_VALIDATION:value:{check}"):
        await agent.ask("probe it")

    assert ran == []


@pytest.mark.asyncio
class TestTheDecisionTrail:
    async def test_a_denial_is_recorded_with_a_receipt(self):
        def sql_query(query: str) -> str:
            return "rows"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.arg_validation("sql_query", {"query": {"max_length": 10}})],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("sql_query", query="SELECT * FROM everything"), "Done."),
            tools=[sql_query],
            middleware=[governance],
        )

        with pytest.raises(Exception, match="ARG_VALIDATION"):
            await agent.ask("run it")

        decision = governance.decisions[-1]
        assert decision.action == "DENY"
        assert decision.reason_codes == ["ARG_VALIDATION:query:max_length"]
        assert decision.risk_score == 85
        assert decision.tool_name == "sql_query"
        assert decision.agent_name == "assistant"
        assert governance.deny_count == 1
        assert governance.receipts[-1].execution_outcome == "blocked"

    async def test_observe_mode_skips_evaluation_entirely(self):
        ran: list[str] = []

        def sql_query(query: str) -> str:
            ran.append(query)
            return "dropped"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.arg_validation("sql_query", {"query": {"blocked_terms": ["DROP"]}})],
            mode=GovernanceMode.OBSERVE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("sql_query", query="DROP TABLE users"), "Done."),
            tools=[sql_query],
            middleware=[governance],
        )

        reply = await agent.ask("clean up")

        assert reply.body == "Done."
        assert ran == ["DROP TABLE users"]
        assert governance.decisions[-1].action == "ALLOW"
        assert governance.decisions[-1].reason_codes == ["OBSERVE_PASSTHROUGH"]

    async def test_monitor_mode_records_the_denial_but_lets_the_call_through(self):
        ran: list[str] = []

        def sql_query(query: str) -> str:
            ran.append(query)
            return "dropped"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.arg_validation("sql_query", {"query": {"blocked_terms": ["DROP"]}})],
            mode=GovernanceMode.MONITOR,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(_call("sql_query", query="DROP TABLE users"), "Done."),
            tools=[sql_query],
            middleware=[governance],
        )

        reply = await agent.ask("clean up")

        assert reply.body == "Done."
        assert ran == ["DROP TABLE users"]
        assert governance.decisions[-1].action == "DENY"
        assert governance.decisions[-1].reason_codes == ["ARG_VALIDATION:query:blocked_terms"]


class TestAMalformedSpecIsRejectedAtConstruction:
    """A typo must not survive to become a policy that quietly checks nothing —
    or one that crashes the governance path on the first call it evaluates."""

    def test_empty_tool_raises(self):
        with pytest.raises(ValueError, match="`tool` must not be empty"):
            GovernancePolicy.arg_validation("", {"query": {"max_length": 10}})

    def test_empty_constraints_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            GovernancePolicy.arg_validation("sql_query", {})

    def test_non_dict_spec_raises(self):
        with pytest.raises(ValueError, match="must be a dict, got list"):
            GovernancePolicy.arg_validation("sql_query", {"query": ["max_length"]})

    def test_empty_argument_spec_raises(self):
        with pytest.raises(ValueError, match="Constraint spec for argument 'query' must not be empty"):
            GovernancePolicy.arg_validation("sql_query", {"query": {}})

    def test_unknown_check_raises(self):
        with pytest.raises(ValueError, match="Unknown constraint\\(s\\) for argument 'query': max_len"):
            GovernancePolicy.arg_validation("sql_query", {"query": {"max_len": 10}})

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported type 'complex'"):
            GovernancePolicy.arg_validation("sql_query", {"query": {"type": "complex"}})

    def test_non_string_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported type"):
            GovernancePolicy.arg_validation("sql_query", {"query": {"type": ["int"]}})

    def test_uncompilable_pattern_raises(self):
        with pytest.raises(ValueError, match="Invalid regex in `blocked_patterns`"):
            GovernancePolicy.arg_validation("read_file", {"path": {"blocked_patterns": ["[unclosed"]}})

    def test_non_integer_length_raises(self):
        with pytest.raises(ValueError, match="`max_length` for argument 'query' must be a non-negative integer"):
            GovernancePolicy.arg_validation("sql_query", {"query": {"max_length": "ten"}})

    def test_negative_length_raises(self):
        with pytest.raises(ValueError, match="must be a non-negative integer"):
            GovernancePolicy.arg_validation("sql_query", {"query": {"min_length": -1}})

    def test_max_length_below_min_length_raises(self):
        with pytest.raises(ValueError, match="no value could satisfy both"):
            GovernancePolicy.arg_validation("sql_query", {"query": {"max_length": 2, "min_length": 5}})

    def test_empty_term_list_raises(self):
        with pytest.raises(ValueError, match="`blocked_terms` for argument 'query' must be a non-empty list"):
            GovernancePolicy.arg_validation("sql_query", {"query": {"blocked_terms": []}})

    def test_non_string_term_raises(self):
        with pytest.raises(ValueError, match="must contain only strings"):
            GovernancePolicy.arg_validation("sql_query", {"query": {"blocked_terms": [42]}})

    def test_patterns_are_compiled_once_at_construction(self):
        # Evaluation must never pay for compilation, nor depend on the
        # interpreter's regex cache still holding the pattern.
        policy = GovernancePolicy.arg_validation("read_file", {"path": {"blocked_patterns": [r"\.\.[\\/]"]}})

        compiled = policy.config["constraints"]["path"]["blocked_patterns"]

        assert [p.pattern for p in compiled] == [r"\.\.[\\/]"]

    def test_the_callers_constraints_are_not_mutated(self):
        constraints: dict[str, dict[str, Any]] = {"path": {"blocked_patterns": [r"\.\."]}}

        GovernancePolicy.arg_validation("read_file", constraints)

        assert constraints == {"path": {"blocked_patterns": [r"\.\."]}}
