# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for TealTiger tool blocklist policy.

The blocklist is the complement of the allowlist: every tool is permitted
except those matching a blocked pattern. Patterns match tool names via
``fnmatch``, so both exact names and globs (``delete_*``) are supported.

Each case scripts a real agent's turn with ``TestConfig`` so the middleware runs
where it does in production. Every tool appends to ``ran`` before returning, so
"the call was blocked" is observed as the tool body never having executed. In
ENFORCE mode a denial fails the turn, so ``ask`` raises — the same way any tool
error does.
"""

import pytest

from ag2 import Agent
from ag2.events import ToolCallEvent
from ag2.extensions.tealtiger import GovernanceMode, GovernancePolicy, TealTigerMiddleware
from ag2.testing import TestConfig


@pytest.mark.asyncio
class TestToolBlocklist:
    async def test_tool_outside_the_blocklist_runs(self):
        ran: list[str] = []

        def search() -> str:
            ran.append("search")
            return "found it"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_blocklist(["delete_all", "shell"])],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(ToolCallEvent(name="search", arguments="{}"), "Done."),
            tools=[search],
            middleware=[governance],
        )

        reply = await agent.ask("look it up")

        assert reply.body == "Done."
        assert ran == ["search"]
        assert governance.deny_count == 0

    async def test_blocked_tool_never_executes(self):
        ran: list[str] = []

        def delete_all() -> str:
            ran.append("delete_all")
            return "deleted everything"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_blocklist(["delete_all"])],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(ToolCallEvent(name="delete_all", arguments="{}"), "Done."),
            tools=[delete_all],
            middleware=[governance],
        )

        with pytest.raises(Exception, match=r"\[GOVERNANCE DENIED\].*TOOL_BLOCKED"):
            await agent.ask("clean up")

        assert ran == []

    async def test_glob_pattern_blocks_matching_tools(self):
        ran: list[str] = []

        def delete_database() -> str:
            ran.append("delete_database")
            return "dropped"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_blocklist(["delete_*"])],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(ToolCallEvent(name="delete_database", arguments="{}"), "Done."),
            tools=[delete_database],
            middleware=[governance],
        )

        with pytest.raises(Exception, match="TOOL_BLOCKED"):
            await agent.ask("clean up")

        assert ran == []

    async def test_glob_pattern_leaves_non_matching_tools_alone(self):
        ran: list[str] = []

        def read_file() -> str:
            ran.append("read_file")
            return "file contents"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_blocklist(["delete_*"])],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(ToolCallEvent(name="read_file", arguments="{}"), "Done."),
            tools=[read_file],
            middleware=[governance],
        )

        reply = await agent.ask("read it")

        assert reply.body == "Done."
        assert ran == ["read_file"]

    async def test_denial_is_recorded_with_a_receipt(self):
        def shell() -> str:
            return "shell output"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_blocklist(["shell"])],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(ToolCallEvent(name="shell", arguments="{}"), "Done."),
            tools=[shell],
            middleware=[governance],
        )

        with pytest.raises(Exception, match="TOOL_BLOCKED"):
            await agent.ask("run it")

        decision = governance.decisions[-1]
        assert decision.action == "DENY"
        assert decision.reason_codes == ["TOOL_BLOCKED"]
        assert decision.risk_score == 80
        assert decision.tool_name == "shell"
        assert decision.agent_name == "assistant"
        assert governance.deny_count == 1
        assert governance.receipts[-1].execution_outcome == "blocked"

    async def test_observe_mode_skips_evaluation_entirely(self):
        ran: list[str] = []

        def delete_all() -> str:
            ran.append("delete_all")
            return "deleted everything"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_blocklist(["delete_all"])],
            mode=GovernanceMode.OBSERVE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(ToolCallEvent(name="delete_all", arguments="{}"), "Done."),
            tools=[delete_all],
            middleware=[governance],
        )

        reply = await agent.ask("clean up")

        assert reply.body == "Done."
        assert ran == ["delete_all"]
        # OBSERVE short-circuits before any policy runs, so no TOOL_BLOCKED is ever
        # recorded — the audit trail shows only the pass-through, as with the allowlist.
        assert governance.decisions[-1].action == "ALLOW"
        assert governance.decisions[-1].reason_codes == ["OBSERVE_PASSTHROUGH"]

    async def test_monitor_mode_records_the_denial_but_lets_the_call_through(self):
        ran: list[str] = []

        def delete_all() -> str:
            ran.append("delete_all")
            return "deleted everything"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.tool_blocklist(["delete_all"])],
            mode=GovernanceMode.MONITOR,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(ToolCallEvent(name="delete_all", arguments="{}"), "Done."),
            tools=[delete_all],
            middleware=[governance],
        )

        reply = await agent.ask("clean up")

        assert reply.body == "Done."
        assert ran == ["delete_all"]
        assert governance.decisions[-1].action == "DENY"
        assert governance.decisions[-1].reason_codes == ["TOOL_BLOCKED"]


# The allowlist permits read_*; the blocklist carves out read_secrets. Whichever
# policy denies is the one reported, whatever its position in the list — so both
# orderings have to behave identically.
_COMPOSED_POLICIES = {
    "allowlist-first": [
        GovernancePolicy.tool_allowlist(["read_*"]),
        GovernancePolicy.tool_blocklist(["read_secrets"]),
    ],
    "blocklist-first": [
        GovernancePolicy.tool_blocklist(["read_secrets"]),
        GovernancePolicy.tool_allowlist(["read_*"]),
    ],
}


@pytest.mark.asyncio
@pytest.mark.parametrize("order", list(_COMPOSED_POLICIES))
class TestAllowlistAndBlocklistCompose:
    async def test_a_tool_both_policies_accept_runs(self, order: str):
        ran: list[str] = []

        def read_file() -> str:
            ran.append("read_file")
            return "file contents"

        governance = TealTigerMiddleware(policies=_COMPOSED_POLICIES[order], mode=GovernanceMode.ENFORCE)
        agent = Agent(
            "assistant",
            config=TestConfig(ToolCallEvent(name="read_file", arguments="{}"), "Done."),
            tools=[read_file],
            middleware=[governance],
        )

        reply = await agent.ask("read it")

        assert reply.body == "Done."
        assert ran == ["read_file"]

    async def test_the_blocklist_carve_out_is_denied(self, order: str):
        ran: list[str] = []

        def read_secrets() -> str:
            ran.append("read_secrets")
            return "hunter2"

        governance = TealTigerMiddleware(policies=_COMPOSED_POLICIES[order], mode=GovernanceMode.ENFORCE)
        agent = Agent(
            "assistant",
            config=TestConfig(ToolCallEvent(name="read_secrets", arguments="{}"), "Done."),
            tools=[read_secrets],
            middleware=[governance],
        )

        with pytest.raises(Exception, match="TOOL_BLOCKED"):
            await agent.ask("read it")

        assert ran == []
        assert governance.decisions[-1].reason_codes == ["TOOL_BLOCKED"]

    async def test_a_tool_outside_the_allowlist_still_reports_the_allowlist(self, order: str):
        # Composition must not mask the allowlist: a tool that is neither allowed
        # nor blocked is denied, under the allowlist's own reason code.
        ran: list[str] = []

        def shell() -> str:
            ran.append("shell")
            return "shell output"

        governance = TealTigerMiddleware(policies=_COMPOSED_POLICIES[order], mode=GovernanceMode.ENFORCE)
        agent = Agent(
            "assistant",
            config=TestConfig(ToolCallEvent(name="shell", arguments="{}"), "Done."),
            tools=[shell],
            middleware=[governance],
        )

        with pytest.raises(Exception, match="TOOL_NOT_ALLOWED"):
            await agent.ask("run it")

        assert ran == []
        assert governance.decisions[-1].reason_codes == ["TOOL_NOT_ALLOWED"]


def test_empty_blocklist_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        GovernancePolicy.tool_blocklist([])
