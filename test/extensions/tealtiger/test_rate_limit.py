# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for TealTiger rate limiting: max tool calls per rolling time window.

The enforcement lives in ``_evaluate`` (deny) and ``_record_rate_limit_calls``
(count allowed calls), so most cases drive the middleware directly through a
per-turn instance rather than a full agent turn — that keeps the window math
explicit and lets a test advance time deterministically by seeding the call
history. A couple of end-to-end cases confirm the policy denies through a real
``Agent`` too.
"""

import time
from typing import Any

import pytest

from ag2 import Agent
from ag2.events import ToolCallEvent
from ag2.extensions.tealtiger import GovernanceMode, GovernancePolicy, TealTigerMiddleware
from ag2.testing import TestConfig


def _call(tool_name: str, **arguments: Any) -> ToolCallEvent:
    import json

    return ToolCallEvent(name=tool_name, arguments=json.dumps(arguments))


def _decide(middleware: TealTigerMiddleware, tool_name: str) -> str:
    """Run one governance evaluation for `tool_name` and record it if allowed.

    Mirrors what ``on_tool_execution`` does around ``_evaluate``: evaluate, and
    on a non-deny (allowed) outcome count the call against the rate-limit buckets.
    Returns the decision action ("ALLOW" / "DENY" / "MONITOR").
    """
    per_turn = middleware(ToolCallEvent(name=tool_name, arguments="{}"), _FakeContext())
    decision = per_turn._evaluate(tool_name, "{}")
    if decision.action != "DENY":
        per_turn._record_rate_limit_calls(tool_name)
    return decision.action


class _FakeContext:
    """Minimal Context stand-in: no agent in dependencies -> agent_name is None."""

    class _Deps:
        def get(self, _key: Any) -> None:
            return None

    dependencies = _Deps()


class TestRateLimitValidation:
    def test_non_positive_max_calls_is_rejected(self):
        with pytest.raises(ValueError, match="max_calls"):
            GovernancePolicy.rate_limit(0, 60)

    def test_boolean_max_calls_is_rejected(self):
        with pytest.raises(ValueError, match="max_calls"):
            GovernancePolicy.rate_limit(True, 60)  # type: ignore[arg-type]

    def test_non_positive_window_is_rejected(self):
        with pytest.raises(ValueError, match="window_seconds"):
            GovernancePolicy.rate_limit(5, 0)

    def test_empty_tool_is_rejected(self):
        with pytest.raises(ValueError, match="tool"):
            GovernancePolicy.rate_limit(5, 60, tool="")

    def test_valid_policy_stores_config(self):
        policy = GovernancePolicy.rate_limit(5, 60, tool="search")
        assert policy.type == "rate_limit"
        assert policy.config == {"max_calls": 5, "window_seconds": 60.0, "tool": "search"}

    def test_global_policy_has_no_tool(self):
        assert GovernancePolicy.rate_limit(30, 60).config["tool"] is None


class TestGlobalRateLimit:
    def test_allows_up_to_the_limit_then_denies(self):
        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(3, window_seconds=60)],
            mode=GovernanceMode.ENFORCE,
        )
        # Three calls (to any tools) are allowed; the fourth within the window is denied.
        assert _decide(governance, "search") == "ALLOW"
        assert _decide(governance, "read_file") == "ALLOW"
        assert _decide(governance, "search") == "ALLOW"
        # The 4th call within the window is denied with the rate-limit reason code.
        per_turn = governance(ToolCallEvent(name="anything", arguments="{}"), _FakeContext())
        decision = per_turn._evaluate("anything", "{}")
        assert decision.action == "DENY"
        assert "RATE_LIMIT_EXCEEDED" in decision.reason_codes

    def test_window_expiry_lets_calls_through_again(self):
        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(2, window_seconds=60)],
            mode=GovernanceMode.ENFORCE,
        )
        # Seed two calls as if they happened ~61s ago — outside a 60s window.
        governance._call_history["*"] = [time.time() - 61, time.time() - 61]
        # They should have aged out, so a fresh call is allowed.
        assert _decide(governance, "search") == "ALLOW"

    def test_recent_calls_still_count(self):
        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(2, window_seconds=60)],
            mode=GovernanceMode.ENFORCE,
        )
        # Two calls 1s ago are inside the 60s window -> the next is denied.
        governance._call_history["*"] = [time.time() - 1, time.time() - 1]
        assert _decide(governance, "search") == "DENY"


class TestPerToolRateLimit:
    def test_limit_is_isolated_to_the_matching_tool(self):
        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(2, window_seconds=60, tool="search")],
            mode=GovernanceMode.ENFORCE,
        )
        assert _decide(governance, "search") == "ALLOW"
        assert _decide(governance, "search") == "ALLOW"
        # search is now capped...
        assert _decide(governance, "search") == "DENY"
        # ...but a different tool is untouched by this per-tool limit.
        assert _decide(governance, "read_file") == "ALLOW"
        assert _decide(governance, "read_file") == "ALLOW"
        assert _decide(governance, "read_file") == "ALLOW"

    def test_pattern_caps_a_family_together(self):
        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(2, window_seconds=60, tool="db_*")],
            mode=GovernanceMode.ENFORCE,
        )
        assert _decide(governance, "db_read") == "ALLOW"
        assert _decide(governance, "db_write") == "ALLOW"
        # db_read + db_write share the db_* bucket, so the third db_* call is denied.
        assert _decide(governance, "db_delete") == "DENY"

    def test_global_and_per_tool_limits_compose(self):
        governance = TealTigerMiddleware(
            policies=[
                GovernancePolicy.rate_limit(10, window_seconds=60),  # generous global
                GovernancePolicy.rate_limit(1, window_seconds=60, tool="send_email"),  # tight per-tool
            ],
            mode=GovernanceMode.ENFORCE,
        )
        assert _decide(governance, "send_email") == "ALLOW"
        # The per-tool cap denies the 2nd email even though the global budget is fine.
        assert _decide(governance, "send_email") == "DENY"
        # Other tools still flow under the global limit.
        assert _decide(governance, "search") == "ALLOW"


class TestRateLimitRespectsMode:
    @pytest.mark.asyncio
    async def test_observe_never_denies(self):
        def search(query: str) -> str:
            return "results"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(1, window_seconds=60)],
            mode=GovernanceMode.OBSERVE,
        )
        # Bucket already over the limit — but OBSERVE short-circuits before policy
        # evaluation, so the call still runs and is only recorded as a passthrough.
        governance._call_history["*"] = [time.time(), time.time()]
        agent = Agent(
            "assistant",
            config=TestConfig(_call("search", query="hi"), "Done."),
            tools=[search],
            middleware=[governance],
        )

        reply = await agent.ask("search")

        assert reply.body == "Done."
        assert governance.deny_count == 0

    def test_monitor_records_denial_but_would_not_block(self):
        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(1, window_seconds=60)],
            mode=GovernanceMode.MONITOR,
        )
        governance._call_history["*"] = [time.time()]
        per_turn = governance(ToolCallEvent(name="search", arguments="{}"), _FakeContext())
        decision = per_turn._evaluate("search", "{}")
        # In MONITOR the decision still reads DENY (the violation is real and logged),
        # but on_tool_execution only turns a DENY into a block in ENFORCE.
        assert decision.action == "DENY"
        assert "RATE_LIMIT_EXCEEDED" in decision.reason_codes


@pytest.mark.asyncio
async def test_rate_limit_blocks_a_real_agent_turn_in_enforce():
    def search(query: str) -> str:
        return "results"

    governance = TealTigerMiddleware(
        policies=[GovernancePolicy.rate_limit(5, window_seconds=60, tool="search")],
        mode=GovernanceMode.ENFORCE,
    )
    # Pre-fill the search bucket to its limit so the very next call is denied.
    governance._call_history["search"] = [time.time()] * 5

    agent = Agent(
        "assistant",
        config=TestConfig(_call("search", query="hello"), "Done."),
        tools=[search],
        middleware=[governance],
    )

    with pytest.raises(Exception, match=r"\[GOVERNANCE DENIED\].*RATE_LIMIT_EXCEEDED"):
        await agent.ask("search for hello")

    assert governance.deny_count == 1


class TestSharedBucketIsCountedOnce:
    """Regression for PR #3246 review: policies sharing a bucket key must not
    double-count a single real call.
    """

    def test_two_global_limits_share_one_bucket_without_double_counting(self):
        # Both policies are global -> both use the "*" bucket. A single real call
        # must add exactly one timestamp, not one per policy.
        governance = TealTigerMiddleware(
            policies=[
                GovernancePolicy.rate_limit(10, window_seconds=60),
                GovernancePolicy.rate_limit(100, window_seconds=3600),
            ],
            mode=GovernanceMode.ENFORCE,
        )

        for _ in range(5):
            assert _decide(governance, "search") == "ALLOW"

        # Five real calls -> five timestamps in the shared bucket (not ten).
        assert len(governance._call_history["*"]) == 5

        # The 10/min limit must still have five allowances left, not be tripped.
        for _ in range(5):
            assert _decide(governance, "search") == "ALLOW"
        # The 11th real call is the one that trips the 10/min limit.
        per_turn = governance(ToolCallEvent(name="search", arguments="{}"), _FakeContext())
        assert per_turn._evaluate("search", "{}").action == "DENY"

    def test_two_policies_with_same_tool_pattern_share_one_bucket(self):
        governance = TealTigerMiddleware(
            policies=[
                GovernancePolicy.rate_limit(4, window_seconds=60, tool="db_*"),
                GovernancePolicy.rate_limit(50, window_seconds=3600, tool="db_*"),
            ],
            mode=GovernanceMode.ENFORCE,
        )

        for _ in range(4):
            assert _decide(governance, "db_read") == "ALLOW"

        # Four real calls -> four timestamps in the shared "db_*" bucket, not eight.
        assert len(governance._call_history["db_*"]) == 4
        # The 4/min limit trips on the 5th real call, exactly at its threshold.
        per_turn = governance(ToolCallEvent(name="db_write", arguments="{}"), _FakeContext())
        assert per_turn._evaluate("db_write", "{}").action == "DENY"
