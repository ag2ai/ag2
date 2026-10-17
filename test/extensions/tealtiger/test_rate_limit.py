# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for TealTiger rate limiting: how fast tool calls may happen.

Everything runs through a real ``Agent`` with scripted turns, so what is asserted
is what a user sees — the reply and the recorded decisions — never the window
bookkeeping underneath. Time is made to pass by giving a tool a small delay
rather than by seeding timestamps, which keeps the windows honest.

`raise_tool_errors=False` models a real provider throughout: a denied call comes
back to the model as an ordinary failed tool result and the agent carries on,
which is exactly why refusing an over-limit call costs a model round-trip.
"""

import asyncio
import time

import pytest
from dirty_equals import IsList, IsPartialDataclass, IsStr

from ag2 import Agent
from ag2.events import ToolCallEvent
from ag2.extensions.tealtiger import GovernanceMode, GovernancePolicy, TealTigerMiddleware
from ag2.testing import TestConfig, TrackingConfig


class TestRateLimitValidation:
    def test_non_positive_max_calls_is_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            GovernancePolicy.rate_limit(0, window_seconds=60)

    def test_boolean_max_calls_is_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            GovernancePolicy.rate_limit(True, window_seconds=60)

    def test_non_positive_window_is_rejected(self):
        with pytest.raises(ValueError, match="positive number"):
            GovernancePolicy.rate_limit(5, window_seconds=0)

    def test_empty_tool_is_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            GovernancePolicy.rate_limit(5, window_seconds=60, tool="")

    def test_unknown_on_exceeded_is_rejected(self):
        with pytest.raises(ValueError, match="'wait' or 'deny'"):
            GovernancePolicy.rate_limit(5, window_seconds=60, on_exceeded="throttle")

    def test_non_positive_max_wait_is_rejected(self):
        with pytest.raises(ValueError, match="positive number"):
            GovernancePolicy.rate_limit(5, window_seconds=60, max_wait_seconds=0)


@pytest.mark.asyncio
class TestGlobalRateLimit:
    async def test_allows_up_to_the_limit_then_denies(self):
        def search(query: str = "") -> str:
            return "results"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(2, window_seconds=60, on_exceeded="deny")],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(
                ToolCallEvent(name="search", arguments="{}"),
                ToolCallEvent(name="search", arguments="{}"),
                ToolCallEvent(name="search", arguments="{}"),
                "Done.",
                raise_tool_errors=False,
            ),
            tools=[search],
            middleware=[governance],
        )

        reply = await agent.ask("search repeatedly")

        assert reply.body == "Done."
        assert governance.decisions == [
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="DENY", reason_codes=["RATE_LIMIT_EXCEEDED"], risk_score=70),
        ]

    async def test_the_window_refills_so_a_paced_agent_is_never_denied(self):
        async def slow(query: str = "") -> str:
            # Outlasts the window, so each call has aged out before the next starts.
            await asyncio.sleep(0.08)
            return "results"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(1, window_seconds=0.05, on_exceeded="deny")],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(
                ToolCallEvent(name="slow", arguments="{}"),
                ToolCallEvent(name="slow", arguments="{}"),
                ToolCallEvent(name="slow", arguments="{}"),
                "Done.",
                raise_tool_errors=False,
            ),
            tools=[slow],
            middleware=[governance],
        )

        await agent.ask("call slowly")

        assert governance.decisions == [
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="ALLOW"),
        ]

    async def test_a_denied_call_is_handed_back_to_the_model_as_a_tool_error(self):
        """Why refusing is the expensive option: the model gets the refusal and answers it.

        The denial arrives as an ordinary failed tool result, so an agent in a
        loop simply calls again — each refusal costing a model round-trip.
        """

        def search(query: str = "") -> str:
            return "results"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(1, window_seconds=60, on_exceeded="deny")],
            mode=GovernanceMode.ENFORCE,
        )
        tracking = TrackingConfig(
            TestConfig(
                ToolCallEvent(name="search", arguments="{}"),
                ToolCallEvent(name="search", arguments="{}"),
                "Done.",
                raise_tool_errors=False,
            )
        )
        agent = Agent("assistant", config=tracking, tools=[search], middleware=[governance])

        await agent.ask("search twice")

        assert "GOVERNANCE DENIED" in str([call.args[0] for call in tracking.mock.call_args_list])
        # The denial is reported as exactly that, with no stale allow verdict beside it.
        assert governance.decisions == [
            IsPartialDataclass(action="ALLOW", reason_codes=["POLICY_ALLOW"]),
            IsPartialDataclass(action="DENY", reason_codes=["RATE_LIMIT_EXCEEDED"]),
        ]


@pytest.mark.asyncio
class TestPerToolRateLimit:
    async def test_limit_is_isolated_to_the_matching_tool(self):
        def search(query: str = "") -> str:
            return "results"

        def read_file(path: str = "") -> str:
            return "contents"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(1, window_seconds=60, tool="search", on_exceeded="deny")],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(
                ToolCallEvent(name="search", arguments="{}"),
                ToolCallEvent(name="search", arguments="{}"),
                ToolCallEvent(name="read_file", arguments="{}"),
                ToolCallEvent(name="read_file", arguments="{}"),
                "Done.",
                raise_tool_errors=False,
            ),
            tools=[search, read_file],
            middleware=[governance],
        )

        await agent.ask("mix tools")

        # Only the second search trips; read_file is untouched by a search-scoped limit.
        assert governance.decisions == [
            IsPartialDataclass(tool_name="search", action="ALLOW"),
            IsPartialDataclass(tool_name="search", action="DENY"),
            IsPartialDataclass(tool_name="read_file", action="ALLOW"),
            IsPartialDataclass(tool_name="read_file", action="ALLOW"),
        ]

    async def test_pattern_caps_a_family_together(self):
        def db_read(key: str = "") -> str:
            return "row"

        def db_write(key: str = "") -> str:
            return "ok"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(2, window_seconds=60, tool="db_*", on_exceeded="deny")],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(
                ToolCallEvent(name="db_read", arguments="{}"),
                ToolCallEvent(name="db_write", arguments="{}"),
                ToolCallEvent(name="db_read", arguments="{}"),
                "Done.",
                raise_tool_errors=False,
            ),
            tools=[db_read, db_write],
            middleware=[governance],
        )

        await agent.ask("touch the db")

        # db_read and db_write share the db_* scope, so the third db_* call is denied.
        assert governance.decisions == [
            IsPartialDataclass(tool_name="db_read", action="ALLOW"),
            IsPartialDataclass(tool_name="db_write", action="ALLOW"),
            IsPartialDataclass(tool_name="db_read", action="DENY"),
        ]

    async def test_global_and_per_tool_limits_compose(self):
        def send_email(to: str = "") -> str:
            return "sent"

        def search(query: str = "") -> str:
            return "results"

        governance = TealTigerMiddleware(
            policies=[
                GovernancePolicy.rate_limit(10, window_seconds=60, on_exceeded="deny"),
                GovernancePolicy.rate_limit(1, window_seconds=60, tool="send_email", on_exceeded="deny"),
            ],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(
                ToolCallEvent(name="send_email", arguments="{}"),
                ToolCallEvent(name="send_email", arguments="{}"),
                ToolCallEvent(name="search", arguments="{}"),
                "Done.",
                raise_tool_errors=False,
            ),
            tools=[send_email, search],
            middleware=[governance],
        )

        await agent.ask("email then search")

        # The per-tool cap denies the 2nd email though the global budget is fine,
        # and other tools still flow under the global limit.
        assert governance.decisions == [
            IsPartialDataclass(tool_name="send_email", action="ALLOW"),
            IsPartialDataclass(tool_name="send_email", action="DENY"),
            IsPartialDataclass(tool_name="search", action="ALLOW"),
        ]


@pytest.mark.asyncio
class TestRateLimitWaits:
    """The default: an over-limit call is held until the window refills.

    This is the behaviour the feature rests on. Refusing an over-limit call hands
    the model a tool error, and an agent stuck in a loop answers it by calling
    again — so a refusal converts a throttled tool call into a model round-trip
    and burns tokens faster than the call it replaced. Holding the call does not.
    """

    async def test_over_limit_call_is_delayed_not_denied(self):
        def search(query: str = "") -> str:
            return "results"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(1, window_seconds=0.1)],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(
                ToolCallEvent(name="search", arguments="{}"),
                ToolCallEvent(name="search", arguments="{}"),
                "Done.",
                raise_tool_errors=False,
            ),
            tools=[search],
            middleware=[governance],
        )

        started = time.monotonic()
        reply = await agent.ask("search twice")
        elapsed = time.monotonic() - started

        # The second call waited out the window instead of being refused, and the
        # wait it served is on the record rather than being silent latency.
        assert reply.body == "Done."
        assert elapsed >= 0.08
        assert governance.decisions == [
            IsPartialDataclass(action="ALLOW", reason_codes=["POLICY_ALLOW"]),
            IsPartialDataclass(
                action="ALLOW",
                reason_codes=["POLICY_ALLOW", IsStr(regex=r"RATE_LIMIT_WAIT:[0-9.]+s")],
            ),
        ]

    async def test_a_wait_longer_than_max_wait_seconds_denies_instead_of_stalling(self):
        def search(query: str = "") -> str:
            return "results"

        # Room frees only in an hour, but the policy will not hold a call past 50ms.
        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(1, window_seconds=3600, max_wait_seconds=0.05)],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(
                ToolCallEvent(name="search", arguments="{}"),
                ToolCallEvent(name="search", arguments="{}"),
                "Done.",
                raise_tool_errors=False,
            ),
            tools=[search],
            middleware=[governance],
        )

        started = time.monotonic()
        await agent.ask("search twice")

        # Denied promptly rather than held for the hour it would have taken.
        assert time.monotonic() - started < 1.0
        assert governance.decisions == [
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="DENY", reason_codes=["RATE_LIMIT_EXCEEDED"]),
        ]

    async def test_parallel_calls_in_one_turn_do_not_overshoot_the_limit(self):
        """The window check and the slot it reserves have to be atomic.

        A turn's tool calls are executed concurrently, so without an atomic
        reservation every waiter wakes, sees room, and records — letting more
        calls through than the limit allows.
        """
        ran: list[str] = []

        def search(query: str = "") -> str:
            ran.append(query)
            return "results"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(2, window_seconds=60, on_exceeded="deny")],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(
                [ToolCallEvent(name="search", arguments="{}") for _ in range(5)],
                "Done.",
                raise_tool_errors=False,
            ),
            tools=[search],
            middleware=[governance],
        )

        await agent.ask("search five times at once")

        assert governance.decisions == IsList(
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="DENY"),
            IsPartialDataclass(action="DENY"),
            IsPartialDataclass(action="DENY"),
            check_order=False,
        )
        # Exactly two calls reached the tool; the denied three never ran.
        assert len(ran) == 2


@pytest.mark.asyncio
class TestRateLimitRespectsMode:
    async def test_observe_never_denies(self):
        def search(query: str = "") -> str:
            return "results"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(1, window_seconds=60, on_exceeded="deny")],
            mode=GovernanceMode.OBSERVE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(
                ToolCallEvent(name="search", arguments="{}"),
                ToolCallEvent(name="search", arguments="{}"),
                "Done.",
                raise_tool_errors=False,
            ),
            tools=[search],
            middleware=[governance],
        )

        reply = await agent.ask("search twice")

        # OBSERVE short-circuits before policy evaluation: both calls just run.
        assert reply.body == "Done."
        assert governance.decisions == [
            IsPartialDataclass(action="ALLOW", reason_codes=["OBSERVE_PASSTHROUGH"]),
            IsPartialDataclass(action="ALLOW", reason_codes=["OBSERVE_PASSTHROUGH"]),
        ]

    async def test_monitor_records_the_denial_but_runs_the_call_anyway(self):
        ran: list[str] = []

        def search(query: str = "") -> str:
            ran.append(query)
            return "results"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(1, window_seconds=60)],
            mode=GovernanceMode.MONITOR,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(
                ToolCallEvent(name="search", arguments="{}"),
                ToolCallEvent(name="search", arguments="{}"),
                "Done.",
                raise_tool_errors=False,
            ),
            tools=[search],
            middleware=[governance],
        )

        await agent.ask("search twice")

        # The violation is recorded, but only ENFORCE blocks — and MONITOR never
        # waits either, since waiting is itself a form of blocking.
        assert governance.decisions == [
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="DENY", reason_codes=["RATE_LIMIT_EXCEEDED"]),
        ]
        assert len(ran) == 2

    async def test_monitor_counts_the_call_it_did_not_block(self):
        """A MONITOR call that was over the limit still ran, so the window counts it.

        The timings isolate that: with a 0.3s window and a 0.2s tool, by the third
        call the first has aged out but the second has not. The third is therefore
        denied only if the second — denied yet executed — was counted.
        """

        async def slow(query: str = "") -> str:
            await asyncio.sleep(0.2)
            return "results"

        governance = TealTigerMiddleware(
            policies=[GovernancePolicy.rate_limit(1, window_seconds=0.3)],
            mode=GovernanceMode.MONITOR,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(
                ToolCallEvent(name="slow", arguments="{}"),
                ToolCallEvent(name="slow", arguments="{}"),
                ToolCallEvent(name="slow", arguments="{}"),
                "Done.",
                raise_tool_errors=False,
            ),
            tools=[slow],
            middleware=[governance],
        )

        await agent.ask("call slowly three times")

        assert governance.decisions == [
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="DENY"),
            IsPartialDataclass(action="DENY"),
        ]


@pytest.mark.asyncio
async def test_enforce_surfaces_the_governance_error_to_the_caller():
    def search(query: str = "") -> str:
        return "results"

    governance = TealTigerMiddleware(
        policies=[GovernancePolicy.rate_limit(1, window_seconds=60, tool="search", on_exceeded="deny")],
        mode=GovernanceMode.ENFORCE,
    )
    agent = Agent(
        "assistant",
        config=TestConfig(
            ToolCallEvent(name="search", arguments="{}"),
            ToolCallEvent(name="search", arguments="{}"),
            "Done.",
        ),
        tools=[search],
        middleware=[governance],
    )

    with pytest.raises(Exception, match=r"\[GOVERNANCE DENIED\].*RATE_LIMIT_EXCEEDED"):
        await agent.ask("search twice")

    assert governance.deny_count == 1


@pytest.mark.asyncio
class TestSharedScopeCountsPerPolicy:
    """Regression for PR #3246 review: policies sharing a tool scope must each
    count a real call exactly once against their own window, so the tighter limit
    trips at its real threshold rather than at half of it from double counting.
    """

    async def test_two_global_limits_do_not_double_count_one_call(self):
        def search(query: str = "") -> str:
            return "results"

        governance = TealTigerMiddleware(
            policies=[
                GovernancePolicy.rate_limit(4, window_seconds=60, on_exceeded="deny"),
                GovernancePolicy.rate_limit(100, window_seconds=3600, on_exceeded="deny"),
            ],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(
                *[ToolCallEvent(name="search", arguments="{}") for _ in range(5)],
                "Done.",
                raise_tool_errors=False,
            ),
            tools=[search],
            middleware=[governance],
        )

        await agent.ask("search five times")

        # Were the shared scope counted twice per call, the 4-call limit would
        # have tripped on the third call instead of the fifth.
        assert governance.decisions == [
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="DENY"),
        ]

    async def test_two_policies_with_the_same_pattern_do_not_double_count(self):
        def db_read(key: str = "") -> str:
            return "row"

        governance = TealTigerMiddleware(
            policies=[
                GovernancePolicy.rate_limit(3, window_seconds=60, tool="db_*", on_exceeded="deny"),
                GovernancePolicy.rate_limit(50, window_seconds=3600, tool="db_*", on_exceeded="deny"),
            ],
            mode=GovernanceMode.ENFORCE,
        )
        agent = Agent(
            "assistant",
            config=TestConfig(
                *[ToolCallEvent(name="db_read", arguments="{}") for _ in range(4)],
                "Done.",
                raise_tool_errors=False,
            ),
            tools=[db_read],
            middleware=[governance],
        )

        await agent.ask("read the db four times")

        assert governance.decisions == [
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="DENY"),
        ]


@pytest.mark.asyncio
class TestMixedWindowSharedScope:
    """Regression for PR #3246 review (discussion r3992565377): two rate_limit
    policies that share a tool scope but use different windows must not prune each
    other's history. A short-window policy must never drop timestamps the longer
    window still needs, and the outcome must not depend on declaration order.

    Both cases pace the calls past the short window so it never binds; the long
    limit then denies the fourth call only if its own history survived intact.
    """

    async def test_the_long_window_still_counts_calls_the_short_one_forgot(self):
        governance = TealTigerMiddleware(
            policies=[
                GovernancePolicy.rate_limit(2, window_seconds=0.05, on_exceeded="deny"),
                GovernancePolicy.rate_limit(3, window_seconds=60, on_exceeded="deny"),
            ],
            mode=GovernanceMode.ENFORCE,
        )

        await self._run(governance)

        assert governance.decisions == [
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="DENY", reason_codes=["RATE_LIMIT_EXCEEDED"]),
        ]

    async def test_the_outcome_does_not_depend_on_policy_order(self):
        governance = TealTigerMiddleware(
            policies=[
                GovernancePolicy.rate_limit(3, window_seconds=60, on_exceeded="deny"),
                GovernancePolicy.rate_limit(2, window_seconds=0.05, on_exceeded="deny"),
            ],
            mode=GovernanceMode.ENFORCE,
        )

        await self._run(governance)

        assert governance.decisions == [
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="ALLOW"),
            IsPartialDataclass(action="DENY", reason_codes=["RATE_LIMIT_EXCEEDED"]),
        ]

    @staticmethod
    async def _run(governance: TealTigerMiddleware) -> None:
        async def slow(query: str = "") -> str:
            # Longer than the short window, so that limit never binds.
            await asyncio.sleep(0.08)
            return "results"

        agent = Agent(
            "assistant",
            config=TestConfig(
                *[ToolCallEvent(name="slow", arguments="{}") for _ in range(4)],
                "Done.",
                raise_tool_errors=False,
            ),
            tools=[slow],
            middleware=[governance],
        )
        await agent.ask("call slowly four times")
