# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for governance middleware."""

from __future__ import annotations

import math
import threading
from unittest import mock

import pytest

from autogen.beta.events import BaseEvent, ModelResponse, ToolCall, ToolError
from autogen.beta.middleware.builtin.governance import (
    CircuitState,
    GovernanceConfig,
    GovernanceMiddleware,
)


def _make_event():
    return mock.MagicMock(spec=BaseEvent)


def _make_context():
    return mock.MagicMock()


def _make_model_response(
    prompt_tokens=100,
    output_tokens=50,
    *,
    key_style="prompt",
):
    """Create ModelResponse with configurable token key names."""
    if key_style == "prompt":
        usage = {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens}
    elif key_style == "completion":
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": output_tokens}
    elif key_style == "input":
        usage = {"input_tokens": prompt_tokens, "output_tokens": output_tokens}
    else:
        usage = {}
    return ModelResponse(usage=usage)


def _make_tool_call(name="search", call_id="call_001"):
    tc = mock.MagicMock(spec=ToolCall)
    tc.name = name
    tc.id = call_id
    return tc


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:

    @pytest.mark.asyncio
    async def test_allows_within_budget(self):
        mw = GovernanceMiddleware(GovernanceConfig(max_cost_usd=10.0))
        inst = mw(_make_event(), _make_context())
        resp = _make_model_response()

        async def ok(events, ctx):
            return resp

        result = await inst.on_llm_call(ok, [_make_event()], _make_context())
        assert result is resp

    @pytest.mark.asyncio
    async def test_blocks_when_exhausted(self):
        mw = GovernanceMiddleware(GovernanceConfig(max_cost_usd=0.001))
        mw._budget._spent = 0.002
        inst = mw(_make_event(), _make_context())

        async def fail(events, ctx):
            raise AssertionError("should not be called")

        result = await inst.on_llm_call(fail, [_make_event()], _make_context())
        assert result.usage == {}

    @pytest.mark.asyncio
    async def test_records_exact_cost(self):
        """Cost must match: 1000/1000*0.00025 + 500/1000*0.0005 = 0.0005."""
        mw = GovernanceMiddleware(GovernanceConfig(max_cost_usd=10.0))
        inst = mw(_make_event(), _make_context())
        resp = _make_model_response(prompt_tokens=1000, output_tokens=500)

        async def ok(events, ctx):
            return resp

        await inst.on_llm_call(ok, [_make_event()], _make_context())
        assert abs(mw.budget.spent - 0.0005) < 1e-9

    @pytest.mark.asyncio
    async def test_completion_tokens_key(self):
        """OpenAI-style completion_tokens must be recognized."""
        mw = GovernanceMiddleware(GovernanceConfig(max_cost_usd=10.0))
        inst = mw(_make_event(), _make_context())
        resp = _make_model_response(prompt_tokens=0, output_tokens=1000, key_style="completion")

        async def ok(events, ctx):
            return resp

        await inst.on_llm_call(ok, [_make_event()], _make_context())
        # 0 input + 1000/1000*0.0005 = 0.0005
        assert abs(mw.budget.spent - 0.0005) < 1e-9

    @pytest.mark.asyncio
    async def test_zero_prompt_tokens_not_masked(self):
        """prompt_tokens=0 must not fall through to input_tokens key."""
        mw = GovernanceMiddleware(GovernanceConfig(max_cost_usd=10.0))
        inst = mw(_make_event(), _make_context())
        # prompt_tokens=0 is present AND input_tokens=9999 -- must use 0
        resp = ModelResponse(usage={"prompt_tokens": 0, "input_tokens": 9999, "output_tokens": 0})

        async def ok(events, ctx):
            return resp

        await inst.on_llm_call(ok, [_make_event()], _make_context())
        assert mw.budget.spent == 0.0  # not 9999 * rate

    @pytest.mark.asyncio
    async def test_input_tokens_key_gemini_style(self):
        """Gemini-style input_tokens/output_tokens must be recognized."""
        mw = GovernanceMiddleware(GovernanceConfig(max_cost_usd=10.0))
        inst = mw(_make_event(), _make_context())
        resp = _make_model_response(prompt_tokens=1000, output_tokens=500, key_style="input")

        async def ok(events, ctx):
            return resp

        await inst.on_llm_call(ok, [_make_event()], _make_context())
        assert abs(mw.budget.spent - 0.0005) < 1e-9

    @pytest.mark.asyncio
    async def test_empty_usage_no_cost(self):
        mw = GovernanceMiddleware(GovernanceConfig(max_cost_usd=10.0))
        inst = mw(_make_event(), _make_context())
        resp = ModelResponse(usage={})

        async def ok(events, ctx):
            return resp

        await inst.on_llm_call(ok, [_make_event()], _make_context())
        assert mw.budget.spent == 0.0

    @pytest.mark.asyncio
    async def test_nan_tokens_ignored(self):
        """NaN token values must not corrupt budget."""
        mw = GovernanceMiddleware(GovernanceConfig(max_cost_usd=10.0))
        inst = mw(_make_event(), _make_context())
        resp = ModelResponse(usage={"prompt_tokens": float("nan"), "output_tokens": float("inf")})

        async def ok(events, ctx):
            return resp

        await inst.on_llm_call(ok, [_make_event()], _make_context())
        assert math.isfinite(mw.budget.spent)
        assert mw.budget.spent == 0.0

    @pytest.mark.asyncio
    async def test_none_tokens_no_crash(self):
        """None token values must not raise TypeError."""
        mw = GovernanceMiddleware(GovernanceConfig(max_cost_usd=10.0))
        inst = mw(_make_event(), _make_context())
        resp = ModelResponse(usage={"prompt_tokens": None, "completion_tokens": None})

        async def ok(events, ctx):
            return resp

        await inst.on_llm_call(ok, [_make_event()], _make_context())
        assert mw.budget.spent == 0.0

    def test_zero_budget_means_unlimited(self):
        """max_cost_usd=0 disables budget enforcement."""
        mw = GovernanceMiddleware(GovernanceConfig(max_cost_usd=0.0))
        mw._budget._spent = 99999.0
        allowed, _ = mw._budget.check()
        assert allowed is True


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:

    @pytest.mark.asyncio
    async def test_trips_after_threshold(self):
        mw = GovernanceMiddleware(GovernanceConfig(failure_threshold=2))

        async def fail(events, ctx):
            raise RuntimeError("model error")

        for _ in range(2):
            inst = mw(_make_event(), _make_context())
            with pytest.raises(RuntimeError):
                await inst.on_llm_call(fail, [_make_event()], _make_context())

        assert mw.circuit_breaker.get_state() == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_blocks_call(self):
        mw = GovernanceMiddleware(GovernanceConfig(failure_threshold=1))

        async def fail(events, ctx):
            raise RuntimeError("fail")

        inst = mw(_make_event(), _make_context())
        with pytest.raises(RuntimeError):
            await inst.on_llm_call(fail, [_make_event()], _make_context())

        # Next call blocked
        async def should_not_run(events, ctx):
            raise AssertionError("should not be called")

        inst2 = mw(_make_event(), _make_context())
        result = await inst2.on_llm_call(should_not_run, [_make_event()], _make_context())
        assert result.usage == {}

    @pytest.mark.asyncio
    async def test_success_resets(self):
        mw = GovernanceMiddleware(GovernanceConfig(failure_threshold=3))

        async def fail(events, ctx):
            raise RuntimeError("fail")

        for _ in range(2):
            inst = mw(_make_event(), _make_context())
            with pytest.raises(RuntimeError):
                await inst.on_llm_call(fail, [_make_event()], _make_context())

        async def ok(events, ctx):
            return _make_model_response()

        inst = mw(_make_event(), _make_context())
        await inst.on_llm_call(ok, [_make_event()], _make_context())
        assert mw.circuit_breaker.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_probe_allowed(self):
        """After recovery timeout, one probe should pass through."""
        mw = GovernanceMiddleware(GovernanceConfig(failure_threshold=1, recovery_timeout_s=0.0))

        # Trip the breaker
        async def fail(events, ctx):
            raise RuntimeError("fail")

        inst = mw(_make_event(), _make_context())
        with pytest.raises(RuntimeError):
            await inst.on_llm_call(fail, [_make_event()], _make_context())

        # recovery_timeout_s=0 -> immediately HALF_OPEN
        async def ok(events, ctx):
            return _make_model_response()

        inst2 = mw(_make_event(), _make_context())
        result = await inst2.on_llm_call(ok, [_make_event()], _make_context())
        assert result.usage != {}  # not blocked
        assert mw.circuit_breaker.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_second_probe_blocked(self):
        """While probe is in flight, second call must be blocked."""
        mw = GovernanceMiddleware(GovernanceConfig(failure_threshold=1, recovery_timeout_s=0.0))

        async def fail(events, ctx):
            raise RuntimeError("fail")

        inst = mw(_make_event(), _make_context())
        with pytest.raises(RuntimeError):
            await inst.on_llm_call(fail, [_make_event()], _make_context())

        # Claim probe via check_and_claim directly
        state, claimed = mw.circuit_breaker.check_and_claim()
        assert state == CircuitState.HALF_OPEN
        assert claimed is True

        # Second call should be blocked (probe in flight)
        async def should_not_run(events, ctx):
            raise AssertionError("should not be called")

        inst2 = mw(_make_event(), _make_context())
        result = await inst2.on_llm_call(should_not_run, [_make_event()], _make_context())
        assert result.usage == {}

        mw.circuit_breaker.release_probe()

    @pytest.mark.asyncio
    async def test_budget_blocked_during_probe_releases_probe(self):
        """If probe is claimed but budget blocks, probe must be released."""
        mw = GovernanceMiddleware(GovernanceConfig(
            failure_threshold=1,
            recovery_timeout_s=0.0,
            max_cost_usd=0.001,
        ))

        async def fail(events, ctx):
            raise RuntimeError("fail")

        inst = mw(_make_event(), _make_context())
        with pytest.raises(RuntimeError):
            await inst.on_llm_call(fail, [_make_event()], _make_context())

        # Exhaust budget
        mw._budget._spent = 0.002

        # Probe claimed -> budget blocked -> probe released
        inst2 = mw(_make_event(), _make_context())
        result = await inst2.on_llm_call(fail, [_make_event()], _make_context())
        assert result.usage == {}

        # Probe should be released, not permanently stuck
        assert mw.circuit_breaker._probe_in_flight is False

    @pytest.mark.asyncio
    async def test_tool_error_does_not_affect_circuit_breaker(self):
        """Tool failures should not trip the LLM circuit breaker."""
        mw = GovernanceMiddleware(GovernanceConfig(failure_threshold=1))
        tc = _make_tool_call("search")

        async def fail(event, ctx):
            raise RuntimeError("tool fail")

        inst = mw(_make_event(), _make_context())
        with pytest.raises(RuntimeError):
            await inst.on_tool_execution(fail, tc, _make_context())

        # CB should still be CLOSED -- tool errors don't participate
        assert mw.circuit_breaker.get_state() == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Tool Policy
# ---------------------------------------------------------------------------


class TestToolPolicy:

    @pytest.mark.asyncio
    async def test_blocks_disallowed(self):
        mw = GovernanceMiddleware(GovernanceConfig(blocked_tools=["dangerous"]))
        inst = mw(_make_event(), _make_context())
        tc = _make_tool_call(name="dangerous")

        async def fail(event, ctx):
            raise AssertionError("should not be called")

        result = await inst.on_tool_execution(fail, tc, _make_context())
        assert isinstance(result, ToolError)
        assert "blocked" in str(result.error)
        assert result.content == str(result.error)

    @pytest.mark.asyncio
    async def test_allows_normal(self):
        mw = GovernanceMiddleware(GovernanceConfig(blocked_tools=["dangerous"]))
        inst = mw(_make_event(), _make_context())
        tc = _make_tool_call(name="safe")
        expected = mock.MagicMock()

        async def ok(event, ctx):
            return expected

        result = await inst.on_tool_execution(ok, tc, _make_context())
        assert result is expected

    @pytest.mark.asyncio
    async def test_allowlist_blocks_unlisted(self):
        mw = GovernanceMiddleware(GovernanceConfig(allowed_tools=["search", "calc"]))
        inst = mw(_make_event(), _make_context())
        tc = _make_tool_call(name="delete_all")

        async def fail(event, ctx):
            raise AssertionError("should not be called")

        result = await inst.on_tool_execution(fail, tc, _make_context())
        assert isinstance(result, ToolError)
        assert "not in the allowed list" in str(result.error)

    @pytest.mark.asyncio
    async def test_allowlist_permits_listed(self):
        mw = GovernanceMiddleware(GovernanceConfig(allowed_tools=["search"]))
        inst = mw(_make_event(), _make_context())
        tc = _make_tool_call(name="search")
        expected = mock.MagicMock()

        async def ok(event, ctx):
            return expected

        result = await inst.on_tool_execution(ok, tc, _make_context())
        assert result is expected


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------


class TestDegradation:

    @pytest.mark.asyncio
    async def test_logged_once(self, caplog):
        import logging

        config = GovernanceConfig(
            max_cost_usd=1.0,
            degradation_threshold=0.5,
            fallback_model="gpt-4o-mini",
        )
        mw = GovernanceMiddleware(config)
        mw._budget._spent = 0.6

        async def ok(events, ctx):
            return _make_model_response(prompt_tokens=10, output_tokens=5)

        with caplog.at_level(logging.INFO):
            inst1 = mw(_make_event(), _make_context())
            await inst1.on_llm_call(ok, [_make_event()], _make_context())
            inst2 = mw(_make_event(), _make_context())
            await inst2.on_llm_call(ok, [_make_event()], _make_context())

        degrade_lines = [r for r in caplog.records if "degraded" in r.message.lower()]
        assert len(degrade_lines) == 1

    @pytest.mark.asyncio
    async def test_disables_expensive_tool(self):
        config = GovernanceConfig(
            max_cost_usd=1.0,
            degradation_threshold=0.5,
            disable_tools_on_degrade=["web_search"],
        )
        mw = GovernanceMiddleware(config)
        mw._budget._spent = 0.6
        inst = mw(_make_event(), _make_context())
        tc = _make_tool_call(name="web_search")

        async def fail(event, ctx):
            raise AssertionError("should not be called")

        result = await inst.on_tool_execution(fail, tc, _make_context())
        assert isinstance(result, ToolError)
        assert "disabled" in str(result.error)

    @pytest.mark.asyncio
    async def test_boundary_at_threshold(self):
        """Exactly at threshold should trigger degradation."""
        config = GovernanceConfig(
            max_cost_usd=1.0,
            degradation_threshold=0.5,
            fallback_model="gpt-4o-mini",
        )
        mw = GovernanceMiddleware(config)
        mw._budget._spent = 0.5  # exactly 50%

        async def ok(events, ctx):
            return _make_model_response(prompt_tokens=0, output_tokens=0)

        inst = mw(_make_event(), _make_context())
        await inst.on_llm_call(ok, [_make_event()], _make_context())
        assert mw._degraded is True


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:

    def test_concurrent_budget_record(self):
        mw = GovernanceMiddleware(GovernanceConfig(max_cost_usd=100000.0))
        errors: list[str] = []

        def spend():
            try:
                for _ in range(100):
                    mw._budget.record({"prompt_tokens": 100, "output_tokens": 50})
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=spend) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # 10 threads * 100 calls * (100/1000*0.00025 + 50/1000*0.0005)
        expected = 10 * 100 * 0.00005
        assert abs(mw.budget.spent - expected) < 1e-6

    def test_concurrent_claim_probe_exactly_one(self):
        cb = GovernanceMiddleware(GovernanceConfig(failure_threshold=1, recovery_timeout_s=0.0))._cb
        cb.record_failure()  # trip to OPEN -> immediately HALF_OPEN (timeout=0)

        results: list[bool] = []
        lock = threading.Lock()

        def try_claim():
            _, claimed = cb.check_and_claim()
            with lock:
                results.append(claimed)

        threads = [threading.Thread(target=try_claim) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results.count(True) == 1
        assert results.count(False) == 9


# ---------------------------------------------------------------------------
# Adversarial
# ---------------------------------------------------------------------------


class TestAdversarial:

    @pytest.mark.asyncio
    async def test_negative_tokens_clamped_to_zero(self):
        """Negative token values must not reduce budget spent."""
        mw = GovernanceMiddleware(GovernanceConfig(max_cost_usd=10.0))
        inst = mw(_make_event(), _make_context())
        resp = ModelResponse(usage={"prompt_tokens": -1000, "output_tokens": -500})

        async def ok(events, ctx):
            return resp

        await inst.on_llm_call(ok, [_make_event()], _make_context())
        assert mw.budget.spent == 0.0

    @pytest.mark.asyncio
    async def test_budget_boundary_triple(self):
        """limit-1, limit, limit+1 on budget check."""
        config = GovernanceConfig(max_cost_usd=1.0)

        # Just below limit -- allowed
        mw1 = GovernanceMiddleware(config)
        mw1._budget._spent = 0.999
        allowed, _ = mw1._budget.check()
        assert allowed is True

        # At limit -- blocked (spent >= limit)
        mw2 = GovernanceMiddleware(config)
        mw2._budget._spent = 1.0
        allowed, _ = mw2._budget.check()
        assert allowed is False

        # Above limit -- blocked
        mw3 = GovernanceMiddleware(config)
        mw3._budget._spent = 1.001
        allowed, _ = mw3._budget.check()
        assert allowed is False

    @pytest.mark.asyncio
    async def test_empty_allowlist_blocks_all_tools(self):
        """allowed_tools=[] means deny all, not no restriction."""
        mw = GovernanceMiddleware(GovernanceConfig(allowed_tools=[]))
        inst = mw(_make_event(), _make_context())
        tc = _make_tool_call(name="any_tool")

        async def fail(event, ctx):
            raise AssertionError("should not be called")

        result = await inst.on_tool_execution(fail, tc, _make_context())
        assert isinstance(result, ToolError)

    @pytest.mark.asyncio
    async def test_blocklist_overrides_allowlist(self):
        """Tool in both blocked and allowed -> blocked wins."""
        mw = GovernanceMiddleware(GovernanceConfig(
            blocked_tools=["search"],
            allowed_tools=["search", "calc"],
        ))
        inst = mw(_make_event(), _make_context())
        tc = _make_tool_call(name="search")

        async def fail(event, ctx):
            raise AssertionError("should not be called")

        result = await inst.on_tool_execution(fail, tc, _make_context())
        assert isinstance(result, ToolError)
        assert "blocked" in str(result.error)

    @pytest.mark.asyncio
    async def test_cumulative_cost_across_calls(self):
        """Multiple LLM calls accumulate cost correctly."""
        mw = GovernanceMiddleware(GovernanceConfig(max_cost_usd=10.0))

        async def ok(events, ctx):
            return _make_model_response(prompt_tokens=1000, output_tokens=0)

        for _ in range(10):
            inst = mw(_make_event(), _make_context())
            await inst.on_llm_call(ok, [_make_event()], _make_context())

        # 10 * (1000/1000 * 0.00025) = 0.0025
        assert abs(mw.budget.spent - 0.0025) < 1e-9

    @pytest.mark.asyncio
    async def test_circuit_open_log_on_trip(self, caplog):
        """Circuit trip to OPEN must log a warning."""
        import logging

        mw = GovernanceMiddleware(GovernanceConfig(failure_threshold=1))

        async def fail(events, ctx):
            raise RuntimeError("boom")

        with caplog.at_level(logging.WARNING):
            inst = mw(_make_event(), _make_context())
            with pytest.raises(RuntimeError):
                await inst.on_llm_call(fail, [_make_event()], _make_context())

        assert any("tripped to OPEN" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_tool_error_content_matches_reason(self):
        """ToolError.content must be the reason string, not format_exc()."""
        mw = GovernanceMiddleware(GovernanceConfig(blocked_tools=["bad"]))
        inst = mw(_make_event(), _make_context())
        tc = _make_tool_call(name="bad")

        async def fail(event, ctx):
            raise AssertionError("should not be called")

        result = await inst.on_tool_execution(fail, tc, _make_context())
        assert isinstance(result, ToolError)
        # content should be the policy reason, not "NoneType: None"
        assert "bad" in result.content
        assert "NoneType" not in result.content
