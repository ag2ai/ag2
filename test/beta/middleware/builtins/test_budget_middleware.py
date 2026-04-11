# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for BudgetMiddleware."""

import asyncio
import math
import threading
from collections.abc import Sequence
from unittest import mock

import pytest

from autogen.beta.context import Context
from autogen.beta.events import BaseEvent, ModelMessage, ModelResponse
from autogen.beta.events.types import Usage
from autogen.beta.middleware.builtin.budget import (
    _SPENT_KEY,
    BudgetConfig,
    BudgetExceededError,
    BudgetMiddleware,
    _BudgetTracker,
)
from autogen.beta.stream import MemoryStream

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event() -> mock.MagicMock:
    return mock.MagicMock(spec=BaseEvent)


def _make_context() -> Context:
    return Context(stream=MemoryStream())


def _make_model_response(
    prompt_tokens: float = 100,
    output_tokens: float = 50,
    *,
    key_style: str = "prompt",
) -> ModelResponse:
    """Build a ModelResponse with the requested token key style.

    key_style options:
      "prompt"       -- OpenAI: prompt_tokens / completion_tokens
      "prompt_count" -- Gemini: prompt_token_count / candidates_token_count
      "input"        -- Generic: input_tokens / output_tokens
    """
    if key_style == "prompt":
        usage: dict[str, float] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
        }
    elif key_style == "prompt_count":
        usage = {
            "prompt_token_count": prompt_tokens,
            "candidates_token_count": output_tokens,
        }
    elif key_style == "input":
        usage = {
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
        }
    else:
        usage = {}
    return ModelResponse(message=ModelMessage(content="ok"), usage=usage)


async def _llm_call_returning(
    response: ModelResponse,
) -> mock.AsyncMock:
    """Return an async callable that yields response."""

    async def _call(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
        return response

    return _call  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# TestBudgetEnforcement
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    @pytest.mark.asyncio()
    async def test_allows_within_budget(self) -> None:
        # Given: budget of $1, no prior spending
        mw = BudgetMiddleware(
            max_cost_usd=1.0,
            cost_per_1k_input_tokens=0.30,
            cost_per_1k_output_tokens=0.60,
        )
        instance = mw(_make_event(), _make_context())
        response = _make_model_response(100, 50)

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return response

        # When: budget is not exhausted
        result = await instance.on_llm_call(call_next, [], _make_context())

        # Then: call goes through and returns the response
        assert result.message is not None
        assert result.message.content == "ok"

    @pytest.mark.asyncio()
    async def test_raises_when_exhausted(self) -> None:
        # Given: an instance whose tracker is exhausted above the limit
        mw = BudgetMiddleware(max_cost_usd=0.001)
        instance = mw(_make_event(), _make_context())
        instance._budget._spent = 0.002  # manually exhaust

        called = False

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            nonlocal called
            called = True
            return _make_model_response()

        # When: budget is exhausted
        # Then: BudgetExceededError is raised and downstream is not invoked
        with pytest.raises(BudgetExceededError) as excinfo:
            await instance.on_llm_call(call_next, [], _make_context())

        assert not called
        assert excinfo.value.spent == 0.002
        assert excinfo.value.limit == 0.001

    @pytest.mark.asyncio()
    async def test_syncs_context_spend_before_check(self) -> None:
        mw = BudgetMiddleware(max_cost_usd=1.0)
        ctx = _make_context()
        instance = mw(_make_event(), ctx)
        ctx.variables[_SPENT_KEY] = 1.0
        called = False

        async def call_next(events: Sequence[BaseEvent], current_ctx: object) -> ModelResponse:
            nonlocal called
            called = True
            return _make_model_response()

        with pytest.raises(BudgetExceededError) as excinfo:
            await instance.on_llm_call(call_next, [], ctx)

        assert not called
        assert excinfo.value.spent == pytest.approx(1.0)
        assert instance._budget.spent == pytest.approx(1.0)

    @pytest.mark.asyncio()
    async def test_records_exact_cost(self) -> None:
        # Given: $0.30/1k input, $0.60/1k output
        mw = BudgetMiddleware(
            max_cost_usd=10.0,
            cost_per_1k_input_tokens=0.30,
            cost_per_1k_output_tokens=0.60,
        )
        instance = mw(_make_event(), _make_context())
        # 1000 prompt + 500 output = 1000/1000*0.30 + 500/1000*0.60 = 0.30 + 0.30 = 0.60 USD
        response = _make_model_response(1000, 500, key_style="prompt")

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return response

        # When: call completes
        await instance.on_llm_call(call_next, [], _make_context())

        # Then: cost is 0.60 on this conversation's tracker
        assert abs(instance._budget.spent - 0.60) < 1e-9

    @pytest.mark.asyncio()
    async def test_completion_tokens_key(self) -> None:
        # Given: OpenAI-style keys with completion_tokens
        mw = BudgetMiddleware(
            max_cost_usd=10.0,
            cost_per_1k_input_tokens=0.0,
            cost_per_1k_output_tokens=1.0,
        )
        instance = mw(_make_event(), _make_context())
        response = ModelResponse(
            message=ModelMessage(content="ok"),
            usage={"prompt_tokens": 0, "completion_tokens": 2000},
        )

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return response

        # When: call completes with completion_tokens
        await instance.on_llm_call(call_next, [], _make_context())

        # Then: 2000/1000 * $1.0 = $2.0
        assert abs(instance._budget.spent - 2.0) < 1e-9

    @pytest.mark.asyncio()
    async def test_zero_prompt_tokens_not_masked(self) -> None:
        # Given: prompt_tokens=0 is present; input_tokens=9999 is also present
        # The `in` check must use prompt_tokens (not fall through to input_tokens)
        mw = BudgetMiddleware(
            max_cost_usd=10.0,
            cost_per_1k_input_tokens=1.0,
            cost_per_1k_output_tokens=0.0,
        )
        instance = mw(_make_event(), _make_context())
        response = ModelResponse(
            message=ModelMessage(content="ok"),
            usage={"prompt_tokens": 0, "input_tokens": 9999, "completion_tokens": 0},
        )

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return response

        # When: call completes
        await instance.on_llm_call(call_next, [], _make_context())

        # Then: cost uses prompt_tokens=0, not input_tokens=9999
        assert instance._budget.spent == 0.0

    @pytest.mark.asyncio()
    async def test_input_tokens_key_gemini_style(self) -> None:
        # Given: generic input_tokens / output_tokens keys
        mw = BudgetMiddleware(
            max_cost_usd=10.0,
            cost_per_1k_input_tokens=1.0,
            cost_per_1k_output_tokens=0.0,
        )
        instance = mw(_make_event(), _make_context())
        response = _make_model_response(2000, 0, key_style="input")

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return response

        # When: call uses input_tokens key
        await instance.on_llm_call(call_next, [], _make_context())

        # Then: 2000/1000 * $1.0 = $2.0
        assert abs(instance._budget.spent - 2.0) < 1e-9

    @pytest.mark.asyncio()
    async def test_empty_usage_no_cost(self) -> None:
        # Given: response with no usage dict entries
        mw = BudgetMiddleware(max_cost_usd=10.0, cost_per_1k_input_tokens=1.0)
        instance = mw(_make_event(), _make_context())
        response = ModelResponse(message=ModelMessage(content="ok"), usage={})

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return response

        # When: call has empty usage
        await instance.on_llm_call(call_next, [], _make_context())

        # Then: no cost accumulated
        assert instance._budget.spent == 0.0

    @pytest.mark.asyncio()
    async def test_gemini_token_keys(self) -> None:
        # Given: Gemini-style prompt_token_count / candidates_token_count
        mw = BudgetMiddleware(
            max_cost_usd=10.0,
            cost_per_1k_input_tokens=0.30,
            cost_per_1k_output_tokens=0.60,
        )
        instance = mw(_make_event(), _make_context())
        response = _make_model_response(1000, 1000, key_style="prompt_count")

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return response

        # When: Gemini keys are used
        await instance.on_llm_call(call_next, [], _make_context())

        # Then: 1000/1000 * 0.30 + 1000/1000 * 0.60 = 0.90
        assert abs(instance._budget.spent - 0.90) < 1e-9

    @pytest.mark.asyncio()
    async def test_string_token_values_coerced(self) -> None:
        # Given: token values as numeric strings (some APIs return strings)
        mw = BudgetMiddleware(
            max_cost_usd=10.0,
            cost_per_1k_input_tokens=1.0,
            cost_per_1k_output_tokens=0.0,
        )
        instance = mw(_make_event(), _make_context())
        response = ModelResponse(
            message=ModelMessage(content="ok"),
            usage={"prompt_tokens": "500", "completion_tokens": "0"},  # type: ignore[dict-item]
        )

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return response

        # When: string tokens are provided
        await instance.on_llm_call(call_next, [], _make_context())

        # Then: 500/1000 * $1.0 = $0.50
        assert abs(instance._budget.spent - 0.50) < 1e-9

    @pytest.mark.asyncio()
    async def test_non_numeric_string_tokens_ignored(self) -> None:
        # Given: non-numeric token value
        mw = BudgetMiddleware(max_cost_usd=10.0, cost_per_1k_input_tokens=1.0)
        instance = mw(_make_event(), _make_context())
        response = ModelResponse(
            message=ModelMessage(content="ok"),
            usage={"prompt_tokens": "not_a_number", "completion_tokens": 0},  # type: ignore[dict-item]
        )

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return response

        # When: non-numeric token value
        await instance.on_llm_call(call_next, [], _make_context())

        # Then: treated as zero, no cost
        assert instance._budget.spent == 0.0

    @pytest.mark.asyncio()
    async def test_nan_tokens_ignored(self) -> None:
        # Given: NaN token value
        mw = BudgetMiddleware(max_cost_usd=10.0, cost_per_1k_input_tokens=1.0)
        instance = mw(_make_event(), _make_context())
        response = ModelResponse(
            message=ModelMessage(content="ok"),
            usage={"prompt_tokens": math.nan, "completion_tokens": 0},
        )

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return response

        # When: NaN tokens
        await instance.on_llm_call(call_next, [], _make_context())

        # Then: treated as zero
        assert instance._budget.spent == 0.0

    @pytest.mark.asyncio()
    async def test_none_tokens_no_crash(self) -> None:
        # Given: None token value
        mw = BudgetMiddleware(max_cost_usd=10.0, cost_per_1k_input_tokens=1.0)
        instance = mw(_make_event(), _make_context())
        response = ModelResponse(
            message=ModelMessage(content="ok"),
            usage={"prompt_tokens": None, "completion_tokens": 0},  # type: ignore[dict-item]
        )

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return response

        # When: None token value
        await instance.on_llm_call(call_next, [], _make_context())

        # Then: no crash, treated as zero
        assert instance._budget.spent == 0.0

    @pytest.mark.asyncio()
    async def test_zero_budget_means_unlimited(self) -> None:
        # Given: max_cost_usd=0 (unlimited mode)
        mw = BudgetMiddleware(max_cost_usd=0, cost_per_1k_input_tokens=1.0)
        instance = mw(_make_event(), _make_context())
        # Manually inflate spent to a huge value
        instance._budget._spent = 999999.0
        called = False

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            nonlocal called
            called = True
            return _make_model_response()

        # When: unlimited budget
        await instance.on_llm_call(call_next, [], _make_context())

        # Then: call is NOT blocked
        assert called


# ---------------------------------------------------------------------------
# TestBudgetMiddlewareConstructor
# ---------------------------------------------------------------------------


class TestBudgetMiddlewareConstructor:
    def test_flat_constructor(self) -> None:
        # Given arguments passed directly (no explicit config object)
        mw = BudgetMiddleware(
            max_cost_usd=1.0,
            cost_per_1k_input_tokens=0.30,
            cost_per_1k_output_tokens=0.60,
        )

        # Then the internal config is built from those arguments
        assert mw._config.max_cost_usd == 1.0
        assert mw._config.cost_per_1k_input_tokens == 0.30
        assert mw._config.cost_per_1k_output_tokens == 0.60

    def test_default_constructor(self) -> None:
        # Given no arguments
        mw = BudgetMiddleware()

        # Then all defaults are zero (unlimited, free pricing)
        assert mw._config.max_cost_usd == 0.0
        assert mw._config.cost_per_1k_input_tokens == 0.0
        assert mw._config.cost_per_1k_output_tokens == 0.0


# ---------------------------------------------------------------------------
# TestBudgetConcurrency
# ---------------------------------------------------------------------------


class TestBudgetConcurrency:
    def test_concurrent_budget_record(self) -> None:
        # Given: budget tracker with $0 pricing (tracks tokens only)
        config = BudgetConfig(
            max_cost_usd=100.0,
            cost_per_1k_input_tokens=1.0,
            cost_per_1k_output_tokens=0.0,
        )

        tracker = _BudgetTracker(config)
        usage = {"prompt_tokens": 1000}  # each record adds $1.0

        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(100):
                    tracker.record(usage)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        # When: 10 threads each record 100 times
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Then: no errors, total is exactly 10 * 100 * $1.0 = $1000
        assert not errors
        assert abs(tracker.spent - 1000.0) < 1e-6

    def test_atomic_update_context_with_threads(self) -> None:
        config = BudgetConfig(
            max_cost_usd=100.0,
            cost_per_1k_input_tokens=1.0,
            cost_per_1k_output_tokens=0.0,
        )
        ctx = _make_context()
        usage = {"prompt_tokens": 1000}
        errors: list[Exception] = []
        barrier = threading.Barrier(10)

        def worker() -> None:
            try:
                tracker = _BudgetTracker(config)
                barrier.wait(timeout=5)
                tracker.atomic_update_context(usage, ctx)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert all(not t.is_alive() for t in threads)
        assert not errors
        assert ctx.variables[_SPENT_KEY] == pytest.approx(10.0)

    @pytest.mark.asyncio()
    async def test_malformed_spent_key_in_context_does_not_crash(self) -> None:
        mw = BudgetMiddleware(max_cost_usd=10.0, cost_per_1k_input_tokens=1.0)
        ctx = _make_context()
        ctx.variables[_SPENT_KEY] = "invalid"
        instance = mw(_make_event(), ctx)
        response = _make_model_response(1000, 0, key_style="prompt")

        async def call_next(events: Sequence[BaseEvent], current_ctx: object) -> ModelResponse:
            return response

        await instance.on_llm_call(call_next, [], ctx)

        assert ctx.variables[_SPENT_KEY] == pytest.approx(1.0)

    @pytest.mark.asyncio()
    async def test_record_accepts_usage_instance_through_middleware(self) -> None:
        mw = BudgetMiddleware(
            max_cost_usd=10.0,
            cost_per_1k_input_tokens=0.30,
            cost_per_1k_output_tokens=0.60,
        )
        ctx = _make_context()
        instance = mw(_make_event(), ctx)
        response = ModelResponse(
            message=ModelMessage(content="ok"),
            usage=Usage(prompt_tokens=1000, completion_tokens=500),
        )

        async def call_next(events: Sequence[BaseEvent], current_ctx: object) -> ModelResponse:
            return response

        await instance.on_llm_call(call_next, [], ctx)

        assert abs(instance._budget.spent - 0.60) < 1e-9
        assert ctx.variables[_SPENT_KEY] == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# TestBudgetConcurrencyContext
# ---------------------------------------------------------------------------


class TestBudgetConcurrencyContext:
    @pytest.mark.asyncio()
    async def test_concurrent_instances_accumulate_shared_context_spend(self) -> None:
        mw = BudgetMiddleware(
            max_cost_usd=100.0,
            cost_per_1k_input_tokens=1.0,
            cost_per_1k_output_tokens=0.0,
        )
        ctx = _make_context()
        task_count = 10
        all_waiting = asyncio.Event()
        release = asyncio.Event()
        waiting = 0

        async def call_next(events: Sequence[BaseEvent], current_ctx: object) -> ModelResponse:
            nonlocal waiting
            waiting += 1
            if waiting == task_count:
                all_waiting.set()
            await release.wait()
            return _make_model_response(1000, 0, key_style="prompt")

        async def run_call() -> None:
            instance = mw(_make_event(), ctx)
            await instance.on_llm_call(call_next, [], ctx)

        tasks = [asyncio.create_task(run_call()) for _ in range(task_count)]
        await all_waiting.wait()
        release.set()
        await asyncio.gather(*tasks)

        assert ctx.variables[_SPENT_KEY] == pytest.approx(10.0)

    @pytest.mark.asyncio()
    async def test_concurrent_calls_may_exceed_budget_postpaid(self) -> None:
        mw = BudgetMiddleware(
            max_cost_usd=0.50,
            cost_per_1k_input_tokens=0.50,
            cost_per_1k_output_tokens=0.0,
        )
        ctx = _make_context()
        task_count = 2
        all_waiting = asyncio.Event()
        release = asyncio.Event()
        waiting = 0

        async def call_next(events: Sequence[BaseEvent], current_ctx: object) -> ModelResponse:
            nonlocal waiting
            waiting += 1
            if waiting == task_count:
                all_waiting.set()
            await release.wait()
            return _make_model_response(1000, 0, key_style="prompt")

        async def run_call() -> None:
            instance = mw(_make_event(), ctx)
            await instance.on_llm_call(call_next, [], ctx)

        tasks = [asyncio.create_task(run_call()) for _ in range(task_count)]
        try:
            await asyncio.wait_for(all_waiting.wait(), timeout=1.0)
            release.set()
            await asyncio.gather(*tasks)
        finally:
            release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        assert ctx.variables[_SPENT_KEY] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestBudgetAdversarial
# ---------------------------------------------------------------------------


class TestBudgetAdversarial:
    @pytest.mark.asyncio()
    async def test_negative_tokens_clamped_to_zero(self) -> None:
        # Given: negative token value (malformed provider response)
        mw = BudgetMiddleware(max_cost_usd=10.0, cost_per_1k_input_tokens=1.0)
        instance = mw(_make_event(), _make_context())
        response = ModelResponse(
            message=ModelMessage(content="ok"),
            usage={"prompt_tokens": -500, "completion_tokens": 0},
        )

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return response

        # When: negative tokens provided
        await instance.on_llm_call(call_next, [], _make_context())

        # Then: clamped to zero, no negative cost
        assert instance._budget.spent == 0.0

    @pytest.mark.asyncio()
    @pytest.mark.parametrize(
        ("spent", "limit", "expected_allowed"),
        [
            (0.999, 1.0, True),  # limit - epsilon: allowed
            (1.0, 1.0, False),  # exactly at limit: blocked
            (1.001, 1.0, False),  # limit + epsilon: blocked
        ],
    )
    async def test_budget_boundary_triple(self, spent: float, limit: float, expected_allowed: bool) -> None:
        # Given: budget at boundary conditions
        mw = BudgetMiddleware(max_cost_usd=limit)
        instance = mw(_make_event(), _make_context())
        instance._budget._spent = spent

        called = False

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            nonlocal called
            called = True
            return ModelResponse(message=ModelMessage(content="ok"), usage={})

        # When: call is made at boundary
        if expected_allowed:
            await instance.on_llm_call(call_next, [], _make_context())
        else:
            with pytest.raises(BudgetExceededError):
                await instance.on_llm_call(call_next, [], _make_context())

        # Then: allowed or blocked as expected
        assert called == expected_allowed

    @pytest.mark.asyncio()
    async def test_cumulative_cost_within_single_conversation(self) -> None:
        # Given: a single conversation instance running several LLM calls
        # repeated calls on the same middleware instance keep accumulating locally
        mw = BudgetMiddleware(
            max_cost_usd=1.0,
            cost_per_1k_input_tokens=0.30,
            cost_per_1k_output_tokens=0.0,
        )
        instance = mw(_make_event(), _make_context())

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return _make_model_response(1000, 0, key_style="prompt")

        # When: 3 LLM calls are made inside the same conversation
        for _ in range(3):
            await instance.on_llm_call(call_next, [], _make_context())

        # Then: cumulative cost on this instance is 3 * $0.30 = $0.90
        assert abs(instance._budget.spent - 0.90) < 1e-9

    @pytest.mark.asyncio()
    async def test_tracker_isolated_across_conversations(self) -> None:
        # Given: a single factory used with two separate contexts
        mw = BudgetMiddleware(
            max_cost_usd=1.0,
            cost_per_1k_input_tokens=0.30,
            cost_per_1k_output_tokens=0.0,
        )

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return _make_model_response(1000, 0, key_style="prompt")

        # When: two different instances each run one call
        instance_a = mw(_make_event(), _make_context())
        await instance_a.on_llm_call(call_next, [], _make_context())

        instance_b = mw(_make_event(), _make_context())
        await instance_b.on_llm_call(call_next, [], _make_context())

        # Then: each conversation has its own tracker, neither sees the other's spend
        assert abs(instance_a._budget.spent - 0.30) < 1e-9
        assert abs(instance_b._budget.spent - 0.30) < 1e-9
        assert instance_a._budget is not instance_b._budget

    @pytest.mark.asyncio()
    async def test_tracker_persists_across_reply_ask(self) -> None:
        mw = BudgetMiddleware(
            max_cost_usd=1.0,
            cost_per_1k_input_tokens=0.30,
            cost_per_1k_output_tokens=0.0,
        )
        ctx = _make_context()

        async def call_next(events: Sequence[BaseEvent], current_ctx: object) -> ModelResponse:
            return _make_model_response(1000, 0, key_style="prompt")

        first_instance = mw(_make_event(), ctx)
        await first_instance.on_llm_call(call_next, [], ctx)

        assert ctx.variables[_SPENT_KEY] == pytest.approx(0.30)

        second_instance = mw(_make_event(), ctx)

        assert second_instance._budget.spent == pytest.approx(0.30)

        await second_instance.on_llm_call(call_next, [], ctx)

        assert second_instance._budget.spent == pytest.approx(0.60)
        assert ctx.variables[_SPENT_KEY] == pytest.approx(0.60)

    @pytest.mark.asyncio()
    async def test_cross_context_isolation(self) -> None:
        mw = BudgetMiddleware(
            max_cost_usd=1.0,
            cost_per_1k_input_tokens=0.30,
            cost_per_1k_output_tokens=0.0,
        )
        ctx_a = _make_context()
        ctx_b = _make_context()

        async def call_next(events: Sequence[BaseEvent], current_ctx: object) -> ModelResponse:
            return _make_model_response(1000, 0, key_style="prompt")

        first_instance = mw(_make_event(), ctx_a)
        await first_instance.on_llm_call(call_next, [], ctx_a)

        second_instance = mw(_make_event(), ctx_b)

        assert ctx_a.variables[_SPENT_KEY] == pytest.approx(0.30)
        assert _SPENT_KEY not in ctx_b.variables
        assert second_instance._budget.spent == 0.0


# ---------------------------------------------------------------------------
# Standalone tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_zero_budget_allows_calls_async() -> None:
    # Given: full async path with max_cost_usd=0 (unlimited)
    mw = BudgetMiddleware(
        max_cost_usd=0,
        cost_per_1k_input_tokens=1.0,
        cost_per_1k_output_tokens=1.0,
    )
    responses = []

    async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
        r = _make_model_response(500, 500)
        responses.append(r)
        return r

    # When: many calls are made
    for _ in range(5):
        instance = mw(_make_event(), _make_context())
        result = await instance.on_llm_call(call_next, [], _make_context())
        assert result.message is not None

    # Then: all 5 calls went through
    assert len(responses) == 5


def test_budget_importable_from_builtin_init() -> None:
    # Given/When/Then: BudgetConfig and BudgetMiddleware are importable from builtin __init__
    import autogen.beta.middleware.builtin as builtin_pkg

    assert builtin_pkg.BudgetConfig is BudgetConfig
    assert builtin_pkg.BudgetMiddleware is BudgetMiddleware


def test_budget_exceeded_error_importable_from_builtin_init() -> None:
    # Given/When/Then: BudgetExceededError is importable from builtin __init__
    import autogen.beta.middleware.builtin as builtin_pkg

    assert builtin_pkg.BudgetExceededError is BudgetExceededError


def test_budget_importable_from_middleware_top_level() -> None:
    # Given/When/Then: BudgetConfig, BudgetMiddleware, BudgetExceededError are importable
    # from the top-level autogen.beta.middleware package (B-2 regression)
    import autogen.beta.middleware as mw_pkg

    assert mw_pkg.BudgetConfig is BudgetConfig
    assert mw_pkg.BudgetMiddleware is BudgetMiddleware
    assert mw_pkg.BudgetExceededError is BudgetExceededError


def test_record_accepts_usage_dataclass() -> None:
    tracker = _BudgetTracker(
        BudgetConfig(
            max_cost_usd=10.0,
            cost_per_1k_input_tokens=0.30,
            cost_per_1k_output_tokens=0.60,
        )
    )

    tracker.record(Usage(prompt_tokens=1000, completion_tokens=500))

    assert tracker.spent == pytest.approx(0.60)
