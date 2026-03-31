# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for BudgetMiddleware."""

import math
import threading
from collections.abc import Sequence
from unittest import mock

import pytest

from autogen.beta.events import BaseEvent, ModelMessage, ModelResponse
from autogen.beta.middleware.builtin.budget import BudgetConfig, BudgetMiddleware

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event() -> mock.MagicMock:
    return mock.MagicMock(spec=BaseEvent)


def _make_context() -> mock.MagicMock:
    return mock.MagicMock()


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
        config = BudgetConfig(
            max_cost_usd=1.0,
            cost_per_1k_input_tokens=0.30,
            cost_per_1k_output_tokens=0.60,
        )
        mw = BudgetMiddleware(config)
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
    async def test_blocks_when_exhausted(self) -> None:
        # Given: budget exhausted by directly setting _spent above limit
        config = BudgetConfig(max_cost_usd=0.001)
        mw = BudgetMiddleware(config)
        mw._budget._spent = 0.002  # manually exhaust

        instance = mw(_make_event(), _make_context())
        called = False

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            nonlocal called
            called = True
            return _make_model_response()

        # When: budget is exhausted
        result = await instance.on_llm_call(call_next, [], _make_context())

        # Then: call is blocked and downstream is not invoked
        assert result.message is None
        assert not called

    @pytest.mark.asyncio()
    async def test_records_exact_cost(self) -> None:
        # Given: $0.30/1k input, $0.60/1k output
        config = BudgetConfig(
            max_cost_usd=10.0,
            cost_per_1k_input_tokens=0.30,
            cost_per_1k_output_tokens=0.60,
        )
        mw = BudgetMiddleware(config)
        instance = mw(_make_event(), _make_context())
        # 1000 prompt + 500 output = 1000/1000*0.30 + 500/1000*0.60 = 0.30 + 0.30 = 0.60 USD
        response = _make_model_response(1000, 500, key_style="prompt")

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return response

        # When: call completes
        await instance.on_llm_call(call_next, [], _make_context())

        # Then: cost is 0.60
        assert abs(mw._budget.spent - 0.60) < 1e-9

    @pytest.mark.asyncio()
    async def test_completion_tokens_key(self) -> None:
        # Given: OpenAI-style keys with completion_tokens
        config = BudgetConfig(
            max_cost_usd=10.0,
            cost_per_1k_input_tokens=0.0,
            cost_per_1k_output_tokens=1.0,
        )
        mw = BudgetMiddleware(config)
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
        assert abs(mw._budget.spent - 2.0) < 1e-9

    @pytest.mark.asyncio()
    async def test_zero_prompt_tokens_not_masked(self) -> None:
        # Given: prompt_tokens=0 is present; input_tokens=9999 is also present
        # The `in` check must use prompt_tokens (not fall through to input_tokens)
        config = BudgetConfig(
            max_cost_usd=10.0,
            cost_per_1k_input_tokens=1.0,
            cost_per_1k_output_tokens=0.0,
        )
        mw = BudgetMiddleware(config)
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
        assert mw._budget.spent == 0.0

    @pytest.mark.asyncio()
    async def test_input_tokens_key_gemini_style(self) -> None:
        # Given: generic input_tokens / output_tokens keys
        config = BudgetConfig(
            max_cost_usd=10.0,
            cost_per_1k_input_tokens=1.0,
            cost_per_1k_output_tokens=0.0,
        )
        mw = BudgetMiddleware(config)
        instance = mw(_make_event(), _make_context())
        response = _make_model_response(2000, 0, key_style="input")

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return response

        # When: call uses input_tokens key
        await instance.on_llm_call(call_next, [], _make_context())

        # Then: 2000/1000 * $1.0 = $2.0
        assert abs(mw._budget.spent - 2.0) < 1e-9

    @pytest.mark.asyncio()
    async def test_empty_usage_no_cost(self) -> None:
        # Given: response with no usage dict entries
        config = BudgetConfig(max_cost_usd=10.0, cost_per_1k_input_tokens=1.0)
        mw = BudgetMiddleware(config)
        instance = mw(_make_event(), _make_context())
        response = ModelResponse(message=ModelMessage(content="ok"), usage={})

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return response

        # When: call has empty usage
        await instance.on_llm_call(call_next, [], _make_context())

        # Then: no cost accumulated
        assert mw._budget.spent == 0.0

    @pytest.mark.asyncio()
    async def test_gemini_token_keys(self) -> None:
        # Given: Gemini-style prompt_token_count / candidates_token_count
        config = BudgetConfig(
            max_cost_usd=10.0,
            cost_per_1k_input_tokens=0.30,
            cost_per_1k_output_tokens=0.60,
        )
        mw = BudgetMiddleware(config)
        instance = mw(_make_event(), _make_context())
        response = _make_model_response(1000, 1000, key_style="prompt_count")

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return response

        # When: Gemini keys are used
        await instance.on_llm_call(call_next, [], _make_context())

        # Then: 1000/1000 * 0.30 + 1000/1000 * 0.60 = 0.90
        assert abs(mw._budget.spent - 0.90) < 1e-9

    @pytest.mark.asyncio()
    async def test_string_token_values_coerced(self) -> None:
        # Given: token values as numeric strings (some APIs return strings)
        config = BudgetConfig(
            max_cost_usd=10.0,
            cost_per_1k_input_tokens=1.0,
            cost_per_1k_output_tokens=0.0,
        )
        mw = BudgetMiddleware(config)
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
        assert abs(mw._budget.spent - 0.50) < 1e-9

    @pytest.mark.asyncio()
    async def test_non_numeric_string_tokens_ignored(self) -> None:
        # Given: non-numeric token value
        config = BudgetConfig(max_cost_usd=10.0, cost_per_1k_input_tokens=1.0)
        mw = BudgetMiddleware(config)
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
        assert mw._budget.spent == 0.0

    @pytest.mark.asyncio()
    async def test_nan_tokens_ignored(self) -> None:
        # Given: NaN token value
        config = BudgetConfig(max_cost_usd=10.0, cost_per_1k_input_tokens=1.0)
        mw = BudgetMiddleware(config)
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
        assert mw._budget.spent == 0.0

    @pytest.mark.asyncio()
    async def test_none_tokens_no_crash(self) -> None:
        # Given: None token value
        config = BudgetConfig(max_cost_usd=10.0, cost_per_1k_input_tokens=1.0)
        mw = BudgetMiddleware(config)
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
        assert mw._budget.spent == 0.0

    @pytest.mark.asyncio()
    async def test_zero_budget_means_unlimited(self) -> None:
        # Given: max_cost_usd=0 (unlimited mode)
        config = BudgetConfig(max_cost_usd=0, cost_per_1k_input_tokens=1.0)
        mw = BudgetMiddleware(config)
        # Manually inflate spent to a huge value
        mw._budget._spent = 999999.0
        instance = mw(_make_event(), _make_context())
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
# TestBudgetConfigValidation
# ---------------------------------------------------------------------------


class TestBudgetConfigValidation:
    def test_negative_budget_rejected(self) -> None:
        # Given/When/Then: negative max_cost_usd raises ValueError
        with pytest.raises(ValueError, match="max_cost_usd"):
            BudgetConfig(max_cost_usd=-0.01)

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("cost_per_1k_input_tokens", math.nan),
            ("cost_per_1k_input_tokens", math.inf),
            ("cost_per_1k_input_tokens", -1.0),
            ("cost_per_1k_output_tokens", math.nan),
            ("cost_per_1k_output_tokens", math.inf),
            ("cost_per_1k_output_tokens", -1.0),
        ],
    )
    def test_non_finite_pricing_rejected(self, field_name: str, value: float) -> None:
        # Given/When/Then: invalid pricing raises ValueError
        with pytest.raises(ValueError, match=field_name):
            BudgetConfig(**{field_name: value})


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
        from autogen.beta.middleware.builtin.budget import _BudgetTracker

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


# ---------------------------------------------------------------------------
# TestBudgetAdversarial
# ---------------------------------------------------------------------------


class TestBudgetAdversarial:
    @pytest.mark.asyncio()
    async def test_negative_tokens_clamped_to_zero(self) -> None:
        # Given: negative token value (malformed provider response)
        config = BudgetConfig(max_cost_usd=10.0, cost_per_1k_input_tokens=1.0)
        mw = BudgetMiddleware(config)
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
        assert mw._budget.spent == 0.0

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
        config = BudgetConfig(max_cost_usd=limit)
        mw = BudgetMiddleware(config)
        mw._budget._spent = spent

        instance = mw(_make_event(), _make_context())
        called = False

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            nonlocal called
            called = True
            return ModelResponse(message=ModelMessage(content="ok"), usage={})

        # When: call is made at boundary
        await instance.on_llm_call(call_next, [], _make_context())

        # Then: allowed or blocked as expected
        assert called == expected_allowed

    @pytest.mark.asyncio()
    async def test_cumulative_cost_across_calls(self) -> None:
        # Given: multiple calls each costing $0.30
        config = BudgetConfig(
            max_cost_usd=1.0,
            cost_per_1k_input_tokens=0.30,
            cost_per_1k_output_tokens=0.0,
        )
        mw = BudgetMiddleware(config)

        async def call_next(events: Sequence[BaseEvent], ctx: object) -> ModelResponse:
            return _make_model_response(1000, 0, key_style="prompt")

        # When: 3 calls are made
        for _ in range(3):
            instance = mw(_make_event(), _make_context())
            await instance.on_llm_call(call_next, [], _make_context())

        # Then: cumulative cost is 3 * $0.30 = $0.90
        assert abs(mw._budget.spent - 0.90) < 1e-9


# ---------------------------------------------------------------------------
# Standalone tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_zero_budget_allows_calls_async() -> None:
    # Given: full async path with max_cost_usd=0 (unlimited)
    config = BudgetConfig(
        max_cost_usd=0,
        cost_per_1k_input_tokens=1.0,
        cost_per_1k_output_tokens=1.0,
    )
    mw = BudgetMiddleware(config)
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
