# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

llm_prices = pytest.importorskip("llm_prices")

from ag2.eval import Feedback, Scorer, Task, Trace  # noqa: E402
from ag2.eval.scorers import cost_budget, token_budget  # noqa: E402
from ag2.events import BaseEvent, Usage, UsageEvent  # noqa: E402
from ag2.usage import price_usage  # noqa: E402

_CACHED = Usage(prompt_tokens=100_000, completion_tokens=1_000, cache_read_input_tokens=95_000)


def _model() -> str:
    return next(p for p in llm_prices.Provider.ANTHROPIC.rows() if p.cache_read is not None).model


def _trace(*events: BaseEvent) -> Trace:
    return Trace(events=list(events), exception=None, duration_ms=0)


async def _score(scorer: Scorer, trace: Trace) -> list[Feedback]:
    return await scorer(
        inputs={"input": "?"},
        outputs={},
        reference_outputs=None,
        trace=trace,
        task=Task(task_id="t1", inputs={"input": "?"}),
    )


@pytest.mark.asyncio
class TestCostBudget:
    async def test_passes_at_or_under_the_priced_cost(self) -> None:
        model = _model()
        usd = price_usage(_CACHED, model=model, provider="anthropic")
        trace = _trace(UsageEvent(_CACHED, model=model, provider="anthropic"))

        feedback = await _score(cost_budget(usd), trace)

        assert feedback == [Feedback(key="cost_budget", score=True)]
        assert feedback[0].detail == {"usd": pytest.approx(usd), "max_usd": usd}

    async def test_fails_over_the_priced_cost(self) -> None:
        model = _model()
        usd = price_usage(_CACHED, model=model, provider="anthropic")
        trace = _trace(UsageEvent(_CACHED, model=model, provider="anthropic"))

        feedback = await _score(cost_budget(usd / 2), trace)

        assert feedback[0].score is False

    async def test_prompt_cache_passes_a_dollar_budget_a_token_budget_fails(self) -> None:
        """Same task, two budgets: the token count treats 95k cached tokens as fresh input."""
        model = _model()
        fresh = Usage(prompt_tokens=100_000, completion_tokens=1_000)
        trace = _trace(UsageEvent(_CACHED, model=model, provider="anthropic"))

        by_tokens = await _score(token_budget(50_000), trace)
        by_dollars = await _score(cost_budget(price_usage(fresh, model=model, provider="anthropic") / 2), trace)

        assert by_tokens[0].score is False
        assert by_dollars[0].score is True

    async def test_unpriced_spend_is_ungraded(self) -> None:
        trace = _trace(
            UsageEvent(Usage(prompt_tokens=10, completion_tokens=1), model=_model(), provider="anthropic"),
            UsageEvent(Usage(prompt_tokens=10, completion_tokens=1), model="llama3", provider="ollama"),
        )

        feedback = await _score(cost_budget(100.0), trace)

        assert feedback[0].score is None
        assert "ollama/llama3" in (feedback[0].comment or "")
