# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pairwise Agent-as-judge and its dual-order position swap.

Where the two orders need different verdicts, a persistent TestConfig returns
them in order across the judge's two calls (order 1, then order 2), so we can
drive the swap deterministically and confirm the conservative "win only if
consistent" rule — including catching a position-bias flip as a tie.
"""

import pytest

from ag2.eval.dataset.task import Task
from ag2.eval.pairwise import PairwiseComparator
from ag2.eval.scorers import pairwise_judge
from ag2.eval.trace import Trace
from ag2.events import ModelMessage, ModelResponse
from ag2.testing import TestConfig, TrackingConfig

_TASK = Task(task_id="t", inputs={"input": "Which is better?"})


def _trace(answer: str) -> Trace:
    return Trace(events=[ModelResponse(message=ModelMessage(answer))], exception=None, duration_ms=0)


async def _compare(judge: PairwiseComparator):
    return await judge.compare(
        task=_TASK, trace_a=_trace("answer A"), trace_b=_trace("answer B"), reference_outputs=None
    )


def _verdict(pref: str, reason: str = "x") -> str:
    return f'{{"preferred": "{pref}", "reasoning": "{reason}"}}'


@pytest.mark.asyncio()
async def test_consistent_winner_across_orders() -> None:
    # the judge prefers A in BOTH orders: "first" while A leads, "second" once swapped -> A wins
    config = TestConfig(_verdict("first"), _verdict("second"), persistent=True)
    judge = pairwise_judge(config, criterion="quality", key="quality")
    assert isinstance(judge, PairwiseComparator)

    outcome = await _compare(judge)
    assert outcome.winner == "a"
    assert outcome.detail == {"order1": "first", "order2": "second"}


@pytest.mark.asyncio()
async def test_second_order_swaps_the_answers_in_the_prompt() -> None:
    config = TrackingConfig(TestConfig(_verdict("tie")))
    judge = pairwise_judge(config, criterion="quality", key="quality")

    await _compare(judge)

    order1, order2 = (repr(call.args[0]) for call in config.mock.call_args_list)
    assert order1.index("answer A") < order1.index("answer B")
    assert order2.index("answer B") < order2.index("answer A")


@pytest.mark.asyncio()
async def test_position_bias_flip_resolves_to_tie() -> None:
    # judge always says "first" -> order1 picks A, order2 picks B -> inconsistent -> tie
    judge = pairwise_judge(TestConfig(_verdict("first")), criterion="quality", key="quality")

    outcome = await _compare(judge)
    assert outcome.winner == "tie"
    assert outcome.detail["order1"] == "first" and outcome.detail["order2"] == "first"


@pytest.mark.asyncio()
async def test_genuine_tie_in_both_orders() -> None:
    judge = pairwise_judge(TestConfig(_verdict("tie")), criterion="quality", key="quality")
    assert (await _compare(judge)).winner == "tie"


@pytest.mark.asyncio()
async def test_reference_rendered_into_prompt_by_default() -> None:
    config = TrackingConfig(TestConfig(_verdict("tie")))
    judge = pairwise_judge(config, criterion="quality", key="quality")

    await judge.compare(
        task=_TASK, trace_a=_trace("answer A"), trace_b=_trace("answer B"), reference_outputs={"answer": "gold"}
    )

    prompt = repr(config.mock.call_args.args[0])
    assert "## Reference" in prompt
    assert "gold" in prompt


@pytest.mark.asyncio()
async def test_include_reference_false_omits_reference_section() -> None:
    config = TrackingConfig(TestConfig(_verdict("tie")))
    judge = pairwise_judge(config, criterion="quality", key="quality", include_reference=False)

    await judge.compare(
        task=_TASK, trace_a=_trace("answer A"), trace_b=_trace("answer B"), reference_outputs={"answer": "gold"}
    )

    prompt = repr(config.mock.call_args.args[0])
    assert "## Reference" not in prompt
    assert "gold" not in prompt


@pytest.mark.asyncio()
async def test_no_swap_uses_single_call() -> None:
    # swap off: one call; "second" with Resp1=A -> B
    judge = pairwise_judge(TestConfig(_verdict("second")), criterion="quality", key="quality", swap=False)

    outcome = await _compare(judge)
    assert outcome.winner == "b"
    assert "order2" not in outcome.detail
