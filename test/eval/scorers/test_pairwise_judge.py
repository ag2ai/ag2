# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pairwise Agent-as-judge and its dual-order position swap.

TestConfig with two responses returns them in order across the judge's two
calls (order 1, then order 2), so we can drive the swap deterministically and
confirm the conservative "win only if consistent" rule — including catching a
position-bias flip as a tie.
"""

import json
from typing import Any

import pytest

from ag2.config import LLMClient, ModelConfig
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


class _PrefersAClient(LLMClient):
    """A content-based judge: prefers whichever response contains 'answer A'."""

    async def __call__(self, messages: Any, context: Any, **kwargs: Any) -> ModelResponse:
        text = " ".join(repr(m) for m in messages)  # prompt lives in ModelRequest.parts[*].content
        preferred = "first" if text.find("answer A") < text.find("answer B") else "second"
        message = ModelMessage(json.dumps({"preferred": preferred, "reasoning": "prefers A"}))
        await context.send(message)
        return ModelResponse(message=message)


class _PrefersAConfig(ModelConfig):
    def copy(self) -> "_PrefersAConfig":
        return self

    def create(self) -> _PrefersAClient:
        return _PrefersAClient()

    def create_files_client(self) -> None:
        raise NotImplementedError


@pytest.mark.asyncio()
async def test_content_based_winner_is_consistent_across_orders() -> None:
    # a content-based judge prefers A in BOTH orders -> A wins (not a position artifact)
    judge = pairwise_judge(_PrefersAConfig(), criterion="quality", key="quality")
    assert isinstance(judge, PairwiseComparator)

    outcome = await _compare(judge)
    assert outcome.winner == "a"
    assert outcome.detail == {"order1": "first", "order2": "second"}


@pytest.mark.asyncio()
async def test_position_bias_flip_resolves_to_tie() -> None:
    # judge always says "first" -> order1 picks A, order2 picks B -> inconsistent -> tie
    judge = pairwise_judge(TestConfig(_verdict("first"), _verdict("first")), criterion="quality", key="quality")

    outcome = await _compare(judge)
    assert outcome.winner == "tie"
    assert outcome.detail["order1"] == "first" and outcome.detail["order2"] == "first"


@pytest.mark.asyncio()
async def test_genuine_tie_in_both_orders() -> None:
    judge = pairwise_judge(TestConfig(_verdict("tie"), _verdict("tie")), criterion="quality", key="quality")
    assert (await _compare(judge)).winner == "tie"


@pytest.mark.asyncio()
async def test_reference_rendered_into_prompt_by_default() -> None:
    config = TrackingConfig(TestConfig(_verdict("tie"), _verdict("tie")))
    judge = pairwise_judge(config, criterion="quality", key="quality")

    await judge.compare(
        task=_TASK, trace_a=_trace("answer A"), trace_b=_trace("answer B"), reference_outputs={"answer": "gold"}
    )

    prompt = repr(config.mock.call_args.args[0])
    assert "## Reference" in prompt
    assert "gold" in prompt


@pytest.mark.asyncio()
async def test_include_reference_false_omits_reference_section() -> None:
    config = TrackingConfig(TestConfig(_verdict("tie"), _verdict("tie")))
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
