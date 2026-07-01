# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Real-LLM behaviour tests for the Agent-as-judge scorer.

These hit real provider APIs and are gated by per-provider marks, so they are
excluded from the default unit run and exercised by the weekly ``beta-llm-test``
action (``-m openai`` / ``-m anthropic`` / ``-m gemini``). They verify what the
mock tests in ``test_agent_judge.py`` cannot: that a real model, given the judge
prompt, returns a valid ``Verdict`` (structured output works end to end) and
actually *discriminates* — a correct answer scores higher than a wrong one.

Run locally with e.g. ``GEMINI_API_KEY=… pytest test/eval/test_agent_judge_llm.py -m gemini``.
"""

import os

import pytest

from ag2.config import AnthropicConfig, GeminiConfig, OpenAIConfig
from ag2.eval.dataset.task import Task
from ag2.eval.scorers import agent_judge
from ag2.eval.trace import Trace
from ag2.events import ModelMessage, ModelResponse


def _require(*envs: str) -> str:
    for env in envs:
        value = os.getenv(env)
        if value:
            return value
    pytest.skip(f"{' / '.join(envs)} not set; skipping real-LLM judge test")


@pytest.fixture(
    params=[
        pytest.param("openai", marks=pytest.mark.openai),
        pytest.param("anthropic", marks=pytest.mark.anthropic),
        pytest.param("gemini", marks=pytest.mark.gemini),
    ]
)
def judge_config(request):
    if request.param == "openai":
        return OpenAIConfig(model="gpt-5.4-nano", api_key=_require("OPENAI_API_KEY"), temperature=0)
    if request.param == "anthropic":
        return AnthropicConfig(model="claude-haiku-4-5", api_key=_require("ANTHROPIC_API_KEY"), temperature=0)
    return GeminiConfig(
        model="gemini-3.1-flash-lite", api_key=_require("GEMINI_API_KEY", "GOOGLE_API_KEY"), temperature=0
    )


def _answer_trace(answer: str) -> Trace:
    return Trace(events=[ModelResponse(message=ModelMessage(answer))], exception=None, duration_ms=0)


async def _grade(scorer, *, answer: str, reference: dict | None) -> object:
    [feedback] = await scorer(
        inputs={"input": "What is the capital of France?"},
        outputs={"body": answer},
        reference_outputs=reference,
        trace=_answer_trace(answer),
        task=Task(task_id="t", inputs={"input": "What is the capital of France?"}, reference_outputs=reference),
    )
    return feedback


@pytest.mark.asyncio()
async def test_judge_discriminates_correct_from_wrong(judge_config) -> None:
    judge = agent_judge(judge_config, criterion="The answer matches the reference answer.", key="correctness")

    correct = await _grade(judge, answer="The capital of France is Paris.", reference={"answer": "Paris"})
    wrong = await _grade(judge, answer="The capital of France is Berlin.", reference={"answer": "Paris"})

    # structured output actually worked: real, in-range floats with a rationale
    for fb in (correct, wrong):
        assert isinstance(fb.score, float)
        assert 0.0 <= fb.score <= 1.0
        assert fb.comment
    # and the judge genuinely discriminates
    assert correct.score > wrong.score


@pytest.mark.asyncio()
async def test_multi_dimensional_keys_are_independent(judge_config) -> None:
    correctness = agent_judge(judge_config, criterion="The answer is factually correct.", key="correctness")
    conciseness = agent_judge(
        judge_config, criterion="The answer is concise (one sentence, no padding).", key="conciseness"
    )

    answer = "Paris."
    c = await _grade(correctness, answer=answer, reference={"answer": "Paris"})
    n = await _grade(conciseness, answer=answer, reference=None)

    assert c.key == "correctness" and isinstance(c.score, float)
    assert n.key == "conciseness" and isinstance(n.score, float)
