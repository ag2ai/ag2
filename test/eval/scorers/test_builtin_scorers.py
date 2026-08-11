# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Public-API tests for the prebuilt scorers in ``ag2.eval.scorers``."""

from typing import Any

import pytest

from ag2.eval import Feedback, Scorer, Task, Trace
from ag2.eval.scorers import (
    final_answer_matches,
    no_tool_errors,
    token_budget,
    tool_called,
)
from ag2.events import (
    BaseEvent,
    ModelResponse,
    ToolCallEvent,
    ToolErrorEvent,
    Usage,
)


def _trace(*events: BaseEvent) -> Trace:
    return Trace(events=list(events), exception=None, duration_ms=0)


async def _run(
    s: Scorer,
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    reference_outputs: dict[str, Any] | None = None,
    trace: Trace | None = None,
    task: Task | None = None,
) -> list[Feedback]:
    return await s(
        inputs=inputs if inputs is not None else {"input": "?"},
        outputs=outputs if outputs is not None else {},
        reference_outputs=reference_outputs,
        trace=trace if trace is not None else _trace(),
        task=task if task is not None else Task(task_id="t1", inputs={"input": "?"}),
    )


class TestToolCalled:
    @pytest.mark.asyncio
    async def test_passes_when_tool_was_called(self) -> None:
        trace = _trace(ToolCallEvent(name="get_weather", arguments="{}"))

        feedback = await _run(tool_called("get_weather"), trace=trace)

        assert feedback == [Feedback(key="tool_called[get_weather]", score=True)]

    @pytest.mark.asyncio
    async def test_fails_when_tool_was_not_called(self) -> None:
        trace = _trace(ToolCallEvent(name="something_else", arguments="{}"))

        feedback = await _run(tool_called("get_weather"), trace=trace)

        assert feedback[0].score is False

    @pytest.mark.asyncio
    async def test_exactly_count_pass(self) -> None:
        trace = _trace(
            ToolCallEvent(name="get_weather", arguments="{}"),
            ToolCallEvent(name="get_weather", arguments="{}"),
        )

        feedback = await _run(tool_called("get_weather", exactly=2), trace=trace)

        assert feedback[0].score is True

    @pytest.mark.asyncio
    async def test_exactly_count_fail_on_too_many(self) -> None:
        trace = _trace(
            ToolCallEvent(name="get_weather", arguments="{}"),
            ToolCallEvent(name="get_weather", arguments="{}"),
        )

        feedback = await _run(tool_called("get_weather", exactly=1), trace=trace)

        assert feedback[0].score is False

    @pytest.mark.asyncio
    async def test_key_includes_tool_name(self) -> None:
        feedback = await _run(tool_called("get_weather"), trace=_trace())

        assert feedback[0].key == "tool_called[get_weather]"

    def test_distinct_factory_invocations_have_distinct_keys(self) -> None:
        """Two factory calls produce keys that won't collide in aggregates."""
        a = tool_called("get_weather")
        b = tool_called("get_news")

        assert a.key != b.key


class TestNoToolErrors:
    @pytest.mark.asyncio
    async def test_passes_when_no_errors(self) -> None:
        trace = _trace(ToolCallEvent(name="ok", arguments="{}"))

        feedback = await _run(no_tool_errors(), trace=trace)

        assert feedback == [Feedback(key="no_tool_errors", score=True)]

    @pytest.mark.asyncio
    async def test_fails_when_a_tool_error_fired(self) -> None:
        call = ToolCallEvent(name="bad_tool", arguments="{}")
        err = ToolErrorEvent.from_call(call, RuntimeError("kaboom"))
        trace = _trace(call, err)

        feedback = await _run(no_tool_errors(), trace=trace)

        assert feedback[0].score is False


class TestFinalAnswerMatches:
    @pytest.mark.asyncio
    async def test_casefold_match_passes(self) -> None:
        feedback = await _run(
            final_answer_matches(field="city"),
            outputs={"body": "TOKYO"},
            reference_outputs={"city": "tokyo"},
        )

        assert feedback[0].score is True

    @pytest.mark.asyncio
    async def test_exact_match_distinguishes_case(self) -> None:
        feedback = await _run(
            final_answer_matches(field="city", matcher="exact"),
            outputs={"body": "TOKYO"},
            reference_outputs={"city": "tokyo"},
        )

        assert feedback[0].score is False

    @pytest.mark.asyncio
    async def test_contains_finds_expected_in_actual(self) -> None:
        feedback = await _run(
            final_answer_matches(field="city", matcher="contains"),
            outputs={"body": "Tokyo is sunny and warm today."},
            reference_outputs={"city": "Tokyo"},
        )

        assert feedback[0].score is True

    @pytest.mark.asyncio
    async def test_contains_fails_when_expected_absent(self) -> None:
        feedback = await _run(
            final_answer_matches(field="city", matcher="contains"),
            outputs={"body": "Paris is sunny today."},
            reference_outputs={"city": "Tokyo"},
        )

        assert feedback[0].score is False

    @pytest.mark.asyncio
    async def test_uses_structured_field_when_outputs_has_one(self) -> None:
        """When the agent used response_schema=, outputs["content"] is the response dict."""
        feedback = await _run(
            final_answer_matches(field="city"),
            outputs={"content": {"city": "Tokyo", "temperature": "72F"}},
            reference_outputs={"city": "Tokyo"},
        )

        assert feedback[0].score is True

    @pytest.mark.asyncio
    async def test_returns_false_when_reference_outputs_missing(self) -> None:
        feedback = await _run(
            final_answer_matches(field="city"),
            outputs={"body": "Tokyo"},
            reference_outputs=None,
        )

        assert feedback[0].score is False

    @pytest.mark.asyncio
    async def test_returns_false_when_field_missing_from_reference(self) -> None:
        feedback = await _run(
            final_answer_matches(field="city"),
            outputs={"body": "Tokyo"},
            reference_outputs={"temperature": "72F"},
        )

        assert feedback[0].score is False


class TestTokenBudget:
    @pytest.mark.asyncio
    async def test_passes_when_under_budget(self) -> None:
        trace = _trace(ModelResponse(usage=Usage(prompt_tokens=100, completion_tokens=50)))

        feedback = await _run(token_budget(1_000), trace=trace)

        assert feedback[0].score is True

    @pytest.mark.asyncio
    async def test_fails_when_over_budget(self) -> None:
        trace = _trace(ModelResponse(usage=Usage(prompt_tokens=1_500, completion_tokens=600)))

        feedback = await _run(token_budget(1_000), trace=trace)

        assert feedback[0].score is False

    @pytest.mark.asyncio
    async def test_passes_at_exact_budget(self) -> None:
        trace = _trace(ModelResponse(usage=Usage(prompt_tokens=600, completion_tokens=400)))

        feedback = await _run(token_budget(1_000), trace=trace)

        assert feedback[0].score is True

    @pytest.mark.asyncio
    async def test_zero_tokens_passes(self) -> None:
        feedback = await _run(token_budget(100), trace=_trace())

        assert feedback[0].score is True
