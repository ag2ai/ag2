# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Public-API tests for the ``@scorer`` decorator and :class:`Scorer`.

Covers the three responsibilities the decorator owns:

* signature introspection (only the parameters the function declares are injected)
* return-value normalization (bool/int/float/str/Feedback/list[Feedback]/None)
* exception capture (a raising scorer never fails the run)
"""

from typing import Any

import pytest

from ag2.eval import Feedback, Scorer, ScorerReturnTypeError, Task, Trace, scorer
from ag2.events import ToolCallEvent


def _trace(*events: Any, duration_ms: int = 0) -> Trace:
    return Trace(events=list(events), exception=None, duration_ms=duration_ms)


def _task(task_id: str = "t-0", **kwargs: Any) -> Task:
    inputs = kwargs.pop("inputs", {"input": "hi"})
    return Task(task_id=task_id, inputs=inputs, **kwargs)


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
        inputs=inputs if inputs is not None else {"input": "hi"},
        outputs=outputs if outputs is not None else {"body": "ok"},
        reference_outputs=reference_outputs,
        trace=trace if trace is not None else _trace(),
        task=task if task is not None else _task(),
    )


class TestSignatureInjection:
    @pytest.mark.asyncio
    async def test_injects_only_declared_parameters(self) -> None:
        seen: dict[str, Any] = {}

        @scorer
        def only_trace(trace: Trace) -> bool:
            seen["trace"] = trace
            return True

        trace = _trace(ToolCallEvent(name="x", arguments="{}"))
        await _run(only_trace, trace=trace)

        assert list(seen) == ["trace"]
        assert seen["trace"] is trace

    @pytest.mark.asyncio
    async def test_supports_full_parameter_set(self) -> None:
        captured: dict[str, Any] = {}

        @scorer
        def full(inputs, outputs, reference_outputs, trace, task) -> bool:  # type: ignore[no-untyped-def]
            captured["inputs"] = inputs
            captured["outputs"] = outputs
            captured["reference_outputs"] = reference_outputs
            captured["trace"] = trace
            captured["task"] = task
            return True

        trace = _trace()
        task = _task(task_id="weather-001")
        await _run(
            full,
            inputs={"input": "Tokyo?"},
            outputs={"city": "Tokyo"},
            reference_outputs={"city": "Tokyo"},
            trace=trace,
            task=task,
        )

        assert captured == {
            "inputs": {"input": "Tokyo?"},
            "outputs": {"city": "Tokyo"},
            "reference_outputs": {"city": "Tokyo"},
            "trace": trace,
            "task": task,
        }

    def test_unknown_parameter_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="not injectable"):

            @scorer
            def bad(trace, surprise) -> bool:  # type: ignore[no-untyped-def]
                return True

    def test_var_args_rejected(self) -> None:
        with pytest.raises(TypeError, match=r"\*args is not supported"):

            @scorer
            def bad(*args: Any) -> bool:
                return True

    def test_var_kwargs_rejected(self) -> None:
        with pytest.raises(TypeError, match=r"\*\*kwargs is not supported"):

            @scorer
            def bad(**kwargs: Any) -> bool:
                return True

    @pytest.mark.asyncio
    async def test_zero_parameter_scorer_runs(self) -> None:
        @scorer
        def constant() -> bool:
            return True

        feedback = await _run(constant)

        assert feedback == [Feedback(key="constant", score=True)]


class TestReturnNormalization:
    @pytest.mark.asyncio
    async def test_bool_true_becomes_pass_feedback(self) -> None:
        @scorer
        def passes() -> bool:
            return True

        assert await _run(passes) == [Feedback(key="passes", score=True)]

    @pytest.mark.asyncio
    async def test_bool_false_becomes_fail_feedback(self) -> None:
        @scorer
        def fails() -> bool:
            return False

        assert await _run(fails) == [Feedback(key="fails", score=False)]

    @pytest.mark.asyncio
    async def test_int_becomes_numeric_score(self) -> None:
        @scorer
        def count() -> int:
            return 5

        assert await _run(count) == [Feedback(key="count", score=5)]

    @pytest.mark.asyncio
    async def test_float_becomes_numeric_score(self) -> None:
        @scorer
        def ratio() -> float:
            return 0.75

        assert await _run(ratio) == [Feedback(key="ratio", score=0.75)]

    @pytest.mark.asyncio
    async def test_str_becomes_categorical_value(self) -> None:
        @scorer
        def termination_reason() -> str:
            return "completed"

        assert await _run(termination_reason) == [Feedback(key="termination_reason", value="completed")]

    @pytest.mark.asyncio
    async def test_feedback_returned_directly(self) -> None:
        feedback = Feedback(key="custom_key", score=0.9, comment="great")

        @scorer
        def manual() -> Feedback:
            return feedback

        assert await _run(manual) == [feedback]

    @pytest.mark.asyncio
    async def test_list_of_feedback_emits_each(self) -> None:
        @scorer
        def many() -> list[Feedback]:
            return [
                Feedback(key="a", score=True),
                Feedback(key="b", value="completed"),
            ]

        assert await _run(many) == [
            Feedback(key="a", score=True),
            Feedback(key="b", value="completed"),
        ]

    @pytest.mark.asyncio
    async def test_none_returns_empty_list(self) -> None:
        @scorer
        def skip() -> None:
            return None

        assert await _run(skip) == []

    @pytest.mark.asyncio
    async def test_unsupported_return_raises(self) -> None:
        @scorer
        def bad() -> dict[str, int]:
            return {"score": 1}

        with pytest.raises(ScorerReturnTypeError, match="unsupported type dict"):
            await _run(bad)

    @pytest.mark.asyncio
    async def test_list_with_non_feedback_raises(self) -> None:
        @scorer
        def bad() -> list[Any]:
            return [Feedback(key="ok", score=True), "not a feedback"]

        with pytest.raises(ScorerReturnTypeError, match="non-Feedback at index 1"):
            await _run(bad)


class TestExceptionCapture:
    @pytest.mark.asyncio
    async def test_exception_becomes_no_score_feedback(self) -> None:
        @scorer
        def boom() -> bool:
            raise ValueError("kapow")

        [fb] = await _run(boom)

        assert fb == Feedback(
            key="boom",
            score=None,
            comment="scorer raised: ValueError: kapow",
        )

    @pytest.mark.asyncio
    async def test_exception_does_not_propagate(self) -> None:
        @scorer
        def boom() -> bool:
            raise RuntimeError("kaboom")

        # The act of awaiting must not raise.
        feedback = await _run(boom)
        assert len(feedback) == 1
        assert feedback[0].score is None

    @pytest.mark.asyncio
    async def test_async_exception_captured(self) -> None:
        @scorer
        async def aboom() -> bool:
            raise KeyError("missing")

        [fb] = await _run(aboom)

        assert fb.key == "aboom"
        assert fb.score is None
        assert fb.comment is not None
        assert "KeyError" in fb.comment


class TestAsyncScorers:
    @pytest.mark.asyncio
    async def test_async_scorer_awaited(self) -> None:
        @scorer
        async def async_pass(trace: Trace) -> bool:
            return len(trace.events) == 0

        assert await _run(async_pass) == [Feedback(key="async_pass", score=True)]


class TestScorerConstruction:
    @pytest.mark.asyncio
    async def test_explicit_key_overrides_function_name(self) -> None:
        def fn(trace: Trace) -> bool:
            return True

        s = Scorer(fn, key="custom-name")

        feedback = await _run(s)
        assert feedback == [Feedback(key="custom-name", score=True)]

    def test_key_defaults_to_function_name(self) -> None:
        @scorer
        def my_check(trace: Trace) -> bool:
            return True

        assert my_check.key == "my_check"
