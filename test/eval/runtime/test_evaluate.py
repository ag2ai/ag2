# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluate_traces() — grading traces from a TraceSource."""

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence

import pytest

from ag2.eval import (
    BudgetThresholds,
    InMemoryTraceSource,
    MissingTraceError,
    Suite,
    TraceRef,
    evaluate_traces,
    load_run,
    scorer,
)
from ag2.eval.scorers import final_answer_matches, tool_called
from ag2.eval.trace import Trace
from ag2.events import ModelMessage, ModelResponse, ToolCallEvent, Usage, UsageEvent


def _trace(answer: str, *, tool_name: str | None = None, in_tok: int = 0, out_tok: int = 0) -> Trace:
    events: list = []
    if tool_name is not None:
        events.append(ToolCallEvent(tool_name, arguments="{}"))
    usage = Usage(prompt_tokens=in_tok, completion_tokens=out_tok)
    events.append(ModelResponse(message=ModelMessage(answer), usage=usage))
    # A real run emits the accounting event alongside the response; token
    # counting reads that, so a fixture without it would score as free.
    if usage:
        events.append(UsageEvent(usage))
    return Trace(events=events, exception=None, duration_ms=10)


@scorer
def has_one_response(trace: Trace) -> bool:
    return len(trace.events_of(ModelResponse)) == 1


@scorer
def answer_is_paris(outputs: dict) -> bool:
    """True iff the parsed ``content`` carries answer == 'Paris'."""
    content = outputs.get("content")
    return isinstance(content, dict) and content.get("answer") == "Paris"


@scorer
def free_text_content_mirrors_body(outputs: dict) -> bool:
    """For a non-JSON answer, ``content`` is the text itself (mirrors reply.content())."""
    return isinstance(outputs.get("content"), str) and outputs["content"] == outputs.get("body")


class _FailOnLoadSource:
    """A :class:`TraceSource` whose ``load`` raises for one ref.

    Models a real backend (Tempo, a directory of span JSON) that can't
    materialize a particular trace — a network error, a corrupt file. This is
    the public path into ``_evaluate_ref``'s failure handling: a raising
    *scorer* would be swallowed into ``Feedback(score=None)`` by the Scorer
    wrapper, but a raising ``source.load`` propagates to the gather guard.
    """

    def __init__(self, traces: Sequence[tuple[TraceRef, Trace]], *, fail_trace_id: str) -> None:
        self._inner = InMemoryTraceSource(traces)
        self._fail_trace_id = fail_trace_id

    async def list(self) -> AsyncIterator[TraceRef]:
        async for ref in self._inner.list():
            yield ref

    async def load(self, ref: TraceRef) -> Trace:
        if ref.trace_id == self._fail_trace_id:
            raise RuntimeError("trace load exploded")
        return await self._inner.load(ref)


@pytest.mark.asyncio()
async def test_evaluate_scores_persists_and_joins_reference(tmp_path) -> None:
    source = InMemoryTraceSource([
        (TraceRef("t1", task_id="task-1"), _trace("Paris", tool_name="get_weather", in_tok=5, out_tok=2)),
    ])
    suite = Suite.from_list([
        {"task_id": "task-1", "inputs": {"input": "capital of France?"}, "reference_outputs": {"answer": "Paris"}},
    ])

    result = await evaluate_traces(
        source,
        scorers=[tool_called("get_weather"), final_answer_matches(field="answer", matcher="contains")],
        suite=suite,
        store_dir=tmp_path,
    )

    assert result.pass_rate("tool_called[get_weather]") == 1.0
    assert result.pass_rate("final_answer_matches") == 1.0  # reference joined via task_id
    assert result.aggregates.tokens.total == 7
    assert (tmp_path / f"{result.run_id}.json").exists()


@pytest.mark.asyncio()
async def test_evaluate_reference_free_without_suite(tmp_path) -> None:
    source = InMemoryTraceSource([(TraceRef("only"), _trace("hello"))])

    result = await evaluate_traces(source, scorers=[has_one_response], store_dir=tmp_path)

    assert result.pass_rate("has_one_response") == 1.0
    assert len(result.tasks) == 1
    assert result.tasks[0].task.task_id == "only"


@pytest.mark.asyncio()
async def test_evaluate_records_budget_violation(tmp_path) -> None:
    source = InMemoryTraceSource([(TraceRef("big"), _trace("x", in_tok=100, out_tok=100))])

    result = await evaluate_traces(
        source, scorers=[], store_dir=tmp_path, budgets=BudgetThresholds(max_tokens_per_task=50)
    )

    assert result.aggregates.budget_violations == 1


@pytest.mark.asyncio()
async def test_budget_violation_fires_on_accounting_only_spend(tmp_path) -> None:
    """Delegated spend trips the budget even though no response carries it.

    A sub-task's tokens reach the parent as a ``"subtask"`` rollup and nowhere
    else — the parent's own response is cheap. A budget check still reading
    model responses would call this task free and clear the violation flag,
    which is the failure the 0.2 accounting exists to close.
    """
    delegated = Trace(
        events=[
            ModelResponse(message=ModelMessage("done"), usage=Usage(prompt_tokens=1, completion_tokens=1)),
            UsageEvent(Usage(prompt_tokens=1, completion_tokens=1)),
            UsageEvent(Usage(prompt_tokens=100, completion_tokens=100), kind="subtask", label="worker"),
        ],
        exception=None,
        duration_ms=10,
    )
    source = InMemoryTraceSource([(TraceRef("delegating"), delegated)])

    result = await evaluate_traces(
        source, scorers=[], store_dir=tmp_path, budgets=BudgetThresholds(max_tokens_per_task=50)
    )

    assert result.aggregates.budget_violations == 1


@pytest.mark.asyncio()
async def test_json_object_answer_projects_structured_content(tmp_path) -> None:
    """A JSON-object final answer is parsed into outputs["content"] (mirrors reply.content())."""
    source = InMemoryTraceSource([
        (TraceRef("t1", task_id="task-1"), _trace('{"answer": "Paris", "confidence": 0.9}')),
    ])
    suite = Suite.from_list([
        {"task_id": "task-1", "inputs": {"input": "capital of France?"}, "reference_outputs": {"answer": "Paris"}},
    ])

    result = await evaluate_traces(
        source,
        # exact match only passes if "Paris" came from the parsed content, not the raw JSON text
        scorers=[answer_is_paris, final_answer_matches(field="answer", matcher="exact")],
        suite=suite,
        store_dir=tmp_path,
    )

    assert result.pass_rate("answer_is_paris") == 1.0
    assert result.pass_rate("final_answer_matches") == 1.0


@pytest.mark.asyncio()
async def test_free_text_answer_content_mirrors_body(tmp_path) -> None:
    """A non-JSON answer leaves content as the text itself (== body)."""
    source = InMemoryTraceSource([(TraceRef("t1", task_id="task-1"), _trace("Paris is the capital."))])

    result = await evaluate_traces(source, scorers=[free_text_content_mirrors_body], store_dir=tmp_path)

    assert result.pass_rate("free_text_content_mirrors_body") == 1.0


@pytest.mark.asyncio()
async def test_evaluate_does_not_crash_whole_run_on_one_ref_failure(tmp_path) -> None:
    """If loading one ref raises, the run still returns with the surviving ref's result present."""
    source = _FailOnLoadSource(
        [
            (TraceRef("t1", task_id="task-1"), _trace("Paris")),
            (TraceRef("t2", task_id="task-2"), _trace("London")),
        ],
        fail_trace_id="t1",
    )
    suite = Suite.from_list([
        {"task_id": "task-1", "inputs": {"input": "capital of France?"}},
        {"task_id": "task-2", "inputs": {"input": "capital of UK?"}},
    ])

    result = await evaluate_traces(source, scorers=[has_one_response], suite=suite, store_dir=tmp_path)

    # Run must not raise — it returns a RunResult with both entries
    assert len(result.tasks) == 2
    # The failed ref has its exception captured on the trace
    failed = next(tr for tr in result.tasks if tr.task.task_id == "task-1")
    assert isinstance(failed.trace.exception, RuntimeError)
    # The surviving ref scored correctly
    surviving = next(tr for tr in result.tasks if tr.task.task_id == "task-2")
    assert surviving.feedback[0].score is True


@pytest.mark.asyncio()
async def test_evaluate_propagates_cancellation(tmp_path) -> None:
    """A CancelledError from scoring must propagate out (not become a partial RunResult).

    CancelledError is a BaseException, so the Scorer wrapper's ``except Exception``
    does not swallow it — it reaches the gather guard, which re-raises it.
    """

    @scorer
    def cancels() -> bool:
        raise asyncio.CancelledError

    source = InMemoryTraceSource([(TraceRef("t1", task_id="task-1"), _trace("Paris"))])
    suite = Suite.from_list([{"task_id": "task-1", "inputs": {"input": "capital of France?"}}])

    with pytest.raises(asyncio.CancelledError):
        await evaluate_traces(source, scorers=[cancels], suite=suite, store_dir=tmp_path)


def _two_of_three() -> tuple[InMemoryTraceSource, Suite]:
    """A 3-task suite whose source only covers ``task-1`` and ``task-2``."""
    source = InMemoryTraceSource([
        (TraceRef("t1", task_id="task-1"), _trace("Paris")),
        (TraceRef("t2", task_id="task-2"), _trace("London")),
    ])
    suite = Suite.from_list([
        {"task_id": "task-1", "inputs": {"input": "capital of France?"}},
        {"task_id": "task-2", "inputs": {"input": "capital of UK?"}},
        {"task_id": "task-3", "inputs": {"input": "capital of Japan?"}},
    ])
    return source, suite


@pytest.mark.asyncio()
class TestOnMissingTask:
    """A suite task the source has no trace for — the CI-gate failure mode."""

    async def test_default_counts_the_uncovered_task_as_an_error(self, tmp_path) -> None:
        source, suite = _two_of_three()

        result = await evaluate_traces(source, scorers=[has_one_response], suite=suite, store_dir=tmp_path)

        assert [tr.task.task_id for tr in result.tasks] == ["task-1", "task-2", "task-3"]
        assert result.aggregates.errors == 1
        # denominator is 3, not 2: the uncovered task scores as a failure
        assert result.pass_rate("has_one_response") == pytest.approx(2 / 3)
        missing = result.tasks[-1]
        assert isinstance(missing.trace.exception, MissingTraceError)
        assert missing.trace.exception.task_ids == ("task-3",)
        assert "task-3" in str(missing.trace.exception)

    async def test_ignore_drops_the_uncovered_task_and_warns(self, tmp_path, caplog) -> None:
        source, suite = _two_of_three()

        with caplog.at_level(logging.WARNING, logger="ag2.eval.runtime.evaluate"):
            result = await evaluate_traces(
                source, scorers=[has_one_response], suite=suite, store_dir=tmp_path, on_missing_task="ignore"
            )

        assert [tr.task.task_id for tr in result.tasks] == ["task-1", "task-2"]
        assert result.aggregates.errors == 0
        assert result.pass_rate("has_one_response") == 1.0
        [record] = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert "1 suite task(s) have no trace" in record.getMessage()
        assert "task-3" in record.getMessage()

    async def test_raise_aborts_before_anything_is_graded(self, tmp_path) -> None:
        source, suite = _two_of_three()
        graded: list[str] = []

        @scorer
        def record(task) -> bool:
            graded.append(task.task_id)
            return True

        with pytest.raises(MissingTraceError, match="task-3"):
            await evaluate_traces(source, scorers=[record], suite=suite, store_dir=tmp_path, on_missing_task="raise")

        assert graded == []

    async def test_ref_outside_the_suite_is_still_graded_reference_free(self, tmp_path) -> None:
        """The tolerated asymmetry: an extra trace is graded, never dropped."""
        source = InMemoryTraceSource([
            (TraceRef("t1", task_id="task-1"), _trace("Paris")),
            (TraceRef("t9", task_id="not-in-suite"), _trace("Tokyo")),
        ])
        suite = Suite.from_list([{"task_id": "task-1", "inputs": {"input": "capital of France?"}}])

        result = await evaluate_traces(source, scorers=[has_one_response], suite=suite, store_dir=tmp_path)

        assert [tr.task.task_id for tr in result.tasks] == ["task-1", "not-in-suite"]
        assert result.aggregates.errors == 0
        extra = result.tasks[-1]
        assert extra.task.reference_outputs is None
        assert extra.feedback[0].score is True

    async def test_missing_task_error_survives_save_and_load(self, tmp_path) -> None:
        source, suite = _two_of_three()

        result = await evaluate_traces(source, scorers=[has_one_response], suite=suite, store_dir=tmp_path)
        loaded = load_run(result.save())

        assert loaded.aggregates.errors == 1
        assert loaded.pass_rate("has_one_response") == pytest.approx(2 / 3)
        # store.py keeps only the exception's type + message
        restored = loaded.tasks[-1].trace.exception
        assert restored is not None
        assert str(restored) == "MissingTraceError: no trace in the source for task(s): task-3"
