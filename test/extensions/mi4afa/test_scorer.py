# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import threading
from typing import Any

import pytest

from ag2.eval import InMemoryTraceSource, Trace, TraceRef, evaluate_traces
from ag2.eval.dataset.task import Task
from ag2.eval.scorers import Attribution
from ag2.events import ModelMessage, ModelResponse, ToolCallEvent, ToolResultEvent
from ag2.extensions.mi4afa import (
    ActivationExtractor,
    Conversation,
    ProbeAttributor,
    probe_failure_attribution,
)

from .conftest import make_conversation


class _RecordingAttributor:
    """Blames a fixed turn and records what it was asked and on which thread."""

    def __init__(self, step: int) -> None:
        self.step = step
        self.seen: list[Conversation] = []
        self.thread_ids: list[int] = []

    def attribute(self, conversation: Conversation) -> Attribution:
        self.seen.append(conversation)
        self.thread_ids.append(threading.get_ident())
        return Attribution(
            failed=True,
            error_mode="other",
            decisive_step=self.step,
            responsible_agent=conversation.history[self.step].name,
            reasoning="fixed",
        )


def _trace() -> Trace:
    call = ToolCallEvent("search", arguments="{}")
    events = [
        ModelResponse(message=ModelMessage("plan")),
        call,
        ToolResultEvent.from_call(call, "wrong price"),
        ModelResponse(message=ModelMessage("answer $65")),
    ]
    return Trace(events=events, exception=None, duration_ms=0)


async def _score(scorer: Any, trace: Trace, reference_outputs: dict[str, Any] | None = None) -> list[Any]:
    return await scorer(
        inputs={"input": "How much did I save?"},
        outputs={},
        reference_outputs=reference_outputs,
        trace=trace,
        task=Task(task_id="t", inputs={}),
    )


@pytest.mark.asyncio()
async def test_feedback_names_the_agent_and_the_trace_event() -> None:
    attributor = _RecordingAttributor(step=2)
    scorer = probe_failure_attribution(attributor, key="blame", agent_name="planner")  # type: ignore[arg-type]

    [feedback] = await _score(scorer, _trace(), {"answer": "$55"})

    assert scorer.key == "blame"
    assert feedback.key == "blame"
    assert feedback.value == "tool:search"
    assert feedback.comment == "fixed"
    assert feedback.detail["decisive_step"] == 2  # index into trace.events, like the core scorer
    [conversation] = attributor.seen
    assert conversation.question == "How much did I save?"
    assert conversation.ground_truth == "$55"
    assert conversation.history[0].name == "planner"


@pytest.mark.asyncio()
async def test_decisive_step_is_mapped_through_skipped_events() -> None:
    trace = Trace(
        events=[ModelResponse(message=None), ModelResponse(message=ModelMessage("only turn"))],
        exception=None,
        duration_ms=0,
    )
    [feedback] = await _score(probe_failure_attribution(_RecordingAttributor(step=0)), trace)  # type: ignore[arg-type]
    assert feedback.detail["decisive_step"] == 1


@pytest.mark.asyncio()
async def test_attribution_runs_off_the_event_loop_thread() -> None:
    attributor = _RecordingAttributor(step=0)
    await _score(probe_failure_attribution(attributor), _trace())  # type: ignore[arg-type]
    assert attributor.thread_ids[0] != threading.get_ident()


@pytest.mark.asyncio()
async def test_trace_without_turns_yields_no_signal() -> None:
    attributor = _RecordingAttributor(step=0)
    [feedback] = await _score(
        probe_failure_attribution(attributor),  # type: ignore[arg-type]
        Trace(events=[], exception=None, duration_ms=0),
    )
    assert feedback.value is None
    assert feedback.score is None
    assert "no turns" in feedback.comment
    assert attributor.seen == []


@pytest.mark.parametrize(
    ("reference_outputs", "field", "expected"),
    [
        (None, None, ""),
        ({"answer": 55}, None, "55"),
        ({"answer": "$55", "unit": "USD"}, "answer", "$55"),
        ({"answer": "$55", "unit": "USD"}, "missing", ""),
        ({"answer": "$55", "unit": "USD"}, None, '{"answer": "$55", "unit": "USD"}'),
    ],
)
@pytest.mark.asyncio()
async def test_ground_truth_resolution(reference_outputs: Any, field: str | None, expected: str) -> None:
    attributor = _RecordingAttributor(step=0)
    await _score(probe_failure_attribution(attributor, ground_truth_field=field), _trace(), reference_outputs)  # type: ignore[arg-type]
    assert attributor.seen[0].ground_truth == expected


@pytest.mark.asyncio()
async def test_end_to_end_with_a_fitted_probe(model: Any, tokenizer: Any, tmp_path: Any) -> None:
    attributor = ProbeAttributor(ActivationExtractor(model, tokenizer), epochs=10)
    attributor.fit([make_conversation(3, mistake_step=index % 3, tag=str(index)) for index in range(6)])
    source = InMemoryTraceSource([(TraceRef("r1", task_id="r1"), _trace()), (TraceRef("r2", task_id="r2"), _trace())])

    result = await evaluate_traces(source, scorers=[probe_failure_attribution(attributor)], store_dir=tmp_path)

    counts = result.value_counts("failure_probe")
    assert sum(counts.values()) == 2
    assert set(counts) <= {"assistant", "tool:search"}
