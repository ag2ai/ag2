# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

from ag2.eval import Trace
from ag2.events import (
    ModelMessage,
    ModelResponse,
    TaskCompleted,
    TaskFailed,
    TaskStarted,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
    Usage,
)
from ag2.events.types import UsageEvent
from ag2.extensions.mi4afa import Turn, conversation_from_trace


def _trace(events: list) -> Trace:
    return Trace(events=events, exception=None, duration_ms=0)


def test_turns_and_event_indices() -> None:
    search = ToolCallEvent("search", arguments='{"q": "ticket price"}')
    flaky = ToolCallEvent("flaky", arguments="{}")
    events = [
        TaskStarted(task_id="t1", agent_name="researcher", objective="find prices"),  # 0 skipped
        ModelResponse(message=ModelMessage("I will search.")),  # 1
        search,  # 2
        ToolResultEvent.from_call(search, "daily ticket: $60"),  # 3
        flaky,  # 4
        ToolErrorEvent.from_call(flaky, RuntimeError("timeout")),  # 5
        UsageEvent(usage=Usage(prompt_tokens=1)),  # 6 skipped
        TaskCompleted(task_id="t1", agent_name="researcher", objective="find", result="$60", task_stream=uuid4()),  # 7
        TaskFailed(task_id="t2", agent_name="coder", objective="sum", error=ValueError("bad sum")),  # 8
        ModelResponse(message=None),  # 9 skipped: no text
        ModelResponse(message=ModelMessage("The answer is $65.")),  # 10
    ]

    converted = conversation_from_trace(_trace(events), question="Q", ground_truth="$55", agent_name="planner")

    assert converted is not None
    assert converted.event_indices == (1, 2, 3, 4, 5, 7, 8, 10)
    assert converted.conversation.question == "Q"
    assert converted.conversation.ground_truth == "$55"
    assert converted.conversation.mistake_step is None
    assert converted.conversation.history == (
        Turn("planner", "I will search."),
        Turn("planner", 'calls search({"q": "ticket price"})'),
        Turn("tool:search", "daily ticket: $60"),
        Turn("planner", "calls flaky({})"),
        Turn("tool:flaky", "error: timeout"),
        Turn("researcher", "$60"),
        Turn("coder", "failed: bad sum"),
        Turn("planner", "The answer is $65."),
    )


def test_trace_without_turns_gives_none() -> None:
    assert conversation_from_trace(_trace([]), question="Q", ground_truth="A") is None
    assert conversation_from_trace(_trace([ModelResponse(message=None)]), question="Q", ground_truth="A") is None


def test_default_agent_name() -> None:
    converted = conversation_from_trace(
        _trace([ModelResponse(message=ModelMessage("hi"))]), question="Q", ground_truth="A"
    )
    assert converted is not None
    assert converted.conversation.history == (Turn("assistant", "hi"),)
