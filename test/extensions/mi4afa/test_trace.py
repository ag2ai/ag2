# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest

from ag2 import Agent, tool
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
from ag2.extensions.mi4afa import Turn, conversation_from_trace, subagent_names
from ag2.testing import TestConfig
from ag2.tools.final import Toolkit
from ag2.tools.subagents import background_agent_tool, subagent_tool


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


def test_subagents_name_delegated_results_and_errors() -> None:
    delegate = ToolCallEvent("task_researcher", arguments='{"objective": "find prices"}')
    retry = ToolCallEvent("task_researcher", arguments='{"objective": "find prices again"}')
    search = ToolCallEvent("search", arguments="{}")
    events = [
        delegate,
        ToolResultEvent.from_call(delegate, "$60"),
        retry,
        ToolErrorEvent.from_call(retry, RuntimeError("timeout")),
        search,
        ToolResultEvent.from_call(search, "no results"),
    ]

    converted = conversation_from_trace(
        _trace(events),
        question="Q",
        ground_truth="A",
        agent_name="planner",
        subagents={"task_researcher": "researcher"},
    )

    assert converted is not None
    assert converted.conversation.history == (
        Turn("planner", 'calls task_researcher({"objective": "find prices"})'),
        Turn("researcher", "$60"),
        Turn("planner", 'calls task_researcher({"objective": "find prices again"})'),
        Turn("researcher", "error: timeout"),
        Turn("planner", "calls search({})"),
        Turn("tool:search", "no results"),
    )


@pytest.mark.asyncio()
async def test_run_agent_delegation_is_named_by_subagent_names() -> None:
    pytest.importorskip("opentelemetry.sdk")
    from ag2.eval import run_agent

    calculator = Agent("Calculator", config=TestConfig(ModelResponse(ModelMessage("2 + 2 = 5"))))
    planner = Agent("Planner", tools=[calculator.as_tool(description="Do arithmetic.")])
    replies = TestConfig(
        ToolCallEvent(name="task_Calculator", arguments='{"objective": "add 2 and 2"}'),
        ModelResponse(ModelMessage("The answer is 5.")),
    )

    result = await run_agent("What is 2 + 2?", agent=planner, model_config=replies)
    [task] = result.tasks

    unnamed = conversation_from_trace(task.trace, question="Q", ground_truth="4", agent_name="Planner")
    named = conversation_from_trace(
        task.trace, question="Q", ground_truth="4", agent_name="Planner", subagents=subagent_names(planner)
    )

    assert unnamed is not None
    assert [turn.name for turn in unnamed.conversation.history] == ["Planner", "tool:task_Calculator", "Planner"]
    assert named is not None
    assert named.conversation.history == (
        Turn("Planner", 'calls task_Calculator({"objective": "add 2 and 2"})'),
        Turn("Calculator", "2 + 2 = 5"),
        Turn("Planner", "The answer is 5."),
    )


def test_subagent_names_reads_delegation_tools() -> None:
    @tool
    def search(query: str) -> str:
        return query

    coder = Agent("coder")
    researcher = Agent("researcher")
    writer = Agent("writer")
    planner = Agent(
        "planner",
        tools=[
            coder.as_tool(description="Write code."),
            subagent_tool(researcher, description="Research.", name="ask_research"),
            Toolkit(writer.as_tool(description="Write.")),
            background_agent_tool(coder, description="Code in the background."),
            search,
        ],
    )

    assert subagent_names(planner) == {"task_coder": "coder", "ask_research": "researcher", "task_writer": "writer"}
    assert subagent_names(coder) == {}


def test_trace_without_turns_gives_none() -> None:
    assert conversation_from_trace(_trace([]), question="Q", ground_truth="A") is None
    assert conversation_from_trace(_trace([ModelResponse(message=None)]), question="Q", ground_truth="A") is None


def test_default_agent_name() -> None:
    converted = conversation_from_trace(
        _trace([ModelResponse(message=ModelMessage("hi"))]), question="Q", ground_truth="A"
    )
    assert converted is not None
    assert converted.conversation.history == (Turn("assistant", "hi"),)
