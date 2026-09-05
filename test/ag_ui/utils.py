# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any
from uuid import uuid4

import pytest
from ag_ui.core import Message, RunAgentInput, Tool

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, Usage
from ag2.testing import TestConfig
from ag2.tools import tool


def uuid_str() -> str:
    return str(uuid4())


def create_run_input(
    *messages: Message,
    tools: list[Tool] | None = None,
    thread_id: str | None = None,
    state: Any = None,
) -> RunAgentInput:
    thread_id = thread_id or uuid_str()
    return RunAgentInput(
        thread_id=thread_id,
        run_id=uuid_str(),
        messages=list(messages),
        state=dict(state) if state else {},
        context=[],
        tools=tools or [],
        forwarded_props=None,
    )


def get_weather_tool() -> Tool:
    return Tool(
        name="get_weather",
        description="Get the weather for a given location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get the weather for",
                },
            },
            "required": ["location"],
        },
    )


async def collect_events(
    stream: AGUIStream,
    run_input: RunAgentInput,
    *,
    into: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Decode the events ``dispatch`` yields for one run.

    Pass ``into`` when the run is expected to fail: a failing run emits ``RUN_ERROR``
    and then re-raises, so the return value never arrives. Events are appended to
    ``into`` as they are decoded, leaving them available to assert on after the
    exception has been caught.
    """
    events = into if into is not None else []
    async for event in stream.dispatch(run_input, **kwargs):
        event_str = event.removeprefix("data: ").strip()
        if event_str:
            events.append(json.loads(event_str))
    return events


async def frames_of_failing_run(agent: Agent, run_input: RunAgentInput) -> list[dict[str, Any]]:
    """The frames a run expected to fail emits before ``dispatch`` re-raises.

    The re-raise is swallowed here because these are the callers asserting on the events;
    the ones asserting on the exception itself use ``pytest.raises`` directly so they can
    reach it through ``leaf_exceptions``.

    Swallowing is narrowed to the failure these callers stage — ``exploding_agent``'s
    ``RuntimeError``. A run that died for some unrelated reason would otherwise still
    hand back frames, and every caller would still pass while asserting on a run that
    failed for a reason nobody wrote down.
    """
    frames: list[dict[str, Any]] = []
    with pytest.raises(Exception) as exc_info:
        await collect_events(AGUIStream(agent), run_input, into=frames)
    assert [type(e) for e in leaf_exceptions(exc_info.value)] == [RuntimeError]
    return frames


def exploding_agent(usage: Usage | None = None) -> Agent:
    """An agent whose only tool always fails, optionally having spent ``usage`` first."""

    @tool
    def explode() -> str:
        """A downstream call that always fails."""
        raise RuntimeError("downstream is down")

    calls = ToolCallsEvent(calls=[ToolCallEvent(name="explode", arguments="{}")])
    response = (
        ModelResponse(tool_calls=calls, usage=usage, model="claude-sonnet-4", provider="anthropic")
        if usage
        else ModelResponse(tool_calls=calls)
    )
    return Agent("test_agent", config=TestConfig(response), tools=[explode])


def leaf_exceptions(exc: BaseException) -> list[BaseException]:
    """Flatten anyio's exception groups down to the errors that actually happened.

    ``dispatch`` runs the agent in a task group, so a failure surfaces wrapped in an
    exception group. The nesting is unwrapped by duck-typing ``exceptions`` rather
    than naming the group class, which is a builtin only from Python 3.11.
    """
    nested = getattr(exc, "exceptions", None)
    if nested is None:
        return [exc]
    return [leaf for inner in nested for leaf in leaf_exceptions(inner)]


def assert_event_type(events: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    for event in events:
        if event.get("type") == event_type:
            return event
    raise AssertionError(f"Event of type {event_type} not found in events: {events}")


def assert_no_event_type(events: list[dict[str, Any]], event_type: str) -> None:
    for event in events:
        if event.get("type") == event_type:
            raise AssertionError(f"Unexpected event of type {event_type} found in events: {events}")


def get_events_of_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [e for e in events if e.get("type") == event_type]
