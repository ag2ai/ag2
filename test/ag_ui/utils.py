# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any
from uuid import uuid4

from ag_ui.core import Message, RunAgentInput, Tool

from ag2.ag_ui import AGUIStream


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
