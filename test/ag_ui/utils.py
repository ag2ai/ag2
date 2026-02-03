# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any
from uuid import uuid4

from ag_ui.core import Message, RunAgentInput, Tool

from autogen.ag_ui import AGUIStream


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


async def collect_events(stream: AGUIStream, run_input: RunAgentInput) -> list[dict[str, Any]]:
    events = []
    async for event in stream.dispatch(run_input):
        event_str = event.removeprefix("data: ").strip()
        if event_str:
            events.append(json.loads(event_str))
    return events


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
