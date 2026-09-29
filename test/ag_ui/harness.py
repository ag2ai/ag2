# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Driving a served agent over the generator seam, and reading what it emitted.

One run is one exchange here: `dispatch_run` builds the input, drives
`AGUIStream.dispatch`, and hands back the decoded AG-UI frames. A run that
pauses on a question spans two exchanges and cannot be expressed this way —
`test.ag_ui.serving` drives those over in-process HTTP instead.

Both seams speak the same vocabulary: a run is a `list[dict]` of decoded
frames, read with `types_of`, `only` and `every`.
"""

import json
from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from ag_ui.core import PROTOCOL_VERSION, Message, RunAgentInput, Tool
from dirty_equals import IsPartialDict

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, Usage
from ag2.testing import TestConfig
from ag2.tools import tool

__all__ = (
    "decode",
    "dispatch_run",
    "every",
    "exploding_agent",
    "frames_of_failing_run",
    "only",
    "outcome_of",
    "run_input",
    "sole_interrupt",
    "types_of",
    "weather_tool",
)


def run_input(
    *messages: Message,
    tools: list[Tool] | None = None,
    thread_id: str | None = None,
    state: Any = None,
    protocol_version: str | None = PROTOCOL_VERSION,
) -> RunAgentInput:
    """One `RunAgentInput`, with the ids a client would have generated.

    From a 1.0 client unless `protocol_version` says otherwise; `None` is a
    client predating 1.0, which declares nothing.
    """
    return RunAgentInput(
        thread_id=thread_id or str(uuid4()),
        run_id=str(uuid4()),
        protocol_version=protocol_version,
        messages=list(messages),
        state={} if state is None else state,
        context=[],
        tools=tools or [],
        forwarded_props=None,
    )


def decode(lines: Iterable[str]) -> list[dict[str, Any]]:
    """The AG-UI frames carried by encoded stream output."""
    frames = []
    for line in lines:
        payload = line.removeprefix("data: ").strip()
        if payload:
            frames.append(json.loads(payload))
    return frames


async def dispatch_run(stream: AGUIStream, incoming: RunAgentInput, **kwargs: Any) -> list[dict[str, Any]]:
    """Drive one exchange over the generator seam and decode its frames."""
    return [frame async for chunk in stream.dispatch(incoming, **kwargs) for frame in decode([chunk])]


def types_of(frames: list[dict[str, Any]]) -> list[str]:
    """Every frame's type, in the order they were emitted."""
    return [f["type"] for f in frames]


def only(frames: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    """The one frame of `event_type` — an assertion that there is exactly one."""
    [frame] = every(frames, event_type)
    return frame


def every(frames: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    """Every frame of `event_type`, in order. Empty when the run emitted none."""
    return [f for f in frames if f["type"] == event_type]


def outcome_of(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """The outcome the run finished with."""
    outcome: object = only(frames, "RUN_FINISHED")["outcome"]
    assert isinstance(outcome, dict)
    return outcome


def sole_interrupt(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """The one question the run stopped on."""
    interrupts: list[dict[str, Any]] = outcome_of(frames)["interrupts"]
    [interrupt] = interrupts
    return interrupt


def weather_tool() -> Tool:
    """A client-side tool declaration, the way a browser client sends one."""
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


def exploding_agent(usage: Usage | None = None) -> Agent:
    """An agent whose only tool always fails, optionally having spent `usage` first."""

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


async def frames_of_failing_run(agent: Agent, incoming: RunAgentInput) -> list[dict[str, Any]]:
    """The frames of a run expected to fail on `exploding_agent`'s own error.

    Narrowed to that failure: a run that died for some unrelated reason would
    otherwise still end on `RUN_ERROR`, and every caller would pass while
    asserting on a run that failed for a reason nobody wrote down.
    """
    frames = await dispatch_run(AGUIStream(agent), incoming)
    assert frames[-1] == IsPartialDict({"type": "RUN_ERROR", "message": "RuntimeError('downstream is down')"})
    return frames
