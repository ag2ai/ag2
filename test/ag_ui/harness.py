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

import pytest
from ag_ui.core import Message, RunAgentInput, Tool

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
    "leaf_exceptions",
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
) -> RunAgentInput:
    """One `RunAgentInput`, with the ids a client would have generated."""
    return RunAgentInput(
        thread_id=thread_id or str(uuid4()),
        run_id=str(uuid4()),
        messages=list(messages),
        state=dict(state) if state else {},
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


async def dispatch_run(
    stream: AGUIStream,
    incoming: RunAgentInput,
    *,
    into: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Drive one exchange over the generator seam and decode its frames.

    Pass `into` when the run is expected to fail: a failing run emits
    `RUN_ERROR` and then re-raises, so the return value never arrives. Frames
    are appended to `into` as they are decoded, leaving them available to
    assert on after the exception has been caught.
    """
    frames = into if into is not None else []
    async for chunk in stream.dispatch(incoming, **kwargs):
        frames.extend(decode([chunk]))
    return frames


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
    """The frames a run expected to fail emits before `dispatch` re-raises.

    The re-raise is swallowed here because these are the callers asserting on the
    frames; the ones asserting on the exception itself use `pytest.raises` directly
    so they can reach it through `leaf_exceptions`.

    Swallowing is narrowed to the failure these callers stage — `exploding_agent`'s
    `RuntimeError`. A run that died for some unrelated reason would otherwise still
    hand back frames, and every caller would still pass while asserting on a run
    that failed for a reason nobody wrote down.
    """
    frames: list[dict[str, Any]] = []
    with pytest.raises(Exception) as exc_info:
        await dispatch_run(AGUIStream(agent), incoming, into=frames)
    assert [type(e) for e in leaf_exceptions(exc_info.value)] == [RuntimeError]
    return frames


def leaf_exceptions(exc: BaseException) -> list[BaseException]:
    """Flatten anyio's exception groups down to the errors that actually happened.

    `dispatch` runs the agent in a task group, so a failure surfaces wrapped in an
    exception group. The nesting is unwrapped by duck-typing `exceptions` rather
    than naming the group class, which is a builtin only from Python 3.11.
    """
    nested = getattr(exc, "exceptions", None)
    if nested is None:
        return [exc]
    return [leaf for inner in nested for leaf in leaf_exceptions(inner)]
