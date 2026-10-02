# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Driving a served agent over the generator seam, and reading what it emitted.

One run is one exchange here: `dispatch_run` builds the input, drives
`AGUIStream.dispatch`, and hands back the decoded AG-UI frames. A run that
pauses on a question spans two exchanges and cannot be expressed this way —
`test.ag_ui.serving` drives those over in-process HTTP instead.

A run is read as typed AG-UI events (`dispatch_events`, then `sole`, `each`, `kinds_of`,
and `wire` when what is asserted is how an event looks on the wire). The served-endpoint
tests and the A2UI transport tests still read decoded frames as `list[dict]` with
`dispatch_run`, `types_of`, `only` and `every`.
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar
from uuid import uuid4

from ag_ui.core import PROTOCOL_VERSION, Event, Message, RunAgentInput, Tool
from dirty_equals import IsPartialDict
from pydantic import TypeAdapter

from ag2 import Agent, Context
from ag2.ag_ui import AGUIStream
from ag2.events import BaseEvent, ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent, Usage
from ag2.middleware import BaseMiddleware, LLMCall, Middleware
from ag2.testing import TestConfig
from ag2.tools import tool

__all__ = (
    "ModelCall",
    "ModelHistory",
    "decode",
    "decode_events",
    "dispatch_events",
    "dispatch_run",
    "each",
    "events_of_failing_run",
    "every",
    "exploding_agent",
    "kinds_of",
    "only",
    "outcome_of",
    "recording_history",
    "run_input",
    "sole",
    "sole_interrupt",
    "types_of",
    "weather_tool",
    "wire",
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


_EVENT: TypeAdapter[Event] = TypeAdapter(Event)
_E = TypeVar("_E", bound=Event)


def decode_events(lines: Iterable[str]) -> list[Event]:
    """The typed AG-UI events carried by encoded stream output."""
    return [_EVENT.validate_python(frame) for frame in decode(lines)]


async def dispatch_events(stream: AGUIStream, incoming: RunAgentInput, **kwargs: Any) -> list[Event]:
    """Drive one exchange and return what the client receives, as typed AG-UI events."""
    return [event async for chunk in stream.dispatch(incoming, **kwargs) for event in decode_events([chunk])]


def wire(event: Event) -> dict[str, Any]:
    """What `event` looks like on the wire, without its timestamp (every event is stamped, and only `test_dispatch` says so)."""
    return event.model_dump(mode="json", by_alias=True, exclude_none=True, exclude={"timestamp"})


def kinds_of(events: Sequence[Event]) -> list[type[Event]]:
    """Every event's class, in the order they were emitted."""
    return [type(event) for event in events]


def each(events: Sequence[Event], kind: type[_E]) -> list[_E]:
    """Every event of `kind`, in order. Empty when the run emitted none."""
    return [event for event in events if isinstance(event, kind)]


def sole(events: Sequence[Event], kind: type[_E]) -> _E:
    """The one event of `kind` — an assertion that there is exactly one."""
    [event] = each(events, kind)
    return event


@dataclass(frozen=True)
class ModelCall:
    """One call the model was given: the prompt it ran under and the history it was handed."""

    prompt: list[str]
    events: list[BaseEvent]


class ModelHistory(BaseMiddleware):
    """Records every model call; with `reply` set, answers it instead of reaching the provider."""

    def __init__(self, event: BaseEvent, context: Context, *, calls: list[ModelCall], reply: str | None) -> None:
        super().__init__(event, context)
        self.calls = calls
        self.reply = reply

    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        self.calls.append(ModelCall(prompt=list(context.prompt), events=list(events)))
        if self.reply is not None:
            return ModelResponse(ModelMessage(self.reply))
        return await call_next(events, context)


def recording_history(*, reply: str | None = None) -> tuple[Middleware, list[ModelCall]]:
    """A middleware for `dispatch_*(..., middleware=[...])` and the list it fills, one entry per model call.

    Give `reply` to keep the run off the network when it is served with a real provider config.
    """
    calls: list[ModelCall] = []
    return Middleware(ModelHistory, calls=calls, reply=reply), calls


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


async def events_of_failing_run(agent: Agent, incoming: RunAgentInput) -> list[Event]:
    """The events of a run expected to fail on `exploding_agent`'s own error.

    Narrowed to that failure: a run that died for some unrelated reason would
    otherwise still end on `RUN_ERROR`, and every caller would pass while
    asserting on a run that failed for a reason nobody wrote down.
    """
    events = await dispatch_events(AGUIStream(agent), incoming)
    assert wire(events[-1]) == IsPartialDict({"type": "RUN_ERROR", "message": "RuntimeError('downstream is down')"})
    return events
