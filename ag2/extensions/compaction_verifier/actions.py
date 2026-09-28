# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tool actions read off an AG2 event sequence.

An action is one executed tool call: the :class:`~ag2.events.ToolCallEvent` the
model issued and the result event the executor answered it with. The verifier
reads two things from each action:

* whether it was **blocked** — AG2 answers every failed execution with a
  :class:`~ag2.events.ToolErrorEvent` (:class:`~ag2.events.ToolNotFoundEvent`
  for a tool that does not exist), so this is the framework's own structured
  error contract, not a reading of the result text;
* its **signature** — the tool name plus its arguments as canonical JSON, used
  to recognise a call that repeats one already executed.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass

from ag2.events import (
    BaseEvent,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolResultEvent,
    ToolResultsEvent,
)

__all__ = ("Action", "actions_from_events", "call_signature")


@dataclass(frozen=True, slots=True)
class Action:
    """One executed tool call.

    Attributes:
        call_id: Id of the originating :class:`~ag2.events.ToolCallEvent`.
        name: Tool name as the model called it.
        arguments: Arguments exactly as the model sent them (a JSON string).
        signature: ``name(arguments)`` with the arguments as canonical JSON —
            see :func:`call_signature`.
        blocked: Whether the execution failed.
        error_type: Exception class name of a blocked action. Diagnostic only;
            it never enters a score.
    """

    call_id: str
    name: str
    arguments: str
    signature: str
    blocked: bool
    error_type: str | None = None


def call_signature(name: str, arguments: str) -> str:
    """``name(arguments)`` with the arguments as canonical JSON.

    Key order and whitespace do not change the signature, so ``{"b": 2, "a": 1}``
    and ``{"a":1,"b":2}`` agree. Values are kept verbatim: two calls are the same
    only if they would do the same thing. Arguments that are not valid JSON are
    kept as the stripped raw string, so a malformed call still has a stable
    identity.
    """
    raw = arguments or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return f"{name}({raw.strip()})"
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{name}({canonical})"


def actions_from_events(events: Iterable[BaseEvent]) -> list[Action]:
    """Every executed tool call in ``events``, in the order the calls were issued.

    A call counts once it has a result; a call still waiting for one (a run cut
    off mid-execution) has not been executed and is left out. Calls and results
    are matched by id, and read both from the individual events and from their
    containers (:class:`~ag2.events.ToolCallsEvent`,
    :class:`~ag2.events.ToolResultsEvent`, a response's ``tool_calls``), so each
    is counted once however the sequence was recorded.
    """
    calls: dict[str, ToolCallEvent] = {}
    results: dict[str, ToolResultEvent] = {}
    for event in events:
        for call in calls_in(event):
            calls.setdefault(call.id, call)
        for result in results_in(event):
            results.setdefault(result.parent_id, result)

    actions: list[Action] = []
    for call_id, call in calls.items():
        answer = results.get(call_id)
        if answer is None:
            continue
        actions.append(
            Action(
                call_id=call_id,
                name=call.name,
                arguments=call.arguments,
                signature=call_signature(call.name, call.arguments),
                blocked=isinstance(answer, ToolErrorEvent),
                error_type=type(answer.error).__name__ if isinstance(answer, ToolErrorEvent) else None,
            )
        )
    return actions


def calls_in(event: BaseEvent) -> tuple[ToolCallEvent, ...]:
    if isinstance(event, ToolCallEvent):
        return (event,)
    if isinstance(event, ToolCallsEvent):
        return tuple(event.calls)
    if isinstance(event, ModelResponse):
        return tuple(event.tool_calls.calls)
    return ()


def results_in(event: BaseEvent) -> tuple[ToolResultEvent, ...]:
    if isinstance(event, ToolResultEvent):
        return (event,)
    if isinstance(event, ToolResultsEvent):
        return tuple(event.results)
    return ()
