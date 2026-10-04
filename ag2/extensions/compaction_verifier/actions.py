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

Call ids are not assumed unique across a history: some providers number calls
per response (AG2's Ollama client names the first call of every response
``call_0``). A call is identified by its position in the sequence, and each
result is paired with the call it answers by scanning in order.
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
            Diagnostic only: ids need not be unique across a history.
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
    off mid-execution) has not been executed and is left out. See
    :func:`actions_with_positions` for how calls and results are paired.

    Raises:
        ValueError: Two different calls awaiting results share an id, so a
            result cannot be paired with a single call.
    """
    return [action for action, _ in actions_with_positions(events)]


def actions_with_positions(events: Iterable[BaseEvent]) -> list[tuple[Action, int]]:
    """Every executed tool call, with the index of the event that answered it.

    Calls and results are paired by scanning ``events`` in order, never through
    a global id map, because ids repeat across responses with some providers:

    * a call is read from the individual events and from their containers
      (:class:`~ag2.events.ToolCallsEvent`, a response's ``tool_calls``); a call
      seen again while an identical call (same id, tool and arguments) is still
      awaiting its result is that call repeated by another container, and counts
      once — otherwise it is a new call, even if an earlier, answered call used
      the same id;
    * a result answers the call with its id that is awaiting a result; a result
      with no such call is a container repeating one already paired, and is
      skipped.

    Raises:
        ValueError: Two different calls awaiting results share an id — within
            one event, or across events before the first was answered — so a
            result could not be paired with a single call.
    """
    issued: list[ToolCallEvent] = []
    awaiting: dict[str, int] = {}  # call id -> index in ``issued`` of the call awaiting its result
    answered: dict[int, tuple[ToolResultEvent, int]] = {}
    for index, event in enumerate(events):
        calls = calls_in(event)
        if len({c.id for c in calls}) < len(calls):
            raise ValueError(
                f"{type(event).__name__} at {index} holds several calls with one id; results cannot be paired"
            )
        for call in calls:
            open_index = awaiting.get(call.id)
            if open_index is None:
                awaiting[call.id] = len(issued)
                issued.append(call)
            elif not _same_call(issued[open_index], call):
                raise ValueError(
                    f"call id {call.id!r} at {index} is reused while an earlier call with that id is "
                    "still awaiting its result; results cannot be paired"
                )
        for result in results_in(event):
            open_index = awaiting.pop(result.parent_id, None)
            if open_index is not None:
                answered[open_index] = (result, index)

    actions: list[tuple[Action, int]] = []
    for position, call in enumerate(issued):
        if position not in answered:
            continue
        answer, at = answered[position]
        actions.append((
            Action(
                call_id=call.id,
                name=call.name,
                arguments=call.arguments,
                signature=call_signature(call.name, call.arguments),
                blocked=isinstance(answer, ToolErrorEvent),
                error_type=type(answer.error).__name__ if isinstance(answer, ToolErrorEvent) else None,
            ),
            at,
        ))
    return actions


def _same_call(a: ToolCallEvent, b: ToolCallEvent) -> bool:
    return call_signature(a.name, a.arguments) == call_signature(b.name, b.arguments)


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
