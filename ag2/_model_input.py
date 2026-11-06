# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Project status into detached provider-neutral request views."""

from collections.abc import Iterable
from copy import copy, deepcopy
from typing import Literal

from ag2.events import BaseEvent, Field, Input, ModelRequest, ModelResponse, StatusObservation, ToolResultsEvent


class _StatusRequest(ModelRequest):
    """Request-only view; never changes a stored observation's classification.

    Non-conversational, so history reducers do not mistake it for a turn that
    can stand on its own (see ``ag2._replay.replayable_span``).
    """

    __transient__ = True
    __conversational__ = False

    origin: Literal["user", "service"] = Field(default="service", compare=False)


def status_request(parts: list[Input]) -> ModelRequest:
    """Wrap already-copied status parts as a service request view."""
    return _StatusRequest(parts, origin="service")


def observation_request(event: StatusObservation) -> ModelRequest:
    """Detached request view of a retained observation."""
    return _StatusRequest(deepcopy(event.parts), origin="service", created_at=event.created_at)


def model_inputs(events: Iterable[BaseEvent]) -> list[BaseEvent]:
    result: list[BaseEvent] = []
    response_index: int | None = None
    pending: set[str] = set()
    for event in events:
        if isinstance(event, ModelResponse):
            if (
                event.response_force
                and event.message is None
                and event.tool_calls
                and response_index is not None
                and all(call.id in pending for call in event.tool_calls.calls)
            ):
                # A client handoff announces pending calls, not another model
                # turn. Replay the approved calls on the original response,
                # leaving the stored/wire events untouched.
                original = result[response_index]
                assert isinstance(original, ModelResponse)
                projected = copy(original)
                projected.tool_calls = copy(original.tool_calls)
                approved = {call.id: call for call in event.tool_calls.calls}
                projected.tool_calls.calls = [approved.get(call.id, call) for call in original.tool_calls.calls]
                result[response_index] = projected
                continue
            response_index = len(result)
            pending = {call.id for call in event.tool_calls.calls}
        elif isinstance(event, ModelRequest):
            response_index = None
            pending.clear()
        elif isinstance(event, ToolResultsEvent):
            pending.difference_update(result.parent_id for result in event.results)
        result.append(observation_request(event) if isinstance(event, StatusObservation) else event)
    return result
