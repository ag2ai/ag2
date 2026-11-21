# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Project status into detached provider-neutral request views."""

from collections.abc import Iterable
from copy import copy, deepcopy
from typing import Literal

from ag2.events import (
    BaseEvent,
    Field,
    Input,
    ModelRequest,
    ModelResponse,
    StatusObservation,
    ToolCallsEvent,
    ToolResultsEvent,
)


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
    inputs: list[BaseEvent] = []
    response: ModelResponse | None = None
    response_index = 0
    pending: set[str] = set()
    for event in events:
        if isinstance(event, ModelResponse):
            if (
                event.response_force
                and event.message is None
                and event.tool_calls.calls
                and response is not None
                and all(call.id in pending for call in event.tool_calls.calls)
            ):
                # A client handoff repeats pending calls with their approved arguments.
                # Project them onto the original batch without changing stored history.
                approved = {call.id: call for call in event.tool_calls.calls}
                response = copy(response)
                response.tool_calls = ToolCallsEvent([
                    approved.get(call.id, call) for call in response.tool_calls.calls
                ])
                inputs[response_index] = response
                continue
            response = event
            response_index = len(inputs)
            pending = {call.id for call in event.tool_calls.calls}
        elif isinstance(event, ToolResultsEvent):
            pending.difference_update(result.parent_id for result in event.results)
        inputs.append(observation_request(event) if isinstance(event, StatusObservation) else event)
    return inputs
