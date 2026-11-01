# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Project status into detached provider-neutral request views."""

from collections.abc import Iterable
from copy import deepcopy
from typing import Literal

from ag2.events import BaseEvent, Field, Input, ModelRequest, StatusObservation


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
    return [observation_request(event) if isinstance(event, StatusObservation) else event for event in events]
