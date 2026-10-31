# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Project retained service observations into detached provider-neutral inputs."""

from collections.abc import Iterable
from copy import deepcopy
from typing import Literal

from ag2.events import BaseEvent, Field, ModelRequest, StatusObservation


class _StatusRequest(ModelRequest):
    """Request-only view; never changes a stored observation's classification."""

    __transient__ = True
    __conversational__ = False

    origin: Literal["user", "service"] = Field(default="service", compare=False)


def _status_input(event: StatusObservation) -> ModelRequest:
    return _StatusRequest(deepcopy(event.parts), origin="service", created_at=event.created_at)


def _model_inputs(events: Iterable[BaseEvent]) -> list[BaseEvent]:
    return [_status_input(event) if isinstance(event, StatusObservation) else event for event in events]
