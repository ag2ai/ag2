# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .events import Input, ModelRequest

PendingMessagePriority = Literal["asap", "when_idle"]


@dataclass(slots=True)
class PendingMessage:
    request: ModelRequest
    priority: PendingMessagePriority

    @classmethod
    def from_content(
        cls,
        *content: str | Input | ModelRequest,
        priority: PendingMessagePriority = "asap",
    ) -> "PendingMessage | None":
        if not content:
            return None

        if priority not in ("asap", "when_idle"):
            raise ValueError("Pending message priority must be 'asap' or 'when_idle'.")

        requests: list[ModelRequest] = []
        pending_parts: list[str | Input] = []
        for item in content:
            if isinstance(item, ModelRequest):
                if pending_parts:
                    requests.append(ModelRequest.ensure_request(pending_parts))
                    pending_parts = []
                requests.append(item)
            else:
                pending_parts.append(item)

        if pending_parts:
            requests.append(ModelRequest.ensure_request(pending_parts))

        return cls(_combine_requests(requests), priority)


def drain_pending_messages(
    queue: list[PendingMessage],
    priority: PendingMessagePriority,
) -> list[PendingMessage]:
    drained: list[PendingMessage] = []
    remaining: list[PendingMessage] = []

    for message in queue:
        if message.priority == priority:
            drained.append(message)
        else:
            remaining.append(message)

    queue[:] = remaining
    return drained


def combine_model_requests(requests: Iterable[ModelRequest]) -> ModelRequest:
    return _combine_requests(list(requests))


def _combine_requests(requests: list[ModelRequest]) -> ModelRequest:
    parts: list[Input] = []
    for request in requests:
        parts.extend(request.parts)
    return ModelRequest(parts)
