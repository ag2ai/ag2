# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import TypeAlias

import httpx
from a2a.client.errors import A2AClientError, A2AClientTimeoutError
from a2a.types import TaskState

HttpxClientFactory: TypeAlias = Callable[[], httpx.AsyncClient]


# Terminal task states from which no further progress is expected.
# A2A 1.0 ``TaskState`` is a proto enum (int subclass via ``EnumTypeWrapper``),
# so frozenset membership against the raw enum values just works.
TERMINAL_TASK_STATES: frozenset[int] = frozenset({
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_REJECTED,
})


# Errors that should trigger reconnect on the streaming-message path.
TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    A2AClientError,
    A2AClientTimeoutError,
)
