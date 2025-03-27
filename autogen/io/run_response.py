# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from typing import Any, AsyncIterable, Iterable, Optional, Protocol
from uuid import UUID

from ..events.base_event import BaseEvent

Message = dict[str, Any]


class RunInfoProtocol(Protocol):
    @property
    def uuid(self) -> UUID: ...

    @property
    def above_run(self) -> Optional["RunResponseProtocol"]: ...


class RunResponseProtocol(RunInfoProtocol, Protocol):
    @property
    def events(self) -> Iterable[BaseEvent]: ...

    @property
    def messages(self) -> Iterable[Message]: ...

    @property
    def summary(self) -> Optional[str]: ...


class AsyncRunResponseProtocol(RunInfoProtocol, Protocol):
    @property
    def events(self) -> AsyncIterable[BaseEvent]: ...

    @property
    def messages(self) -> AsyncIterable[Message]: ...

    @property
    async def summary(self) -> str: ...
