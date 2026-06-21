# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..run_response import AsyncRunResponseProtocol, RunResponseProtocol

__all__ = ["AsyncEventProcessorProtocol", "EventProcessorProtocol"]


class EventProcessorProtocol(Protocol):
    def process(self, response: "RunResponseProtocol") -> None: ...


class AsyncEventProcessorProtocol(Protocol):
    async def process(self, response: "AsyncRunResponseProtocol") -> None: ...
