# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol, runtime_checkable

__all__ = ["WebSocketProtocol"]


@runtime_checkable
class WebSocketProtocol(Protocol):
    """WebSocket protocol for sending and receiving JSON data modelled after FastAPI's WebSocket."""

    async def send_json(self, data: Any, mode: str = "text") -> None: ...

    async def receive_json(self, mode: str = "text") -> Any: ...

    async def receive_text(self) -> str: ...
