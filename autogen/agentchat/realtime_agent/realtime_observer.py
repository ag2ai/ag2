# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .client import OpenAIRealtimeClient


class RealtimeObserver(ABC):
    """Observer for the OpenAI Realtime API."""

    def __init__(self) -> None:
        self._client: Optional["OpenAIRealtimeClient"] = None

    @property
    def client(self) -> "OpenAIRealtimeClient":
        """Get the client associated with the observer."""
        if self._client is None:
            raise ValueError("Observer client is not registered.")

        return self._client

    def register_client(self, client: "OpenAIRealtimeClient") -> None:
        """Register a client with the observer."""
        self._client = client

    @abstractmethod
    async def run(self) -> None:
        """Run the observer."""
        ...

    @abstractmethod
    async def update(self, message: dict[str, Any]) -> None:
        """Update the observer with a message from the OpenAI Realtime API."""
        ...
