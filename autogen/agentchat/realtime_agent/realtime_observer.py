# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod


class RealtimeObserver(ABC):
    """Observer for the OpenAI Realtime API."""

    def __init__(self):
        self._client = None

    def register_client(self, client):
        """Register a client with the observer."""
        self._client = client

    @abstractmethod
    async def run(self, openai_ws):
        """Run the observer."""
        pass

    @abstractmethod
    async def update(self, message):
        """Update the observer with a message from the OpenAI Realtime API."""
        pass
