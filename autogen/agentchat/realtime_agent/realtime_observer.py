# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod


class RealtimeObserver(ABC):
    def __init__(self):
        self._client = None

    def register_client(self, client):
        self._client = client

    @abstractmethod
    async def run(self, openai_ws):
        pass

    @abstractmethod
    async def update(self, message):
        pass
