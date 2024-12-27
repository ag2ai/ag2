# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from asyncio import CancelledError
from unittest.mock import MagicMock

import anyio
import pytest
from openai.types.beta.realtime.realtime_server_event import RealtimeServerEvent

from autogen.agentchat.realtime_agent.realtime_observer import RealtimeObserver


class MyObserver(RealtimeObserver):
    def __init__(self, mock: MagicMock) -> None:
        super().__init__()
        self.mock = mock

    async def _run(self) -> None:
        self.mock("started")
        try:
            self.mock("running")
            print("-> running", end="", flush=True)
            while True:
                await anyio.sleep(0.05)
                print(".", end="", flush=True)
        finally:
            print("stopped", flush=True)
            self.mock("stopped")

    async def update(self, message: RealtimeServerEvent) -> None:
        pass


class TestRealtimeObserver:
    @pytest.mark.asyncio()
    async def test_shutdown(self) -> None:

        mock = MagicMock()
        observer = MyObserver(mock)

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(observer.run)
                await anyio.sleep(1.0)
                observer.request_shutdown()

        except Exception as e:
            print(e)

        mock.assert_any_call("started")
        mock.assert_any_call("running")
        mock.assert_called_with("stopped")
