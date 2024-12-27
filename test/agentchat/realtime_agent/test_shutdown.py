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
            while True:
                await anyio.sleep(0.1)
                self.mock("running")
        finally:
            self.mock("stopped")

    async def update(self, message: RealtimeServerEvent) -> None:
        pass


@pytest.mark.asyncio()
async def test_shutdown() -> None:

    mock = MagicMock()
    observer = MyObserver(mock)

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(observer.run)
            await anyio.sleep(1.0)
            observer.request_shutdown()

    except Exception as e:
        print(e)

    mock.assert_called_with("started")
    mock.assert_called_with("running")
