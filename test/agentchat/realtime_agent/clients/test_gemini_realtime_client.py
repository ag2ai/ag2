# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from unittest.mock import MagicMock

import pytest
from anyio import move_on_after

from autogen.agentchat.realtime_agent.clients import GeminiRealtimeClient, RealtimeClientProtocol
from autogen.agentchat.realtime_agent.realtime_events import AudioDelta, SessionCreated

from ....conftest import Credentials


class TestGeminiRealtimeClient:
    @pytest.fixture
    def client(self, credentials_gemini_realtime: Credentials) -> RealtimeClientProtocol:
        llm_config = credentials_gemini_realtime.llm_config
        return GeminiRealtimeClient(
            llm_config=llm_config,
        )

    def test_init(self, mock_credentials: Credentials) -> None:
        llm_config = mock_credentials.llm_config

        client = GeminiRealtimeClient(
            llm_config=llm_config,
        )
        assert isinstance(client, RealtimeClientProtocol)

    @pytest.mark.gemini
    @pytest.mark.asyncio
    async def test_not_connected(self, client: GeminiRealtimeClient) -> None:
        with pytest.raises(RuntimeError, match=r"Client is not connected, call connect\(\) first."):
            with move_on_after(1) as scope:
                async for _ in client.read_events():
                    pass

        assert not scope.cancelled_caught

    @pytest.mark.gemini
    @pytest.mark.asyncio
    async def test_start_read_events(self, client: GeminiRealtimeClient) -> None:
        mock = MagicMock()

        async with client.connect():
            # read events for 3 seconds and then interrupt
            with move_on_after(3) as scope:
                print("Reading events...")

                async for event in client.read_events():
                    print(f"-> Received event: {event}")
                    mock(event)

        # checking if the scope was cancelled by move_on_after
        assert scope.cancelled_caught

        # check that we received the expected SessionCreated event
        calls_args = [arg_list.args for arg_list in mock.call_args_list]

        assert isinstance(calls_args[0][0], SessionCreated)

    @pytest.mark.gemini
    @pytest.mark.asyncio
    async def test_send_text(self, client: GeminiRealtimeClient) -> None:
        mock = MagicMock()

        async with client.connect():
            # read events for 3 seconds and then interrupt
            with move_on_after(3) as scope:
                print("Reading events...")
                async for event in client.read_events():
                    print(f"-> Received event: {event}")
                    mock(event)

                    if isinstance(event, SessionCreated):
                        await client.send_text(role="user", text="Hello, how are you?")

        # checking if the scope was cancelled by move_on_after
        assert scope.cancelled_caught

        # check that we received the expected SessionCreated and AudioDelta events
        calls_args = [arg_list.args for arg_list in mock.call_args_list]
        assert isinstance(calls_args[0][0], SessionCreated)

        assert isinstance(calls_args[-1][0], AudioDelta)
