# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from anyio import move_on_after

from autogen.agentchat.realtime_agent.clients import OpenAIRealtimeClient, RealtimeClientProtocol
from autogen.agentchat.realtime_agent.realtime_events import AudioDelta, SessionCreated, SessionUpdated

from ....conftest import Credentials


class TestOAIRealtimeClient:
    @pytest.fixture
    def client(self, credentials_gpt_4o_realtime: Credentials) -> RealtimeClientProtocol:
        llm_config = credentials_gpt_4o_realtime.llm_config
        return OpenAIRealtimeClient(
            llm_config=llm_config,
            voice="alloy",
        )

    def test_init(self, mock_credentials: Credentials) -> None:
        llm_config = mock_credentials.llm_config

        client = OpenAIRealtimeClient(
            llm_config=llm_config,
            voice="alloy",
        )
        assert isinstance(client, RealtimeClientProtocol)

    @pytest.mark.openai
    @pytest.mark.asyncio
    async def test_not_connected(self, client: OpenAIRealtimeClient) -> None:
        with pytest.raises(RuntimeError, match=r"Client is not connected, call connect\(\) first."):
            with move_on_after(1) as scope:
                async for _ in client.read_events():
                    pass

        assert not scope.cancelled_caught

    @pytest.mark.openai
    @pytest.mark.asyncio
    async def test_start_read_events(self, client: OpenAIRealtimeClient) -> None:
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

        # check that we received the expected two events
        calls_kwargs = [arg_list.args for arg_list in mock.call_args_list]
        assert isinstance(calls_kwargs[0][0], SessionCreated)
        assert isinstance(calls_kwargs[1][0], SessionUpdated)

    @pytest.mark.openai
    @pytest.mark.asyncio
    async def test_send_text(self, client: OpenAIRealtimeClient) -> None:
        mock = MagicMock()

        async with client.connect():
            # read events for 3 seconds and then interrupt
            with move_on_after(3) as scope:
                print("Reading events...")
                async for event in client.read_events():
                    print(f"-> Received event: {event}")
                    mock(event)

                    if isinstance(event, SessionUpdated):
                        await client.send_text(role="user", text="Hello, how are you?")

        # checking if the scope was cancelled by move_on_after
        assert scope.cancelled_caught

        # check that we received the expected two events
        calls_args = [arg_list.args for arg_list in mock.call_args_list]
        assert isinstance(calls_args[0][0], SessionCreated)
        assert isinstance(calls_args[1][0], SessionUpdated)

        # check that we received the model audio response
        assert isinstance(calls_args[-1][0], AudioDelta)
