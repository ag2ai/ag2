# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from anyio import move_on_after

from autogen.agentchat.realtime_agent.clients import GeminiRealtimeClient, RealtimeClientProtocol

from ....conftest import Credentials


class TestGeminiRealtimeClient:
    @pytest.fixture
    def client(self, credentials_gemini_realtime: Credentials) -> RealtimeClientProtocol:
        llm_config = credentials_gemini_realtime.llm_config
        return GeminiRealtimeClient(
            llm_config=llm_config,
            voice="charon",
            system_message="You are a helpful AI assistant with voice capabilities.",
        )

    def test_init(self, mock_credentials: Credentials) -> None:
        llm_config = mock_credentials.llm_config

        client = GeminiRealtimeClient(
            llm_config=llm_config,
            voice="charon",
            system_message="You are a helpful AI assistant with voice capabilities.",
        )
        assert isinstance(client, RealtimeClientProtocol)

    @pytest.mark.openai
    @pytest.mark.asyncio
    async def test_not_connected(self, client: GeminiRealtimeClient) -> None:
        with pytest.raises(RuntimeError, match=r"Client is not connected, call connect\(\) first."):
            with move_on_after(1) as scope:
                async for _ in client.read_events():
                    pass

        assert not scope.cancelled_caught

    @pytest.mark.openai
    @pytest.mark.asyncio
    async def test_start_read_events(self, client: GeminiRealtimeClient) -> None:
        mock = MagicMock()

        async with client.connect():
            # read events for 3 seconds and then interrupt
            with move_on_after(3) as scope:
                print("Reading events...")

                async for event in client.read_events():
                    print(f"-> Received event: {event}")
                    mock(**event)

        # checking if the scope was cancelled by move_on_after
        assert scope.cancelled_caught

        # check that we received the expected two events
        calls_kwargs = [arg_list.kwargs for arg_list in mock.call_args_list]
        assert calls_kwargs[0]["type"] == "session.created"
        assert calls_kwargs[1]["type"] == "session.updated"

    @pytest.mark.openai
    @pytest.mark.asyncio
    async def test_send_text(self, client: GeminiRealtimeClient) -> None:
        mock = MagicMock()

        async with client.connect():
            # read events for 3 seconds and then interrupt
            with move_on_after(3) as scope:
                print("Reading events...")
                async for event in client.read_events():
                    print(f"-> Received event: {event}")
                    mock(**event)

                    if event["type"] == "session.updated":
                        await client.send_text(role="user", text="Hello, how are you?")

        # checking if the scope was cancelled by move_on_after
        assert scope.cancelled_caught

        # check that we received the expected two events
        calls_kwargs = [arg_list.kwargs for arg_list in mock.call_args_list]
        assert calls_kwargs[0]["type"] == "session.created"
        assert calls_kwargs[1]["type"] == "session.updated"

        assert calls_kwargs[2]["type"] == "error"
        assert calls_kwargs[2]["error"]["message"] == "Cancellation failed: no active response found"

        assert calls_kwargs[3]["type"] == "conversation.item.created"
        assert calls_kwargs[3]["item"]["content"][0]["text"] == "Hello, how are you?"
