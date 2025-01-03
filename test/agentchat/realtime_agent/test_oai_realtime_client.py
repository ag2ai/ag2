# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import MagicMock

import pytest
from anyio import move_on_after

import autogen
from autogen.agentchat.realtime_agent.oai_realtime_client import OpenAIRealtimeClient
from autogen.agentchat.realtime_agent.realtime_client import RealtimeClientProtocol

from ...conftest import MOCK_OPEN_AI_API_KEY, reason, skip_openai  # noqa: E402
from ..test_assistant_agent import KEY_LOC, OAI_CONFIG_LIST


class TestOAIRealtimeClient:
    @pytest.fixture
    def llm_config(self) -> dict[str, Any]:
        config_list = autogen.config_list_from_json(
            OAI_CONFIG_LIST,
            filter_dict={
                "tags": ["gpt-4o-realtime"],
            },
            file_location=KEY_LOC,
        )
        assert config_list, "No config list found"
        return {
            "config_list": config_list,
            "temperature": 0.8,
        }

    @pytest.fixture
    def client(self, llm_config: dict[str, Any]) -> RealtimeClientProtocol:
        return OpenAIRealtimeClient(
            llm_config=llm_config,
            voice="alloy",
            system_message="You are a helpful AI assistant with voice capabilities.",
        )

    def test_init(self) -> None:
        llm_config = {
            "config_list": [
                {
                    "model": "gpt-4o",
                    "api_key": MOCK_OPEN_AI_API_KEY,
                },
            ],
            "temperature": 0.8,
        }
        client = OpenAIRealtimeClient(
            llm_config=llm_config,
            voice="alloy",
            system_message="You are a helpful AI assistant with voice capabilities.",
        )
        assert isinstance(client, RealtimeClientProtocol)

    @pytest.mark.skipif(skip_openai, reason=reason)
    @pytest.mark.asyncio()
    async def test_not_connected(self, client: OpenAIRealtimeClient) -> None:
        with pytest.raises(RuntimeError, match=r"Client is not connected, call connect\(\) first."):
            with move_on_after(1) as scope:
                async for _ in client.read_events():
                    pass

        assert not scope.cancelled_caught

    @pytest.mark.skipif(skip_openai, reason=reason)
    @pytest.mark.asyncio()
    async def test_start_read_events(self, client: OpenAIRealtimeClient) -> None:
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

    @pytest.mark.skipif(skip_openai, reason=reason)
    @pytest.mark.asyncio()
    async def test_send_text(self, client: OpenAIRealtimeClient) -> None:
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
