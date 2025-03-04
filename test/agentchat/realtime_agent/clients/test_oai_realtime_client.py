# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from anyio import move_on_after

from autogen.agentchat.realtime.experimental.clients import OpenAIRealtimeClient, RealtimeClientProtocol
from autogen.agentchat.realtime.experimental.realtime_events import AudioDelta, SessionCreated, SessionUpdated
from autogen.import_utils import run_for_optional_imports

from ....conftest import Credentials


class TestOAIRealtimeClient:
    @pytest.fixture
    def client(self, credentials_gpt_4o_realtime: Credentials) -> OpenAIRealtimeClient:
        llm_config = credentials_gpt_4o_realtime.llm_config
        return OpenAIRealtimeClient(
            llm_config=llm_config,
        )

    def test_init(self, mock_credentials: Credentials) -> None:
        llm_config = mock_credentials.llm_config

        client = OpenAIRealtimeClient(
            llm_config=llm_config,
        )
        assert isinstance(client, RealtimeClientProtocol)

    @run_for_optional_imports(["openai", "websockets"], "openai-realtime")
    @pytest.mark.asyncio
    async def test_not_connected(self, client: OpenAIRealtimeClient) -> None:
        with pytest.raises(RuntimeError, match=r"Client is not connected, call connect\(\) first."):
            with move_on_after(1) as scope:
                async for _ in client.read_events():
                    pass

        assert not scope.cancelled_caught

    @pytest.mark.skip
    @run_for_optional_imports(["openai-realtime"], "openai-realtime")
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
        assert len(mock.call_args_list) > 0
        assert len(mock.call_args_list) > 1
        assert isinstance(calls_kwargs[0][0], SessionCreated)
        assert isinstance(calls_kwargs[1][0], SessionUpdated)

    @pytest.mark.skip
    @run_for_optional_imports(["openai-realtime"], "openai-realtime")
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

        # check that we received the model response (audio or just text)
        assert isinstance(calls_args[-1][0], AudioDelta) or (calls_args[-1][0].raw_message["type"] == "response.done")
