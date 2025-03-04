# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from unittest.mock import MagicMock

import pytest
from anyio import move_on_after

from autogen.agentchat.realtime.experimental.clients import GeminiRealtimeClient, RealtimeClientProtocol
from autogen.agentchat.realtime.experimental.realtime_events import AudioDelta, SessionCreated
from autogen.import_utils import run_for_optional_imports

from ....conftest import Credentials, suppress_gemini_resource_exhausted


class TestGeminiRealtimeClient:
    @pytest.fixture
    def client(self, credentials_gemini_realtime: Credentials) -> GeminiRealtimeClient:
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

    @pytest.mark.asyncio
    @run_for_optional_imports(["websockets"], "gemini-realtime")
    @suppress_gemini_resource_exhausted
    async def test_not_connected(self, client: GeminiRealtimeClient) -> None:
        with pytest.raises(RuntimeError, match=r"Client is not connected, call connect\(\) first."):
            with move_on_after(1) as scope:
                async for _ in client.read_events():
                    assert False

        assert not scope.cancelled_caught

    @pytest.mark.asyncio
    @pytest.mark.skip("Test is not giving expected result in CI")
    @run_for_optional_imports(["websockets"], "gemini-realtime")
    @suppress_gemini_resource_exhausted
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

    @pytest.mark.skip
    @pytest.mark.asyncio
    @run_for_optional_imports(["websockets"], "gemini-realtime")
    @suppress_gemini_resource_exhausted
    async def test_send_text(self, client: GeminiRealtimeClient) -> None:
        mock = MagicMock()

        agent_finished_speaking = False

        async with client.connect():
            # give the agent 5 seconds to speak and finish speaking
            with move_on_after(5) as scope:
                print("Reading events...")
                async for event in client.read_events():
                    print(f"-> Received event: {event}")
                    mock(event)

                    if isinstance(event, SessionCreated):
                        await client.send_text(role="user", text="Hello, how are you?")

                    if event.raw_message == {"serverContent": {"turnComplete": True}}:
                        agent_finished_speaking = True
                        break

        # check that we received the expected SessionCreated and AudioDelta events
        calls_args = [arg_list.args for arg_list in mock.call_args_list]
        assert isinstance(calls_args[0][0], SessionCreated), f"Type of calls_args[0][0] is {type(calls_args[0][0])}"

        # if the agent finished speaking, the last event should be an Event with turnComplete=True
        if agent_finished_speaking:
            assert isinstance(calls_args[-2][0], AudioDelta), f"Type of calls_args[-1][0] is {type(calls_args[-1][0])}"
            assert calls_args[-1][0].raw_message == {"serverContent": {"turnComplete": True}}
        # if the agent did not finish speaking, the last event should be an AudioDelta and the scope should be cancelled
        else:
            assert scope.cancelled_caught
            assert isinstance(calls_args[-1][0], AudioDelta), f"Type of calls_args[-1][0] is {type(calls_args[-1][0])}"
