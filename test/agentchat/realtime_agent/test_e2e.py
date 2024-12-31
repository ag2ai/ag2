# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from typing import Annotated, Any
from unittest.mock import MagicMock

import pytest
from anyio import Event, fail_after, sleep
from asyncer import create_task_group
from conftest import reason, skip_openai  # noqa: E402
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from test_assistant_agent import KEY_LOC, OAI_CONFIG_LIST

import autogen
from autogen.agentchat.realtime_agent import RealtimeAgent, RealtimeObserver, WebSocketAudioAdapter

from .realtime_test_utils import text_to_speech, trace

logger = getLogger(__name__)


@pytest.mark.skipif(skip_openai, reason=reason)
class TestE2E:
    @pytest.fixture
    def llm_config(self) -> dict[str, Any]:
        """Fixture to load the LLM config."""
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
            "temperature": 0.0,
        }

    @pytest.fixture
    def openai_api_key(self, llm_config: dict[str, Any]) -> str:
        """Fixture to get the OpenAI API key."""
        return llm_config["config_list"][0]["api_key"]  # type: ignore[no-any-return]

    async def _test_e2e(self, llm_config: dict[str, Any], openai_api_key: str) -> None:
        """End-to-end test for the RealtimeAgent.

        Create a FastAPI app with a WebSocket endpoint that handles audio stream and OpenAI.

        """
        # Event for synchronization and tracking state
        weather_func_called_event = Event()
        weather_func_mock = MagicMock()

        app = FastAPI()
        mock_observer = MagicMock(spec=RealtimeObserver)

        @app.websocket("/media-stream")
        async def handle_media_stream(websocket: WebSocket) -> None:
            """Handle WebSocket connections providing audio stream and OpenAI."""
            await websocket.accept()

            audio_adapter = WebSocketAudioAdapter(websocket)
            agent = RealtimeAgent(
                name="Weather Bot",
                system_message="Hello there! I am an AI voice assistant powered by Autogen and the OpenAI Realtime API. You can ask me about weather, jokes, or anything you can imagine. Start by saying 'How can I help you?'",
                llm_config=llm_config,
                audio_adapter=audio_adapter,
            )

            agent.register_observer(mock_observer)

            @agent.register_realtime_function(name="get_weather", description="Get the current weather")
            @trace(weather_func_mock, postcall_event=weather_func_called_event)
            def get_weather(location: Annotated[str, "city"]) -> str:
                return "The weather is cloudy." if location == "Seattle" else "The weather is sunny."

            async with create_task_group() as tg:
                tg.soonify(agent.run)()
                await sleep(10)  # Run for 10 seconds
                tg.cancel_scope.cancel()

            assert tg.cancel_scope.cancel_called, "Task group was not cancelled"

            await websocket.close()

        client = TestClient(app)
        with client.websocket_connect("/media-stream") as websocket:
            websocket.send_json(
                {
                    "event": "media",
                    "media": {
                        "timestamp": 0,
                        "payload": text_to_speech(text="How is the weather in Seattle?", openai_api_key=openai_api_key),
                    },
                }
            )

            # Wait for the weather function to be called or timeout
            try:
                with fail_after(20) as _:
                    await weather_func_called_event.wait()
            except TimeoutError:
                assert False, "Weather function was not called within the expected time"

            # Verify the function call details
            weather_func_mock.assert_called_with(location="Seattle")

            last_response_transcript = mock_observer.on_event.call_args_list[-1][0][0]["response"]["output"][0][
                "content"
            ][0]["transcript"]
            assert "Seattle" in last_response_transcript, "Weather response did not include the location"
            assert "cloudy" in last_response_transcript, "Weather response did not include the weather condition"

    @pytest.mark.asyncio()
    async def test_e2e(self, llm_config: dict[str, Any], openai_api_key: str) -> None:
        """End-to-end test for the RealtimeAgent.

        Retry the test up to 3 times if it fails. Sometimes the test fails due to voice not being recognized by the OpenAI API.

        """
        last_exception = None

        for i in range(3):
            try:
                await self._test_e2e(llm_config, openai_api_key=openai_api_key)
                return  # Exit the function if the test passes
            except Exception as e:
                logger.warning(
                    f"Test 'TestE2E.test_e2e' failed on attempt {i + 1} with exception: {e}", stack_info=True
                )
                last_exception = e  # Keep track of the last exception

        # If the loop finishes without success, raise the last exception
        if last_exception:
            raise last_exception
