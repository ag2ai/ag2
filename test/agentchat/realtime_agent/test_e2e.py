# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from typing import Annotated, Any
from unittest.mock import MagicMock

import pytest
from anyio import Event, move_on_after, sleep
from asyncer import create_task_group
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from autogen.agentchat.realtime_agent import RealtimeAgent, RealtimeObserver, WebSocketAudioAdapter

from ...conftest import reason, skip_openai  # noqa: E402
from .realtime_test_utils import Credentials, text_to_speech, trace

logger = getLogger(__name__)


@pytest.mark.skipif(skip_openai, reason=reason)
class TestE2E:
    async def _test_e2e(self, credentials: Credentials) -> None:
        """End-to-end test for the RealtimeAgent.

        Create a FastAPI app with a WebSocket endpoint that handles audio stream and OpenAI.

        """
        llm_config = credentials.llm_config
        openai_api_key = credentials.openai_api_key

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
            with move_on_after(20) as scope:
                await weather_func_called_event.wait()
            assert weather_func_called_event.is_set(), "Weather function was not called within the expected time"
            assert not scope.cancel_called, "Cancel scope was called before the weather function was called"

            # Verify the function call details
            weather_func_mock.assert_called_with(location="Seattle")

            last_response_transcript = mock_observer.on_event.call_args_list[-1][0][0]["response"]["output"][0][
                "content"
            ][0]["transcript"]
            assert "Seattle" in last_response_transcript, "Weather response did not include the location"
            assert "cloudy" in last_response_transcript, "Weather response did not include the weather condition"

    @pytest.mark.asyncio()
    async def test_e2e(self, credentials: Credentials) -> None:
        """End-to-end test for the RealtimeAgent.

        Retry the test up to 3 times if it fails. Sometimes the test fails due to voice not being recognized by the OpenAI API.

        """
        i = 0
        count = 3
        while True:
            try:
                await self._test_e2e(credentials=credentials)
                return  # Exit the function if the test passes
            except Exception as e:
                logger.warning(
                    f"Test 'TestE2E.test_e2e' failed on attempt {i + 1} with exception: {e}", stack_info=True
                )
                if i + 1 >= count:
                    raise
            i += 1
