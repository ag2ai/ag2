# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import os
from typing import Annotated, Any
from unittest.mock import MagicMock

import pytest
from anyio import sleep
from asyncer import create_task_group
from conftest import MOCK_OPEN_AI_API_KEY, reason, skip_openai  # noqa: E402
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from openai import OpenAI
from test_assistant_agent import KEY_LOC, OAI_CONFIG_LIST

import autogen
from autogen.agentchat.realtime_agent import FunctionObserver, RealtimeAgent, WebsocketAudioAdapter
from autogen.agentchat.realtime_agent.oai_realtime_client import OpenAIRealtimeClient
from autogen.agentchat.realtime_agent.realtime_observer import RealtimeObserver


@pytest.mark.skipif(skip_openai, reason=reason)
class TestE2E:
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
            "temperature": 0.0,
        }

    @pytest.mark.asyncio()
    async def test_init(self, llm_config: dict[str, Any]) -> None:
        # Event for synchronization and tracking state
        weather_func_called_event = asyncio.Event()
        location_arg = ""

        app = FastAPI()
        mock_observer = MagicMock(spec=RealtimeObserver)

        @app.websocket("/media-stream")
        async def handle_media_stream(websocket: WebSocket) -> None:
            """Handle WebSocket connections providing audio stream and OpenAI."""
            await websocket.accept()

            audio_adapter = WebsocketAudioAdapter(websocket)
            agent = RealtimeAgent(
                name="Weather Bot",
                system_message="Hello there! I am an AI voice assistant powered by Autogen and the OpenAI Realtime API. You can ask me about weather, jokes, or anything you can imagine. Start by saying 'How can I help you?'",
                llm_config=llm_config,
                audio_adapter=audio_adapter,
            )

            agent.register_observer(mock_observer)

            @agent.register_realtime_function(name="get_weather", description="Get the current weather")
            def get_weather(location: Annotated[str, "city"]) -> str:
                nonlocal location_arg
                location_arg = location
                weather_func_called_event.set()  # Signal that the function was called
                return "The weather is cloudy." if location == "Seattle" else "The weather is sunny."

            async with create_task_group() as tg:
                tg.soonify(agent.run)()
                await sleep(10)  # Run for 10 seconds
                tg.cancel_scope.cancel()

            await websocket.close()

        # Simulate speech input
        tts_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = tts_client.audio.speech.create(
            model="tts-1", voice="alloy", input="What is the weather in Seattle?", response_format="pcm"
        )

        pcm_data = response.content  # Get the PCM content

        client = TestClient(app)
        with client.websocket_connect("/media-stream") as websocket:
            websocket.send_json(
                {
                    "event": "media",
                    "media": {
                        "timestamp": 0,
                        "payload": base64.b64encode(pcm_data).decode("utf-8"),
                    },
                }
            )

            # Wait for the weather function to be called or timeout
            try:
                await asyncio.wait_for(weather_func_called_event.wait(), timeout=15)
            except asyncio.TimeoutError:
                assert False, "Weather function was not called within the expected time"

            # Verify the function call details
            assert location_arg == "Seattle", "Weather function was not called with the correct location"

            last_response_transcript = mock_observer.on_event.call_args_list[-1][0][0]["response"]["output"][0][
                "content"
            ][0]["transcript"]
            assert "Seattle" in last_response_transcript, "Weather response did not include the location"
            assert "cloudy" in last_response_transcript, "Weather response did not include the weather condition"

        print("test_init() finished")
