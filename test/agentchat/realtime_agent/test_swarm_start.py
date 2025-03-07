# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from typing import Annotated
from unittest.mock import MagicMock

import pytest
from anyio import Event, move_on_after, sleep
from asyncer import create_task_group
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from pytest import FixtureRequest

from autogen import ConversableAgent
from autogen.agentchat.realtime.experimental import RealtimeAgent, RealtimeObserver, WebSocketAudioAdapter
from autogen.agentchat.realtime.experimental.realtime_swarm import register_swarm
from autogen.tools.dependency_injection import Field as AG2Field

from ...conftest import Credentials
from .realtime_test_utils import text_to_speech, trace

logger = getLogger(__name__)


class TestSwarmE2E:
    async def _test_e2e(self, credentials_llm_realtime: Credentials, credentials_gpt_4o_mini: Credentials) -> None:
        """End-to-end test for the RealtimeAgent.

        Create a FastAPI app with a WebSocket endpoint that handles audio stream and Realtime API.

        """
        openai_api_key = credentials_gpt_4o_mini.api_key

        # Event for synchronization and tracking state
        weather_func_called_event = Event()
        weather_func_mock = MagicMock()

        app = FastAPI()
        mock_observer = MagicMock(spec=RealtimeObserver)

        @app.websocket("/media-stream")
        async def handle_media_stream(websocket: WebSocket) -> None:
            """Handle WebSocket connections providing audio stream and Realtime API."""
            await websocket.accept()

            audio_adapter = WebSocketAudioAdapter(websocket)
            agent = RealtimeAgent(
                name="Weather_Bot",
                llm_config=credentials_llm_realtime.llm_config,
                audio_adapter=audio_adapter,
            )

            agent.register_observer(mock_observer)

            @trace(weather_func_mock, postcall_event=weather_func_called_event)
            def get_weather(location: Annotated[str, AG2Field(description="city")]) -> str:
                return "The weather is cloudy." if location == "Seattle" else "The weather is sunny."

            weatherman = ConversableAgent(
                name="Weatherman",
                system_message="You are a weatherman. You can answer questions about the weather.",
                llm_config=credentials_gpt_4o_mini.llm_config,
                functions=[get_weather],
            )

            register_swarm(
                realtime_agent=agent,
                initial_agent=weatherman,
                agents=[weatherman],
            )

            async with create_task_group() as tg:
                tg.soonify(agent.run)()
                await sleep(25)  # Run for 10 seconds
                tg.cancel_scope.cancel()

            assert tg.cancel_scope.cancel_called, "Task group was not cancelled"

            await websocket.close()

        client = TestClient(app)
        with client.websocket_connect("/media-stream") as websocket:
            await sleep(5)
            websocket.send_json({
                "event": "media",
                "media": {
                    "timestamp": 0,
                    "payload": text_to_speech(text="How is the weather in Seattle?", openai_api_key=openai_api_key),
                },
            })

            # Wait for the weather function to be called or timeout
            with move_on_after(10) as scope:
                await weather_func_called_event.wait()
            assert weather_func_called_event.is_set(), "Weather function was not called within the expected time"
            assert not scope.cancel_called, "Cancel scope was called before the weather function was called"

            # Verify the function call details
            weather_func_mock.assert_called_with(location="Seattle")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "credentials_llm_realtime",
        [
            pytest.param("credentials_gpt_4o_realtime", marks=[pytest.mark.openai_realtime, pytest.mark.aux_neg_flag]),
            pytest.param("credentials_gemini_realtime", marks=[pytest.mark.gemini_realtime, pytest.mark.aux_neg_flag]),
        ],
    )
    async def test_e2e(
        self, credentials_llm_realtime: str, credentials_gpt_4o_mini: Credentials, request: FixtureRequest
    ) -> None:
        """End-to-end test for the RealtimeAgent.

        Retry the test up to 5 times if it fails. Sometimes the test fails due to voice not being recognized by the realtime API.

        """
        i = 0
        count = 5
        while True:
            try:
                credentials = request.getfixturevalue(credentials_llm_realtime)
                await self._test_e2e(
                    credentials_llm_realtime=credentials,
                    credentials_gpt_4o_mini=credentials_gpt_4o_mini,
                )
                return  # Exit the function if the test passes
            except Exception as e:
                logger.warning(
                    f"Test 'TestSwarmE2E.test_e2e' failed on attempt {i + 1} with exception: {e}", stack_info=True
                )
                if i + 1 >= count:
                    raise
            i += 1
