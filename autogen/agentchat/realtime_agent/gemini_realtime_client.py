# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0


"""
## Setup
    pip install google-genai
"""

import asyncio
import json
from contextlib import asynccontextmanager
from logging import Logger, getLogger
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Optional

from google import genai
from google.genai import types
from google.genai.live import AsyncSession

from .realtime_client import Role

if TYPE_CHECKING:
    from .realtime_client import RealtimeClientProtocol
from websockets.asyncio.client import connect

__all__ = ["GeminiRealtimeClient", "Role"]

global_logger = getLogger(__name__)


HOST = "generativelanguage.googleapis.com"
# uri = f"wss://{host}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={api_key}"


# TODO: Configurable or hardcode? Align with OAI class?
GEMINI_CONFIG = {
    # TODO: function call description: https://ai.google.dev/gemini-api/docs/models/gemini-v2#compositional-function-calling
    # TODO: function call example: https://github.com/google-gemini/cookbook/blob/main/gemini-2/live_api_tool_use.ipynb
    "tools": [],
    # Voice setting: https://github.com/google-gemini/cookbook/issues/378
    "generation_config": {"response_modalities": ["AUDIO"], "speech_config": "Charon"},
}
API_VERSION = "v1alpha"


class GeminiRealtimeClient:
    """(Experimental) Client for Gemini Realtime API."""

    def __init__(
        self,
        *,
        llm_config: dict[str, Any],
        # voice: str,
        system_message: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """(Experimental) Client for Gemini Realtime API.

        Args:
            llm_config (dict[str, Any]): The config for the client.
        """
        self._llm_config = llm_config
        self._system_message = system_message
        self._logger = logger
        self._connection: Optional[AsyncSession] = None
        self._model: str = llm_config[
            "model"
        ]  # As of 01/08/2025, only gemini-2.0-flash-exp support Multimodal Live API

        config = llm_config["config_list"][0]
        self._client = genai.Client(
            vertexai=config.get("vertexai", None),
            api_key=config.get("api_key", None),
            credentials=config.get("credentials", None),
            project=config.get("project", None),
            location=config.get("location", None),
            debug_config=config.get("debug_config", None),
            http_options={"api_version": API_VERSION},
        )

        self.api_key = config.get("api_key", None)
        # self.host = "generativelanguage.googleapis.com"
        # self.uri = f"wss://{host}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={api_key}"

        self._final_config = {}
        self._pending_session_updates = []
        self._is_reading_events = False

    @property
    def logger(self) -> Logger:
        """Get the logger for the Gemini Realtime API."""
        return self._logger or global_logger

    @property
    def connection(self) -> AsyncSession:
        """Get the Gemini WebSocket connection."""
        if self._connection is None:
            raise RuntimeError("Gemini WebSocket is not initialized")
        return self._connection

    # TODO: The following is to support the function_declarations. How about code_execution tool and google_search tool?
    # https://github.com/google-gemini/cookbook/blob/main/gemini-2/live_api_tool_use.ipynb
    # gemini-2/websockets/live_api_tool_use.ipynb
    async def send_function_result(self, call_id: str, result: str) -> None:  # Looks like Gemini doesn't results.
        """Send the result of a function call to the Gemini Realtime API.

        Args:
            call_id (str): The ID of the function call.
            result (str): The result of the function call.
        """
        msg = {
            "tool_response": {"function_responses": [{"id": call_id, "response": {"result": {"string_value": result}}}]}
        }
        await self.connection.send(json.dumps(msg))

    # https://github.com/google-gemini/cookbook/blob/18bb4f2bd03c66839dc388bb1e9ae7e7819b1cd0/gemini-2/websockets/live_api_starter.py#L106
    async def send_text(self, *, role: Role, text: str, turn_complete=True) -> None:
        """Send a text message to the Gemini Realtime API.

        Args:
            role (str): The role of the message.
            text (str): The text of the message.
        """
        msg = {
            "client_content": {
                "turn_complete": turn_complete,
                "turns": [{"role": role, "parts": [{"text": text}]}],
            }
        }
        await self.connection.send(json.dumps(msg))

    # https://github.com/google-gemini/cookbook/blob/18bb4f2bd03c66839dc388bb1e9ae7e7819b1cd0/gemini-2/websockets/live_api_starter.py#L201
    async def send_audio(self, audio: str) -> None:
        """Send audio to the Gemini Realtime API.

        Args:
            audio (str): The audio to send.
        """
        msg = {
            "realtime_input": {
                "media_chunks": [
                    {
                        "data": audio,
                        "mime_type": "audio/pcm",
                    }
                ]
            }
        }
        await self.connection.send(json.dumps(msg))

    # TODO: probably need to figure out a way to implement this.
    async def truncate_audio(self, audio_end_ms: int, content_index: int, item_id: str) -> None:
        pass

    def _consolidate_session_updates(self) -> dict[str, Any]:
        """Combine all pending session updates into a single configuration."""
        final_config = {}
        for update in self._pending_session_updates:
            final_config.update(update)
        return final_config

    async def _initialize_session(self) -> None:
        final_session_config = self._consolidate_session_updates()
        await self.connection.send(json.dumps(final_session_config))

    async def session_update(self, session_options: dict[str, Any]) -> None:
        """Record or apply session updates."""
        if self._is_reading_events:
            self.logger.warning("Is reading events. Session update would be ignored.")
        # Record session updates
        else:
            self._pending_session_updates.append(session_options)

    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[None, None]:
        """Connect to the Gemini Realtime API."""
        uri = f"wss://{HOST}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"
        try:
            async with connect(uri, additional_headers={"Content-Type": "application/json"}) as self._connection:
                yield
        finally:
            self._connection = None

    async def read_events(self):
        """Read Audio Events"""
        if self._connection is None:
            raise RuntimeError("Client is not connected, call connect() first.")

        # Last moment you can send the sessions updates, before the conversion starts. The moment conversation starts. Check inside the send text, send audio.
        await self._initialize_session()

        self._is_reading_events = True
        async for raw_response in self.connection:
            try:
                response = json.loads(raw_response.decode("ascii"))
                b64data = response["serverContent"]["modelTurn"]["parts"][0]["inlineData"]["data"]
                event = {
                    "type": "response.audio.delta",
                    "delta": b64data,
                }
                yield event
            except KeyError:
                self.logger.error("Failed to parse audio event: %s", response)

    # TODO: didn't find relevant API: https://github.com/google-gemini/cookbook/blob/main/gemini-2/live_api_tool_use.ipynb
    # async def truncate_audio(self, audio_end_ms: int, content_index: int, item_id: str) -> None:
    #     """Truncate audio in the Gemini Realtime API.

    # TODO: Decide what to return here: message, message,txt, message.data, message.tool_call or whatever.
    # async def read_events(self) -> AsyncGenerator[types.LiveServerMessage, None]:
    #     """Read messages from the Gemini Realtime API."""
    #     if self._connection is None:
    #         raise RuntimeError("Client is not connected, call connect() first.")
    #     try:
    #         async for message in self._connection.receive():
    #             yield message.
    #     finally:
    #         self._connection = None


# needed for mypy to check if GeminiRealtimeClient implements RealtimeClientProtocol
if TYPE_CHECKING:
    _client: RealtimeClientProtocol = GeminiRealtimeClient(
        llm_config={}, system_message="You are a helpful AI voice assistant."
    )
