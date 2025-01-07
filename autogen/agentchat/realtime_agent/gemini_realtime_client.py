# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0


"""
## Setup
    pip install google-genai
"""

from contextlib import asynccontextmanager
from logging import Logger, getLogger
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Optional

from google import genai
from google.genai import types
from google.genai.live import AsyncSession

from .realtime_client import Role

if TYPE_CHECKING:
    from .realtime_client import RealtimeClientProtocol

__all__ = ["GeminiRealtimeClient", "Role"]

global_logger = getLogger(__name__)

# TODO: Configuable or hardcode? Align with OAI class?
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
        self._model: str = config["model"]

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
    # TODO: https://github.com/google-gemini/cookbook/blob/main/gemini-2/live_api_tool_use.ipynb
    async def send_function_result(self, call_id: str, result: str) -> None:  # Looks like Gemini doesn't results.
        """Send the result of a function call to the Gemini Realtime API.

        Args:
            call_id (str): The ID of the function call.
            result (str): The result of the function call.
        """
        tool_response = types.LiveClientToolResponse(
            function_responses=[
                types.FunctionResponse(
                    # name=fc.name
                    id=call_id,
                    response={"result": "ok"},
                )
            ]
        )
        await self.connection.send(tool_response)

    # TODO: No role?
    async def send_text(self, *, role: Role, text: str) -> None:
        """Send a text message to the Gemini Realtime API.

        Args:
            role (str): The role of the message.
            text (str): The text of the message.
        """
        await self.connection.send(text, end_of_turn=True)

    # TODO: the audio is str for OAI.
    async def send_audio(self, audio_stream: AsyncIterator[bytes]) -> None:
        """Send audio to the Gemini Realtime API.

        Args:
            audio (str): The audio to send.
        """
        await self.connection.start_stream(stream=audio_stream, mime_type="audio/pcm")

    # TODO: didn't find relevant API: https://github.com/google-gemini/cookbook/blob/main/gemini-2/live_api_tool_use.ipynb
    # async def truncate_audio(self, audio_end_ms: int, content_index: int, item_id: str) -> None:
    #     """Truncate audio in the Gemini Realtime API.

    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[None, None]:
        """Connect to the Gemini Realtime API."""
        try:
            async with self._client.aio.live.connect(
                model=self._model,
                config=GEMINI_CONFIG,
            ) as self._connection:
                yield
        finally:
            self._connection = None

    # TODO: Decide what to return here: message, meesage,txt, message.data, message.tool_call or whatever.
    async def read_events(self) -> AsyncGenerator[types.LiveServerMessage, None]:
        """Read messages from the Gemini Realtime API."""
        if self._connection is None:
            raise RuntimeError("Client is not connected, call connect() first.")
        try:
            async for message in self._connection.receive():
                yield message
        finally:
            self._connection = None


# needed for mypy to check if GeminiRealtimeClient implements RealtimeClientProtocol
if TYPE_CHECKING:
    _client: RealtimeClientProtocol = GeminiRealtimeClient(
        llm_config={}, system_message="You are a helpful AI voice assistant."
    )
