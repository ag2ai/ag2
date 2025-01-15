# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0


"""
## Setup
    pip install google-genai
"""

import json
from contextlib import asynccontextmanager
from logging import Logger, getLogger
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, Optional

from .realtime_client import Role, register_realtime_client

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

    from .realtime_client import RealtimeClientProtocol

from websockets.asyncio.client import connect

__all__ = ["GeminiRealtimeClient", "Role"]

global_logger = getLogger(__name__)


HOST = "generativelanguage.googleapis.com"
API_VERSION = "v1alpha"


@register_realtime_client()
class GeminiRealtimeClient:
    """(Experimental) Client for Gemini Realtime API."""

    def __init__(
        self,
        *,
        llm_config: dict[str, Any],
        voice: str,
        system_message: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """(Experimental) Client for Gemini Realtime API.

        Args:
            llm_config (dict[str, Any]): The config for the client.
        """
        self._llm_config = llm_config
        self._voice = voice
        self._system_message = system_message
        self._logger = logger
        self._connection: Optional["ClientConnection"] = None
        config = llm_config["config_list"][0]

        self._model: str = config["model"]
        self._temperature: float = config.get("temperature", 0.8)  # type: ignore[union-attr]

        self._response_modality = "AUDIO"

        self.api_key = config.get("api_key", None)
        self._final_config: Dict[str, Any] = {}
        self._pending_session_updates: dict[str, Any] = {}
        self._is_reading_events = False

    @property
    def logger(self) -> Logger:
        """Get the logger for the Gemini Realtime API."""
        return self._logger or global_logger

    @property
    def connection(self) -> "ClientConnection":
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
        if self._is_reading_events:
            await self.connection.send(json.dumps(msg))

    # https://github.com/google-gemini/cookbook/blob/18bb4f2bd03c66839dc388bb1e9ae7e7819b1cd0/gemini-2/websockets/live_api_starter.py#L106
    async def send_text(self, *, role: Role, text: str, turn_complete: bool = True) -> None:
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
        if self._is_reading_events:
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
        if self._is_reading_events:
            await self.connection.send(json.dumps(msg))

    async def truncate_audio(self, audio_end_ms: int, content_index: int, item_id: str) -> None:
        self.logger.info("This is not natively supported by Gemini Realtime API.")
        pass

    async def _initialize_session(self) -> None:
        # https://ai.google.dev/api/multimodal-live
        # https://ai.google.dev/api/multimodal-live#bidigeneratecontentsetup

        session_config = {
            "setup": {
                "system_instruction": {"role": "system", "parts": [{"text": self._system_message}]},
                "model": f"models/{self._model}",
                "tools": [
                    {
                        "function_declarations": [
                            {
                                "name": tool_schema["name"],
                                "description": tool_schema["description"],
                                "parameters": tool_schema[
                                    "parameters"
                                ],  # GeminiClient._create_gemini_function_parameters(tool_schema["parameters"]),
                            }
                            for tool_schema in self._pending_session_updates.get("tools", [])
                        ]
                    },
                ],
                "generation_config": {
                    "response_modalities": [self._response_modality],
                    "speech_config": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self._voice}}},
                    "temperature": self._temperature,
                },
            }
        }

        self.logger.info(f"Sending session update: {session_config}")
        await self.connection.send(json.dumps(session_config))

    async def session_update(self, session_options: dict[str, Any]) -> None:
        """Record or apply session updates."""
        if self._is_reading_events:
            self.logger.warning("Is reading events. Session update would be ignored.")
        # Record session updates
        else:
            self._pending_session_updates.update(session_options)

    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[None, None]:
        """Connect to the Gemini Realtime API."""
        # https://github.com/google-gemini/cookbook/blob/main/gemini-2/websockets/live_api_starter.py#L82
        uri = f"wss://{HOST}/ws/google.ai.generativelanguage.{API_VERSION}.GenerativeService.BidiGenerateContent?key={self.api_key}"
        try:
            async with connect(uri, additional_headers={"Content-Type": "application/json"}) as self._connection:
                yield
        finally:
            self._connection = None

    async def read_events(self) -> AsyncGenerator[dict[str, Any], None]:
        """Read Audio Events"""
        if self._connection is None:
            raise RuntimeError("Client is not connected, call connect() first.")
        await self._initialize_session()

        self._is_reading_events = True

        async for raw_response in self.connection:
            response = json.loads(raw_response.decode("ascii")) if isinstance(raw_response, bytes) else raw_response
            try:
                events = self.dispatch(response)
                for event in events:
                    yield event
            except (KeyError, ValueError):
                self.logger.error(f"Failed to parse message: {response}")

    def dispatch(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        # Determine the type of message and dispatch it to the appropriate handler
        if "serverContent" in response and "modelTurn" in response["serverContent"]:
            return self.parse_audio_response(response)
        elif "toolCall" in response:
            return self.parse_tool_call(response)
        else:
            raise ValueError("Unknown message type")

    def parse_audio_response(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        # Handle audio response
        b64data = response["serverContent"]["modelTurn"]["parts"][0]["inlineData"].pop("data")
        return [
            {
                "type": "response.audio.delta",
                "delta": b64data,
                "item_id": None,
            }
        ]

    def parse_tool_call(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "type": "response.function_call_arguments.done",
                "name": call["name"],
                "arguments": json.dumps(call["args"]),
                "call_id": call["id"],
            }
            for call in response["toolCall"]["functionCalls"]
        ]

    @classmethod
    def get_factory(
        cls, llm_config: dict[str, Any], voice: str, system_message: str, logger: Logger, **kwargs: Any
    ) -> Optional[Callable[[], "RealtimeClientProtocol"]]:
        """Create a Realtime API client.

        Args:
            model (str): The model to create the client for.
            voice (str): The voice to use.
            system_message (str): The system message to use.
            kwargs (Any): Additional arguments.

        Returns:
            RealtimeClientProtocol: The Realtime API client is returned if the model matches the pattern
        """
        if llm_config["config_list"][0].get("api_type") == "google" and list(kwargs.keys()) == []:
            return lambda: GeminiRealtimeClient(
                llm_config=llm_config, voice=voice, system_message=system_message, logger=logger, **kwargs
            )
        return None


# needed for mypy to check if GeminiRealtimeClient implements RealtimeClientProtocol
if TYPE_CHECKING:
    _client: RealtimeClientProtocol = GeminiRealtimeClient(
        llm_config={}, voice="Charon", system_message="You are a helpful AI voice assistant."
    )
