# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager
from logging import Logger, getLogger
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

from asyncer import TaskGroup, create_task_group
from openai import DEFAULT_MAX_RETRIES, NOT_GIVEN, AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection

from .realtime_client import Role

if TYPE_CHECKING:
    from .realtime_client import RealtimeClientProtocol

__all__ = ["OpenAIRealtimeClient", "Role"]

global_logger = getLogger(__name__)


class OpenAIRealtimeClient:
    """(Experimental) Client for OpenAI Realtime API."""

    def __init__(
        self,
        *,
        llm_config: dict[str, Any],
        voice: str,
        system_message: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """(Experimental) Client for OpenAI Realtime API.

        Args:
            llm_config (dict[str, Any]): The config for the client.
        """
        self._llm_config = llm_config
        self._voice = voice
        self._system_message = system_message
        self._logger = logger

        self._connection: Optional[AsyncRealtimeConnection] = None

        config = llm_config["config_list"][0]
        self._model: str = config["model"]
        self._temperature: float = llm_config.get("temperature", 0.8)  # type: ignore[union-attr]

        self._client = AsyncOpenAI(
            api_key=config.get("api_key", None),
            organization=config.get("organization", None),
            project=config.get("project", None),
            base_url=config.get("base_url", None),
            websocket_base_url=config.get("websocket_base_url", None),
            timeout=config.get("timeout", NOT_GIVEN),
            max_retries=config.get("max_retries", DEFAULT_MAX_RETRIES),
            default_headers=config.get("default_headers", None),
            default_query=config.get("default_query", None),
        )

    @property
    def logger(self) -> Logger:
        """Get the logger for the OpenAI Realtime API."""
        return self._logger or global_logger

    @property
    def connection(self) -> AsyncRealtimeConnection:
        """Get the OpenAI WebSocket connection."""
        if self._connection is None:
            raise RuntimeError("OpenAI WebSocket is not initialized")
        return self._connection

    async def send_function_result(self, call_id: str, result: str) -> None:
        """Send the result of a function call to the OpenAI Realtime API.

        Args:
            call_id (str): The ID of the function call.
            result (str): The result of the function call.
        """
        await self.connection.conversation.item.create(
            item={
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            },
        )

        await self.connection.response.create()

    async def send_text(self, *, role: Role, text: str) -> None:
        """Send a text message to the OpenAI Realtime API.

        Args:
            role (str): The role of the message.
            text (str): The text of the message.
        """
        await self.connection.response.cancel()
        await self.connection.conversation.item.create(
            item={"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]}
        )
        await self.connection.response.create()

    async def send_audio(self, audio: str) -> None:
        """Send audio to the OpenAI Realtime API.

        Args:
            audio (str): The audio to send.
        """
        await self.connection.input_audio_buffer.append(audio=audio)

    async def truncate_audio(self, audio_end_ms: int, content_index: int, item_id: str) -> None:
        """Truncate audio in the OpenAI Realtime API.

        Args:
            audio_end_ms (int): The end of the audio to truncate.
            content_index (int): The index of the content to truncate.
            item_id (str): The ID of the item to truncate.
        """
        await self.connection.conversation.item.truncate(
            audio_end_ms=audio_end_ms, content_index=content_index, item_id=item_id
        )

    async def _initialize_session(self) -> None:
        """Control initial session with OpenAI."""
        session_update = {
            "turn_detection": {"type": "server_vad"},
            "voice": self._voice,
            "instructions": self._system_message,
            "modalities": ["audio", "text"],
            "temperature": self._temperature,
        }
        await self.session_update(session_options=session_update)

    async def session_update(self, session_options: dict[str, Any]) -> None:
        """Send a session update to the OpenAI Realtime API.

        Args:
            session_options (dict[str, Any]): The session options to update.
        """
        logger = self.logger
        logger.info(f"Sending session update: {session_options}")
        await self.connection.session.update(session=session_options)  # type: ignore[arg-type]
        logger.info("Sending session update finished")

    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[None, None]:
        """Connect to the OpenAI Realtime API."""
        try:
            async with self._client.beta.realtime.connect(
                model=self._model,
            ) as self._connection:
                await self._initialize_session()
                yield
        finally:
            self._connection = None

    async def read_events(self) -> AsyncGenerator[dict[str, Any], None]:
        """Read messages from the OpenAI Realtime API."""
        if self._connection is None:
            raise RuntimeError("Client is not connected, call connect() first.")

        try:
            async for event in self._connection:
                yield event.model_dump()

        finally:
            self._connection = None


# needed for mypy to check if OpenAIRealtimeClient implements RealtimeClientProtocol
if TYPE_CHECKING:
    _client: RealtimeClientProtocol = OpenAIRealtimeClient(
        llm_config={}, voice="alloy", system_message="You are a helpful AI voice assistant."
    )
