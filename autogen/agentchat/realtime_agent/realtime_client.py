# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, AsyncContextManager, AsyncGenerator, Literal, Protocol, runtime_checkable

__all__ = ["RealtimeClientProtocol", "Role"]

# define role literal type for typing
Role = Literal["user", "assistant", "system"]


@runtime_checkable
class RealtimeClientProtocol(Protocol):
    async def send_function_result(self, call_id: str, result: str) -> None:
        """Send the result of a function call to a Realtime API.

        Args:
            call_id (str): The ID of the function call.
            result (str): The result of the function call.
        """
        ...

    async def send_text(self, *, role: Role, text: str) -> None:
        """Send a text message to a Realtime API.

        Args:
            role (str): The role of the message.
            text (str): The text of the message.
        """
        ...

    async def send_audio(self, audio: str) -> None:
        """Send audio to a Realtime API.

        Args:
            audio (str): The audio to send.
        """
        ...

    async def truncate_audio(self, audio_end_ms: int, content_index: int, item_id: str) -> None:
        """Truncate audio in a Realtime API.

        Args:
            audio_end_ms (int): The end of the audio to truncate.
            content_index (int): The index of the content to truncate.
            item_id (str): The ID of the item to truncate.
        """
        ...

    async def session_update(self, session_options: dict[str, Any]) -> None:
        """Send a session update to a Realtime API.

        Args:
            session_options (dict[str, Any]): The session options to update.
        """
        ...

    def connect(self) -> AsyncContextManager[None]: ...

    def read_events(self) -> AsyncGenerator[dict[str, Any], None]:
        """Read messages from a Realtime API."""
        ...
