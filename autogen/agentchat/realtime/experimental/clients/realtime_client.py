# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator
from logging import Logger
from typing import Any, AsyncContextManager, Callable, Literal, Optional, Protocol, Type, TypeVar, runtime_checkable

from .....doc_utils import export_module
from ..realtime_events import RealtimeEvent

__all__ = ["RealtimeClientProtocol", "Role", "get_client", "register_realtime_client"]

# define role literal type for typing
Role = Literal["user", "assistant", "system"]


@runtime_checkable
@export_module("autogen.agentchat.realtime.experimental.clients")
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

    def read_events(self) -> AsyncGenerator[RealtimeEvent, None]:
        """Read messages from a Realtime API."""
        ...

    def _parse_message(self, message: dict[str, Any]) -> list[RealtimeEvent]:
        """Parse a message from a Realtime API.

        Args:
            message (dict[str, Any]): The message to parse.

        Returns:
            list[RealtimeEvent]: The parsed events.
        """
        ...

    @classmethod
    def get_factory(
        cls, llm_config: dict[str, Any], logger: Logger, **kwargs: Any
    ) -> Optional[Callable[[], "RealtimeClientProtocol"]]:
        """Create a Realtime API client.

        Args:
            llm_config (dict[str, Any]): The config for the client.
            kwargs (Any): Additional arguments.

        Returns:
            RealtimeClientProtocol: The Realtime API client is returned if the model matches the pattern
        """
        ...


_realtime_client_classes: dict[str, Type[RealtimeClientProtocol]] = {}

T = TypeVar("T", bound=RealtimeClientProtocol)


def register_realtime_client() -> Callable[[Type[T]], Type[T]]:
    """Register a Realtime API client.

    Args:
        name (str): The name of the Realtime API client.

    Returns:
        Callable[[Type[T]], Type[T]]: The decorator to register the Realtime API client
    """

    def decorator(client_cls: Type[T]) -> Type[T]:
        """Register a Realtime API client.

        Args:
            client (RealtimeClientProtocol): The client to register.
        """
        global _realtime_client_classes
        fqn = f"{client_cls.__module__}.{client_cls.__name__}"
        _realtime_client_classes[fqn] = client_cls

        return client_cls

    return decorator


@export_module("autogen.agentchat.realtime.experimental.clients")
def get_client(llm_config: dict[str, Any], logger: Logger, **kwargs: Any) -> "RealtimeClientProtocol":
    """Get a registered Realtime API client.

    Args:
        llm_config (dict[str, Any]): The config for the client.
        kwargs (Any): Additional arguments.

    Returns:
        RealtimeClientProtocol: The Realtime API client.
    """
    global _realtime_client_classes
    for _, client_cls in _realtime_client_classes.items():
        factory = client_cls.get_factory(llm_config=llm_config, logger=logger, **kwargs)
        if factory:
            return factory()

    raise ValueError("Realtime API client not found.")
