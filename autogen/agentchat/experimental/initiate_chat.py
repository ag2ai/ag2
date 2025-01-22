# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterable, Iterable, Optional, Protocol, Union, runtime_checkable
from uuid import UUID

from ...io import IOStream
from .chat_context import ChatContext

if TYPE_CHECKING:
    from ...messages import BaseMessage
    from ..agent import Agent

__all__ = ["AsyncResponseProtocol", "ChatContext", "ResponseProtocol", "a_initiate_chat", "initiate_chat"]


@runtime_checkable
class ResponseProtocol(Protocol):
    @property
    def messages(self) -> Iterable["BaseMessage"]:
        """The messages received by the agent."""
        ...

    # todo: replace request_uuid with InputResponseMessage
    def send(self, request_uuid: UUID, response: str) -> None:
        """Send a response to a request."""
        ...


@runtime_checkable
class AsyncResponseProtocol(Protocol):
    @property
    def messages(self) -> AsyncIterable["BaseMessage"]:
        """The messages received by the agent."""
        ...


@dataclass
class Response:
    iostream: IOStream
    chat_context: ChatContext

    @property
    def messages(self) -> Iterable["BaseMessage"]:
        """The messages received by the agent."""
        raise NotImplementedError("This function is not implemented yet.")

    # todo: replace request_uuid with InputResponseMessage
    def send(self, request_uuid: UUID, response: str) -> None:
        """Send a response to a request."""
        raise NotImplementedError("This function is not implemented yet.")

    # check if the Response implements the ResponseProtocol protocol
    if TYPE_CHECKING:

        @staticmethod
        def _type_check(iostream: IOStream, chat_context: ChatContext) -> ResponseProtocol:
            return Response(iostream=iostream, chat_context=chat_context)


class InitiateChatIOStream:
    def print(self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        """Print data to the output stream.

        Args:
            objects (any): The data to print.
            sep (str, optional): The separator between objects. Defaults to " ".
            end (str, optional): The end of the output. Defaults to "\n".
            flush (bool, optional): Whether to flush the output. Defaults to False.
        """
        raise NotImplementedError("This function is not implemented yet.")

    def send(self, message: BaseMessage) -> None:
        """Send a message to the output stream.

        Args:
            message (Any): The message to send.
        """
        raise NotImplementedError("This function is not implemented yet.")

    def input(self, prompt: str = "", *, password: bool = False) -> str:
        """Read a line from the input stream.

        Args:
            prompt (str, optional): The prompt to display. Defaults to "".
            password (bool, optional): Whether to read a password. Defaults to False.

        Returns:
            str: The line read from the input stream.

        """
        raise NotImplementedError("This function is not implemented yet.")

    # check if the InitiateChatIOStream implements the IOStream protocol
    if TYPE_CHECKING:

        @staticmethod
        def _type_check(agent: Agent) -> IOStream:
            return InitiateChatIOStream()


def initiate_chat(
    agent: "Agent",
    *,
    message: Union[str, dict[str, Any]],
    recipient: Optional["Agent"] = None,
) -> ResponseProtocol:
    # start initiate chat in a background thread
    iostream = InitiateChatIOStream()
    chat_context = ChatContext(
        initial_agent=agent,
        agents=[recipient] if recipient else [],
        initial_message=message,
    )
    response = Response(iostream=iostream, chat_context=chat_context)

    with ThreadPoolExecutor() as executor:
        executor.submit(agent.initiate_chat, agent, message=message, recipient=recipient)

    return response


async def a_initiate_chat(
    agent: "Agent",
    *,
    message: Union[str, dict[str, Any]],
    recipient: Optional["Agent"] = None,
) -> ResponseProtocol:
    raise NotImplementedError("This function is not implemented yet.")
