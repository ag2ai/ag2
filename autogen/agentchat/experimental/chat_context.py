# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from contextvars import ContextVar
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import TYPE_CHECKING, Any, Optional, Protocol, Union, runtime_checkable
from uuid import UUID

from pydantic import UUID4

if TYPE_CHECKING:
    from ..agent import Agent


@runtime_checkable
class ChatContextProtocol(Protocol):
    @property
    def initial_agent(self) -> "Agent":
        """The agent that initiated the chat."""
        ...

    @property
    def agents(self) -> list["Agent"]:
        """The agents participating in the chat."""
        ...

    @property
    def initial_message(self) -> Union[str, dict[str, Any]]:
        """The messages received by the agent."""
        ...

    @property
    def messages(self) -> list[dict[str, Any]]: ...

    @property
    def logger(self) -> Logger: ...

    @classmethod
    def get_registered_chat(cls) -> "ChatContextProtocol": ...

    def __entry__(self) -> "ChatContextProtocol": ...

    def __exit__(
        self, exc_type: Optional[type], exc_value: Optional[BaseException], traceback: Optional[Any]
    ) -> None: ...


@dataclass
class ChatContext:
    initial_agent: "Agent"
    agents: list["Agent"]
    initial_message: Union[str, dict[str, Any]]
    messages: list[dict[str, Any]]
    logger: Logger
    uuid: UUID

    _registered_chats: ContextVar[list["ChatContext"]] = ContextVar("registered_chats", default=[])

    def __init__(
        self,
        *,
        initial_agent: "Agent",
        agents: Optional[list["Agent"]] = None,
        initial_message: Union[str, dict[str, Any]],
        logger: Optional[Logger] = None,
        uuid: Optional[UUID] = None,
    ):
        self.initial_agent = initial_agent
        self.agents = agents or []
        self.initial_message = initial_message
        self.messages = []
        self.logger = logger or getLogger(__name__)
        self.uuid = uuid or UUID4()

    @classmethod
    def get_registered_chat(cls) -> "ChatContext":
        registered_chats: list[ChatContext] = cls._registered_chats.get()
        if registered_chats:
            return registered_chats[-1]
        raise ValueError("No registered chats found.")

    def __entry__(self) -> "ChatContext":
        registered_chats = ChatContext._registered_chats.get()
        registered_chats.append(self)
        return self

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[BaseException], traceback: Optional[Any]) -> None:
        registered_chats = ChatContext._registered_chats.get()
        registered_chats.pop()

    # check if the InitiateChatIOStream implements the IOStream protocol
    if TYPE_CHECKING:

        @staticmethod
        def _type_check(
            *,
            initial_agent: "Agent",
            agents: Optional[list["Agent"]] = None,
            initial_message: Union[str, dict[str, Any]],
            logger: Optional[Logger] = None,
            uuid: Optional[UUID] = None,
        ) -> ChatContextProtocol:
            return ChatContext(
                initial_agent=initial_agent, agents=agents, initial_message=initial_message, logger=logger, uuid=uuid
            )
