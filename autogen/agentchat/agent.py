# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from typing import TYPE_CHECKING, Any, Optional, Protocol, TypeVar, runtime_checkable

from ..doc_utils import export_module

__all__ = ["Agent", "LLMAgent", "LLMMessageType"]

Tool = TypeVar("Tool")

LLMMessageType = dict[str, Any]

DEFAULT_SUMMARY_METHOD = "last_msg"


@runtime_checkable
@export_module("autogen")
class Agent(Protocol):
    """(In preview) A protocol for Agent.

    An agent can communicate with other agents and perform actions.
    Different agents can differ in what actions they perform in the `receive` method.
    """

    @property
    def name(self) -> str:
        """The name of the agent."""
        ...

    @property
    def description(self) -> str:
        """The description of the agent. Used for the agent's introduction in
        a group chat setting.
        """
        ...

    def send(
        self,
        message: list[dict[str, Any]],
        recipient: "Agent",
        request_reply: bool | None = None,
    ) -> None:
        """Send a list[message] to another agent.

        Args:
            message (list[dict[str, Any]], str, or dict): the message to send.
                - If a list of dicts, should be JSON-serializable and follow OpenAI's ChatCompletion schema.
                - If a str or dict, will be automatically normalized to list format.
            recipient (Agent): the recipient of the message.
            request_reply (bool): whether to request a reply from the recipient.
        """
        ...

    async def a_send(
        self,
        message: list[dict[str, Any]],
        recipient: "Agent",
        request_reply: bool | None = None,
    ) -> None:
        """(Async) Send a list[message] to another agent.
        Args:
            message (list[dict[str, Any]], str, or dict): the message to send.
                - If a list of dicts, should be JSON-serializable and follow OpenAI's ChatCompletion schema.
                - If a str or dict, will be automatically normalized to list format.
            recipient (Agent): the recipient of the message.
            request_reply (bool): whether to request a reply from the recipient.
        """
        ...

    def receive(
        self,
        message: list[dict[str, Any]],
        sender: "Agent",
        request_reply: bool | None = None,
        silent: bool | None = False,
    ) -> None:
        """Receive a list[message] from another agent.

        Args:
            message (list[messages]): the list[messages] received. If a list of messages, it should be
                a JSON-serializable and follows the OpenAI's ChatCompletion schema.
            sender (Agent): the sender of the message.
            request_reply (bool): whether the sender requests a reply.
            silent (bool): whether to print the message received.
        """

    async def a_receive(
        self,
        message: list[dict[str, Any]],
        sender: "Agent",
        request_reply: bool | None = None,
        silent: bool | None = False,
    ) -> None:
        """(Async) Receive a list[message] from another agent.

        Args:
            message (list[messages]): the list[messages] received. If a list of messages, it should be
                a JSON-serializable and follows the OpenAI's ChatCompletion schema.
            sender (Agent): the sender of the message.
            request_reply (bool): whether the sender requests a reply.
            silent (bool): whether to print the message received.
        """
        ...

    def generate_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: Optional["Agent"] = None,
    ) -> str | dict[str, Any] | None:
        """Generate a reply based on the received messages.

        Args:
            messages (list[dict[str, Any]]): a list of messages received from other agents. can be a single message.
                The messages are dictionaries that are JSON-serializable and
                follows the OpenAI's ChatCompletion schema.
            sender: sender of an Agent instance.
            **kwargs: Additional keyword arguments.

        Returns:
            str or dict or None: the generated reply. If None, no reply is generated.
        """

    async def a_generate_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: Optional["Agent"] = None,
    ) -> str | dict[str, Any] | None:
        """(Async) Generate a reply based on the received messages.

        Args:
            messages (list[dict[str, Any]]): a list of messages received from other agents. can be a single message.
                The messages are dictionaries that are JSON-serializable and
                follows the OpenAI's ChatCompletion schema.
            sender: sender of an Agent instance.
            **kwargs: Additional keyword arguments.

        Returns:
            str or dict or None: the generated reply. If None, no reply is generated.
        """
        ...

    def set_ui_tools(self, tools: list[Tool]) -> None:
        """Set the UI tools for the agent.

        Args:
            tools: a list of UI tools to set.
        """
        ...

    def unset_ui_tools(self, tools: list[Tool]) -> None:
        """Unset the UI tools for the agent.

        Args:
            tools: a list of UI tools to set.
        """
        ...


@runtime_checkable
@export_module("autogen")
class LLMAgent(Agent, Protocol):
    """(In preview) A protocol for an LLM agent."""

    @property
    def system_message(self) -> str:
        """The system message of this agent."""

    def update_system_message(self, system_message: str) -> None:
        """Update this agent's system message.

        Args:
            system_message (str): system message for inference.
        """


if TYPE_CHECKING:
    # mypy will fail if Conversable agent does not implement Agent protocol
    from .conversable_agent import ConversableAgent

    def _check_protocol_implementation(agent: ConversableAgent) -> Agent:
        return agent
