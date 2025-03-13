# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Protocol, Union, runtime_checkable

from ..cache.abstract_cache_base import AbstractCache
from ..doc_utils import export_module
from .chat import ChatResult

if TYPE_CHECKING:
    # mypy will fail if Conversible agent does not implement Agent protocol
    from .chat import ChatResult
    from .conversable_agent import ConversableAgent


__all__ = ["DEFAULT_SUMMARY_METHOD", "Agent", "LLMAgent", "LLMMessageType"]

DEFAULT_SUMMARY_METHOD = "last_msg"

LLMMessageType = dict[str, Any]


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

    @property
    def llm_config(self) -> Union[dict[str, Any], Literal[False]]:
        """The LLM configuration of the agent."""
        ...

    def send(
        self,
        message: Union["LLMMessageType", str],
        recipient: "Agent",
        request_reply: Optional[bool] = None,
    ) -> None:
        """Send a message to another agent.

        Args:
            message (dict or str): the message to send. If a dict, it should be
                a JSON-serializable and follows the OpenAI's ChatCompletion schema.
            recipient (Agent): the recipient of the message.
            request_reply (bool): whether to request a reply from the recipient.
        """
        ...

    async def a_send(
        self,
        message: Union["LLMMessageType", str],
        recipient: "Agent",
        request_reply: Optional[bool] = None,
    ) -> None:
        """(Async) Send a message to another agent.

        Args:
            message (dict or str): the message to send. If a dict, it should be
                a JSON-serializable and follows the OpenAI's ChatCompletion schema.
            recipient (Agent): the recipient of the message.
            request_reply (bool): whether to request a reply from the recipient.
        """
        ...

    def receive(
        self,
        message: Union["LLMMessageType", str],
        sender: "Agent",
        request_reply: Optional[bool] = None,
    ) -> None:
        """Receive a message from another agent.

        Args:
            message (dict or str): the message received. If a dict, it should be
                a JSON-serializable and follows the OpenAI's ChatCompletion schema.
            sender (Agent): the sender of the message.
            request_reply (bool): whether the sender requests a reply.
        """

    async def a_receive(
        self,
        message: Union["LLMMessageType", str],
        sender: "Agent",
        request_reply: Optional[bool] = None,
    ) -> None:
        """(Async) Receive a message from another agent.

        Args:
            message (dict or str): the message received. If a dict, it should be
                a JSON-serializable and follows the OpenAI's ChatCompletion schema.
            sender (Agent): the sender of the message.
            request_reply (bool): whether the sender requests a reply.
        """
        ...

    def generate_reply(
        self,
        messages: Optional[list["LLMMessageType"]] = None,
        sender: Optional["Agent"] = None,
        **kwargs: Any,
    ) -> Union[str, dict[str, Any], None]:
        """Generate a reply based on the received messages.

        Args:
            messages (list[dict[str, Any]]): a list of messages received from other agents.
                The messages are dictionaries that are JSON-serializable and
                follows the OpenAI's ChatCompletion schema.
            sender: sender of an Agent instance.
            **kwargs: Additional keyword arguments.

        Returns:
            str or dict or None: the generated reply. If None, no reply is generated.
        """

    async def a_generate_reply(
        self,
        messages: Optional[list["LLMMessageType"]] = None,
        sender: Optional["Agent"] = None,
        **kwargs: Any,
    ) -> Union[str, dict[str, Any], None]:
        """(Async) Generate a reply based on the received messages.

        Args:
            messages (list[dict[str, Any]]): a list of messages received from other agents.
                The messages are dictionaries that are JSON-serializable and
                follows the OpenAI's ChatCompletion schema.
            sender: sender of an Agent instance.
            **kwargs: Additional keyword arguments.

        Returns:
            str or dict or None: the generated reply. If None, no reply is generated.
        """
        ...

    def initiate_chat(
        self,
        recipient: "ConversableAgent",
        clear_history: bool = True,
        silent: Optional[bool] = False,
        cache: Optional[AbstractCache] = None,
        max_turns: Optional[int] = None,
        summary_method: Optional[Union[str, Callable[..., Any]]] = DEFAULT_SUMMARY_METHOD,
        summary_args: Optional[dict[str, Any]] = {},
        message: Optional[Union["LLMMessageType", str, Callable[..., Any]]] = None,
        **kwargs: Any,
    ) -> "ChatResult": ...

    async def a_initiate_chat(
        self,
        recipient: "ConversableAgent",
        clear_history: bool = True,
        silent: Optional[bool] = False,
        cache: Optional[AbstractCache] = None,
        max_turns: Optional[int] = None,
        summary_method: Optional[Union[str, Callable[..., Any]]] = DEFAULT_SUMMARY_METHOD,
        summary_args: Optional[dict[str, Any]] = {},
        message: Optional[Union["LLMMessageType", str, Callable[..., Any]]] = None,
        **kwargs: Any,
    ) -> "ChatResult": ...


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
    # mypy will fail if Conversible agent does not implement Agent protocol
    def _check_protocol_implementation(agent: ConversableAgent) -> LLMAgent:
        return agent
