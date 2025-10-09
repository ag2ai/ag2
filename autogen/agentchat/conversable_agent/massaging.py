# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from ...events.agent_events import create_received_event_model
from ...io.base import IOStream
from ...runtime_logging import log_event, logging_enabled
from ..agent import Agent

if TYPE_CHECKING:
    from .base import ConversableAgentBase


class MessagingMixin:
    """Mixin class for message handling and communication functionality"""

    @staticmethod
    def _message_to_dict(message: dict[str, Any] | str) -> dict:
        """Convert a message to a dictionary.

        The message can be a string or a dictionary. The string will be put in the "content" field of the new dictionary.
        """
        if isinstance(message, str):
            return {"content": message}
        elif isinstance(message, dict):
            return message
        else:
            return dict(message)

    def _append_oai_message(
        self: "ConversableAgentBase", message: dict[str, Any] | str, role, conversation_id: Agent, is_sending: bool
    ) -> bool:
        """Append a message to the ChatCompletion conversation.

        If the message received is a string, it will be put in the "content" field of the new dictionary.
        If the message received is a dictionary but does not have any of the three fields "content", "function_call", or "tool_calls",
            this message is not a valid ChatCompletion message.
        If only "function_call" or "tool_calls" is provided, "content" will be set to None if not provided, and the role of the message will be forced "assistant".

        Args:
            message (dict or str): message to be appended to the ChatCompletion conversation.
            role (str): role of the message, can be "assistant" or "function".
            conversation_id (Agent): id of the conversation, should be the recipient or sender.
            is_sending (bool): If the agent (aka self) is sending to the conversation_id agent, otherwise receiving.

        Returns:
            bool: whether the message is appended to the ChatCompletion conversation.
        """
        message = self._message_to_dict(message)
        # create oai message to be appended to the oai conversation that can be passed to oai directly.
        oai_message = {
            k: message[k]
            for k in ("content", "function_call", "tool_calls", "tool_responses", "tool_call_id", "name", "context")
            if k in message and message[k] is not None
        }
        if "content" not in oai_message:
            if "function_call" in oai_message or "tool_calls" in oai_message:
                oai_message["content"] = None  # if only function_call is provided, content will be set to None.
            else:
                return False

        if message.get("role") in ["function", "tool"]:
            oai_message["role"] = message.get("role")
            if "tool_responses" in oai_message:
                for tool_response in oai_message["tool_responses"]:
                    tool_response["content"] = str(tool_response["content"])
        elif "override_role" in message:
            # If we have a direction to override the role then set the
            # role accordingly. Used to customise the role for the
            # select speaker prompt.
            oai_message["role"] = message.get("override_role")
        else:
            oai_message["role"] = role

        if oai_message.get("function_call", False) or oai_message.get("tool_calls", False):
            oai_message["role"] = "assistant"  # only messages with role 'assistant' can have a function call.
        elif "name" not in oai_message:
            # If we don't have a name field, append it
            if is_sending:
                oai_message["name"] = self.name
            else:
                oai_message["name"] = conversation_id.name

        self._oai_messages[conversation_id].append(oai_message)

        return True

    def _process_message_before_send(
        self: "ConversableAgentBase", message: dict[str, Any] | str, recipient: Agent, silent: bool
    ) -> dict[str, Any] | str:
        """Process the message before sending it to the recipient."""
        hook_list = self.hook_lists["process_message_before_send"]
        for hook in hook_list:
            message = hook(sender=self, message=message, recipient=recipient, silent=self._is_silent(self, silent))
        return message

    def send(
        self: "ConversableAgentBase",
        message: dict[str, Any] | str,
        recipient: Agent,
        request_reply: bool | None = None,
        silent: bool | None = False,
    ):
        """Send a message to another agent.

        Args:
            message (dict or str): message to be sent.
                The message could contain the following fields:
                - content (str or List): Required, the content of the message. (Can be None)
                - function_call (str): the name of the function to be called.
                - name (str): the name of the function to be called.
                - role (str): the role of the message, any role that is not "function"
                    will be modified to "assistant".
                - context (dict): the context of the message, which will be passed to
                    [OpenAIWrapper.create](https://docs.ag2.ai/latest/docs/api-reference/autogen/OpenAIWrapper/#autogen.OpenAIWrapper.create).
                    For example, one agent can send a message A as:
        ```python
        {
            "content": lambda context: context["use_tool_msg"],
            "context": {"use_tool_msg": "Use tool X if they are relevant."},
        }
        ```
                    Next time, one agent can send a message B with a different "use_tool_msg".
                    Then the content of message A will be refreshed to the new "use_tool_msg".
                    So effectively, this provides a way for an agent to send a "link" and modify
                    the content of the "link" later.
            recipient (Agent): the recipient of the message.
            request_reply (bool or None): whether to request a reply from the recipient.
            silent (bool or None): (Experimental) whether to print the message sent.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
        """
        message = self._process_message_before_send(message, recipient, self._is_silent(self, silent))
        # When the agent composes and sends the message, the role of the message is "assistant"
        # unless it's "function".
        valid = self._append_oai_message(message, "assistant", recipient, is_sending=True)
        if valid:
            recipient.receive(message, self, request_reply, silent)
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

    async def a_send(
        self: "ConversableAgentBase",
        message: dict[str, Any] | str,
        recipient: Agent,
        request_reply: bool | None = None,
        silent: bool | None = False,
    ):
        """(async) Send a message to another agent.

        Args:
            message (dict or str): message to be sent.
                The message could contain the following fields:
                - content (str or List): Required, the content of the message. (Can be None)
                - function_call (str): the name of the function to be called.
                - name (str): the name of the function to be called.
                - role (str): the role of the message, any role that is not "function"
                    will be modified to "assistant".
                - context (dict): the context of the message, which will be passed to
                    [OpenAIWrapper.create](https://docs.ag2.ai/latest/docs/api-reference/autogen/OpenAIWrapper/#autogen.OpenAIWrapper.create).
                    For example, one agent can send a message A as:
        ```python
        {
            "content": lambda context: context["use_tool_msg"],
            "context": {"use_tool_msg": "Use tool X if they are relevant."},
        }
        ```
                    Next time, one agent can send a message B with a different "use_tool_msg".
                    Then the content of message A will be refreshed to the new "use_tool_msg".
                    So effectively, this provides a way for an agent to send a "link" and modify
                    the content of the "link" later.
            recipient (Agent): the recipient of the message.
            request_reply (bool or None): whether to request a reply from the recipient.
            silent (bool or None): (Experimental) whether to print the message sent.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
        """
        message = self._process_message_before_send(message, recipient, self._is_silent(self, silent))
        # When the agent composes and sends the message, the role of the message is "assistant"
        # unless it's "function".
        valid = self._append_oai_message(message, "assistant", recipient, is_sending=True)
        if valid:
            await recipient.a_receive(message, self, request_reply, silent)
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

    def _print_received_message(
        self: "ConversableAgentBase", message: dict[str, Any] | str, sender: Agent, skip_head: bool = False
    ):
        message = self._message_to_dict(message)
        message_model = create_received_event_model(event=message, sender=sender, recipient=self)
        iostream = IOStream.get_default()
        # message_model.print(iostream.print)
        iostream.send(message_model)

    def _process_received_message(
        self: "ConversableAgentBase", message: dict[str, Any] | str, sender: Agent, silent: bool
    ):
        # When the agent receives a message, the role of the message is "user". (If 'role' exists and is 'function', it will remain unchanged.)
        valid = self._append_oai_message(message, "user", sender, is_sending=False)
        if logging_enabled():
            log_event(self, "received_message", message=message, sender=sender.name, valid=valid)

        if not valid:
            raise ValueError(
                "Received message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

        if not self._is_silent(sender, silent):
            self._print_received_message(message, sender)

    def receive(
        self: "ConversableAgentBase",
        message: dict[str, Any] | str,
        sender: Agent,
        request_reply: bool | None = None,
        silent: bool | None = False,
    ):
        """Receive a message from another agent.

        Once a message is received, this function sends a reply to the sender or stop.
        The reply can be generated automatically or entered manually by a human.

        Args:
            message (dict or str): message from the sender. If the type is dict, it may contain the following reserved fields (either content or function_call need to be provided).
                1. "content": content of the message, can be None.
                2. "function_call": a dictionary containing the function name and arguments. (deprecated in favor of "tool_calls")
                3. "tool_calls": a list of dictionaries containing the function name and arguments.
                4. "role": role of the message, can be "assistant", "user", "function", "tool".
                    This field is only needed to distinguish between "function" or "assistant"/"user".
                5. "name": In most cases, this field is not needed. When the role is "function", this field is needed to indicate the function name.
                6. "context" (dict): the context of the message, which will be passed to
                    [OpenAIWrapper.create](https://docs.ag2.ai/latest/docs/api-reference/autogen/OpenAIWrapper/#autogen.OpenAIWrapper.create).
            sender: sender of an Agent instance.
            request_reply (bool or None): whether a reply is requested from the sender.
                If None, the value is determined by `self.reply_at_receive[sender]`.
            silent (bool or None): (Experimental) whether to print the message received.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
        """
        self._process_received_message(message, sender, silent)
        if request_reply is False or (request_reply is None and self.reply_at_receive[sender] is False):
            return
        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            self.send(reply, sender, silent=silent)

    async def a_receive(
        self: "ConversableAgentBase",
        message: dict[str, Any] | str,
        sender: Agent,
        request_reply: bool | None = None,
        silent: bool | None = False,
    ):
        """(async) Receive a message from another agent.

        Once a message is received, this function sends a reply to the sender or stop.
        The reply can be generated automatically or entered manually by a human.

        Args:
            message (dict or str): message from the sender. If the type is dict, it may contain the following reserved fields (either content or function_call need to be provided).
                1. "content": content of the message, can be None.
                2. "function_call": a dictionary containing the function name and arguments. (deprecated in favor of "tool_calls")
                3. "tool_calls": a list of dictionaries containing the function name and arguments.
                4. "role": role of the message, can be "assistant", "user", "function".
                    This field is only needed to distinguish between "function" or "assistant"/"user".
                5. "name": In most cases, this field is not needed. When the role is "function", this field is needed to indicate the function name.
                6. "context" (dict): the context of the message, which will be passed to
                    [OpenAIWrapper.create](https://docs.ag2.ai/latest/docs/api-reference/autogen/OpenAIWrapper/#autogen.OpenAIWrapper.create).
            sender: sender of an Agent instance.
            request_reply (bool or None): whether a reply is requested from the sender.
                If None, the value is determined by `self.reply_at_receive[sender]`.
            silent (bool or None): (Experimental) whether to print the message received.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
        """
        self._process_received_message(message, sender, silent)
        if request_reply is False or (request_reply is None and self.reply_at_receive[sender] is False):
            return
        reply = await self.a_generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            await self.a_send(reply, sender, silent=silent)
