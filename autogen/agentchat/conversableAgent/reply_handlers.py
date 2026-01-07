# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import inspect
import logging
from collections.abc import Container
from inspect import iscoroutine
from typing import TYPE_CHECKING, Any, Optional

from ...events.agent_events import (
    TerminationAndHumanReplyNoInputEvent,
    TerminationEvent,
    UsingAutoReplyEvent,
)
from ...exception_utils import SenderRequiredError
from ...fast_depends.utils import is_coroutine_callable
from ...io.base import AsyncIOStreamProtocol, AsyncInputStream, IOStream, IOStreamProtocol, InputStream
from ...runtime_logging import log_event, logging_enabled
from ..agent import Agent

if TYPE_CHECKING:
    from .base import ConversableAgentBase

logger = logging.getLogger(__name__)


class ReplyHandlersMixin:
    """Mixin class for reply generation and processing functionality"""

    def register_reply(
        self: "ConversableAgentBase",
        trigger: type[Agent] | str | Agent | callable | list,
        reply_func: callable,
        position: int = 0,
        config: Any | None = None,
        reset_config: callable | None = None,
        *,
        ignore_async_in_sync_chat: bool = False,
        remove_other_reply_funcs: bool = False,
    ):
        """Register a reply function.

        The reply function will be called when the trigger matches the sender.
        The function registered later will be checked earlier by default.
        To change the order, set the position to a positive integer.

        Both sync and async reply functions can be registered. The sync reply function will be triggered
        from both sync and async chats. However, an async reply function will only be triggered from async
        chats (initiated with `ConversableAgent.a_initiate_chat`). If an `async` reply function is registered
        and a chat is initialized with a sync function, `ignore_async_in_sync_chat` determines the behaviour as follows:
            if `ignore_async_in_sync_chat` is set to `False` (default value), an exception will be raised, and
            if `ignore_async_in_sync_chat` is set to `True`, the reply function will be ignored.

        Args:
            trigger (Agent class, str, Agent instance, callable, or list): the trigger.
                If a class is provided, the reply function will be called when the sender is an instance of the class.
                If a string is provided, the reply function will be called when the sender's name matches the string.
                If an agent instance is provided, the reply function will be called when the sender is the agent instance.
                If a callable is provided, the reply function will be called when the callable returns True.
                If a list is provided, the reply function will be called when any of the triggers in the list is activated.
                If None is provided, the reply function will be called only when the sender is None.
                Note: Be sure to register `None` as a trigger if you would like to trigger an auto-reply function with non-empty messages and `sender=None`.
            reply_func (callable): the reply function.
                The function takes a recipient agent, a list of messages, a sender agent and a config as input and returns a reply message.

                ```python
                def reply_func(
                    recipient: ConversableAgent,
                    messages: Optional[List[Dict]] = None,
                    sender: Optional[Agent] = None,
                    config: Optional[Any] = None,
                ) -> Tuple[bool, Union[str, Dict, None]]:
                ```
            position (int): the position of the reply function in the reply function list.
                The function registered later will be checked earlier by default.
                To change the order, set the position to a positive integer.
            config (Any): the config to be passed to the reply function.
                When an agent is reset, the config will be reset to the original value.
            reset_config (callable): the function to reset the config.
                The function returns None. Signature: ```def reset_config(config: Any)```
            ignore_async_in_sync_chat (bool): whether to ignore the async reply function in sync chats. If `False`, an exception
                will be raised if an async reply function is registered and a chat is initialized with a sync
                function.
            remove_other_reply_funcs (bool): whether to remove other reply functions when registering this reply function.
        """
        if not isinstance(trigger, (type, str, Agent, callable, list)):
            raise ValueError("trigger must be a class, a string, an agent, a callable or a list.")
        if remove_other_reply_funcs:
            self._reply_func_list.clear()
        self._reply_func_list.insert(
            position,
            {
                "trigger": trigger,
                "reply_func": reply_func,
                "config": copy.copy(config),
                "init_config": config,
                "reset_config": reset_config,
                "ignore_async_in_sync_chat": ignore_async_in_sync_chat and is_coroutine_callable(reply_func),
            },
        )

    def replace_reply_func(self: "ConversableAgentBase", old_reply_func: callable, new_reply_func: callable):
        """Replace a registered reply function with a new one.

        Args:
            old_reply_func (callable): the old reply function to be replaced.
            new_reply_func (callable): the new reply function to replace the old one.
        """
        for f in self._reply_func_list:
            if f["reply_func"] == old_reply_func:
                f["reply_func"] = new_reply_func

    def _raise_exception_on_async_reply_functions(self: "ConversableAgentBase") -> None:
        """Raise an exception if any async reply functions are registered.

        Raises:
            RuntimeError: if any async reply functions are registered.
        """
        reply_functions = {
            f["reply_func"] for f in self._reply_func_list if not f.get("ignore_async_in_sync_chat", False)
        }

        async_reply_functions = [f for f in reply_functions if is_coroutine_callable(f)]
        if async_reply_functions:
            msg = (
                "Async reply functions can only be used with ConversableAgent.a_initiate_chat(). The following async reply functions are found: "
                + ", ".join([f.__name__ for f in async_reply_functions])
            )

            raise RuntimeError(msg)

    def generate_reply(
        self: "ConversableAgentBase",
        messages: list[dict[str, Any]] | None = None,
        sender: Optional["Agent"] = None,
        exclude: Container[Any] = (),
    ) -> str | dict[str, Any] | None:
        """Reply based on the conversation history and the sender.

        Either messages or sender must be provided.
        Register a reply_func with `None` as one trigger for it to be activated when `messages` is non-empty and `sender` is `None`.
        Use registered auto reply functions to generate replies.
        By default, the following functions are checked in order:
        1. check_termination_and_human_reply
        2. generate_function_call_reply (deprecated in favor of tool_calls)
        3. generate_tool_calls_reply
        4. generate_code_execution_reply
        5. generate_oai_reply
        Every function returns a tuple (final, reply).
        When a function returns final=False, the next function will be checked.
        So by default, termination and human reply will be checked first.
        If not terminating and human reply is skipped, execute function or code and return the result.
        AI replies are generated only when no code execution is performed.

        Args:
            messages: a list of messages in the conversation history.
            sender: sender of an Agent instance.
            exclude: A list of reply functions to exclude from
                the reply generation process. Functions in this list will be skipped even if
                they would normally be triggered.

        Returns:
            str or dict or None: reply. None if no reply is generated.
        """
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender]

        # Call the hookable method that gives registered hooks a chance to update agent state, used for their context variables.
        self.update_agent_state_before_reply(messages)

        # Call the hookable method that gives registered hooks a chance to process the last message.
        # Message modifications do not affect the incoming messages or self._oai_messages.
        messages = self.process_last_received_message(messages)

        # Call the hookable method that gives registered hooks a chance to process all messages.
        # Message modifications do not affect the incoming messages or self._oai_messages.
        messages = self.process_all_messages_before_reply(messages)

        for reply_func_tuple in self._reply_func_list:
            reply_func = reply_func_tuple["reply_func"]
            if reply_func in exclude:
                continue
            if is_coroutine_callable(reply_func):
                continue
            if self._match_trigger(reply_func_tuple["trigger"], sender):
                final, reply = reply_func(self, messages=messages, sender=sender, config=reply_func_tuple["config"])
                if logging_enabled():
                    log_event(
                        self,
                        "reply_func_executed",
                        reply_func_module=reply_func.__module__,
                        reply_func_name=reply_func.__name__,
                        final=final,
                        reply=reply,
                    )
                if final:
                    return reply
        return self._default_auto_reply

    async def a_generate_reply(
        self: "ConversableAgentBase",
        messages: list[dict[str, Any]] | None = None,
        sender: Optional["Agent"] = None,
        exclude: Container[Any] = (),
    ) -> str | dict[str, Any] | None:
        """(async) Reply based on the conversation history and the sender.

        Either messages or sender must be provided.
        Register a reply_func with `None` as one trigger for it to be activated when `messages` is non-empty and `sender` is `None`.
        Use registered auto reply functions to generate replies.
        By default, the following functions are checked in order:
        1. check_termination_and_human_reply
        2. generate_function_call_reply
        3. generate_tool_calls_reply
        4. generate_code_execution_reply
        5. generate_oai_reply
        Every function returns a tuple (final, reply).
        When a function returns final=False, the next function will be checked.
        So by default, termination and human reply will be checked first.
        If not terminating and human reply is skipped, execute function or code and return the result.
        AI replies are generated only when no code execution is performed.

        Args:
            messages: a list of messages in the conversation history.
            sender: sender of an Agent instance.
            exclude: A list of reply functions to exclude from
                the reply generation process. Functions in this list will be skipped even if
                they would normally be triggered.

        Returns:
            str or dict or None: reply. None if no reply is generated.
        """
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender]

        # Call the hookable method that gives registered hooks a chance to update agent state, used for their context variables.
        self.update_agent_state_before_reply(messages)

        # Call the hookable method that gives registered hooks a chance to process the last message.
        # Message modifications do not affect the incoming messages or self._oai_messages.
        messages = self.process_last_received_message(messages)

        # Call the hookable method that gives registered hooks a chance to process all messages.
        # Message modifications do not affect the incoming messages or self._oai_messages.
        messages = self.process_all_messages_before_reply(messages)

        for reply_func_tuple in self._reply_func_list:
            reply_func = reply_func_tuple["reply_func"]
            if reply_func in exclude:
                continue

            if self._match_trigger(reply_func_tuple["trigger"], sender):
                if is_coroutine_callable(reply_func):
                    final, reply = await reply_func(
                        self, messages=messages, sender=sender, config=reply_func_tuple["config"]
                    )
                else:
                    final, reply = reply_func(self, messages=messages, sender=sender, config=reply_func_tuple["config"])
                if final:
                    return reply
        return self._default_auto_reply

    def _match_trigger(
        self: "ConversableAgentBase", trigger: None | str | type | Agent | callable | list, sender: Agent | None
    ) -> bool:
        """Check if the sender matches the trigger.

        Args:
            trigger (Union[None, str, type, Agent, callable, List]): The condition to match against the sender.
            Can be `None`, string, type, `Agent` instance, callable, or a list of these.
            sender (Agent): The sender object or type to be matched against the trigger.

        Returns:
            `True` if the sender matches the trigger, otherwise `False`.

        Raises:
            ValueError: If the trigger type is unsupported.
        """
        if trigger is None:
            return sender is None
        elif isinstance(trigger, str):
            if sender is None:
                raise SenderRequiredError()
            return trigger == sender.name
        elif isinstance(trigger, type):
            return isinstance(sender, trigger)
        elif isinstance(trigger, Agent):
            # return True if the sender is the same type (class) as the trigger
            return trigger == sender
        elif isinstance(trigger, callable):
            rst = trigger(sender)
            assert isinstance(rst, bool), f"trigger {trigger} must return a boolean value."
            return rst
        elif isinstance(trigger, list):
            return any(self._match_trigger(t, sender) for t in trigger)
        else:
            raise ValueError(f"Unsupported trigger type: {type(trigger)}")

    def get_human_input(
        self: "ConversableAgentBase",
        prompt: str,
        *,
        iostream: InputStream | None = None,
    ) -> str:
        """Get human input.

        Override this method to customize the way to get human input.

        Args:
            prompt (str): prompt for the human input.
            iostream (Optional[InputStream]): The InputStream object to use for sending messages.
        Returns:
            str: human input.
        """
        iostream = iostream or IOStream.get_default()
        reply = iostream.input(prompt)
        # Process the human input through hooks
        processed_reply = self._process_human_input("" if not isinstance(reply, str) and iscoroutine(reply) else reply)
        if processed_reply is None:
            raise ValueError("safeguard_human_inputs hook returned None")

        self._human_input.append(processed_reply)
        return processed_reply

    async def a_get_human_input(
        self: "ConversableAgentBase",
        prompt: str,
        *,
        iostream: AsyncInputStream | None = None,
    ) -> str:
        """(Async) Get human input.

        Override this method to customize the way to get human input.

        Args:
            prompt (str): prompt for the human input.
            iostream (Optional[AsyncInputStream]): The AsyncInputStream object to use for sending messages.
        Returns:
            str: human input.
        """

        iostream = iostream or IOStream.get_default()
        input_func = iostream.input

        if is_coroutine_callable(input_func):
            reply = await input_func(prompt)
        else:
            reply = await asyncio.to_thread(input_func, prompt)
        # Process the human input through hooks
        processed_reply = self._process_human_input(reply)
        if processed_reply is None:
            raise ValueError("safeguard_human_inputs hook returned None")

        self._human_input.append(processed_reply)
        return processed_reply

    def check_termination_and_human_reply(
        self: "ConversableAgentBase",
        messages: list[dict[str, Any]] | None = None,
        sender: Agent | None = None,
        config: Any | None = None,
        iostream: IOStreamProtocol | None = None,
    ) -> tuple[bool, str | None]:
        """Check if the conversation should be terminated, and if human reply is provided.

        This method checks for conditions that require the conversation to be terminated, such as reaching
        a maximum number of consecutive auto-replies or encountering a termination message. Additionally,
        it prompts for and processes human input based on the configured human input mode, which can be
        'ALWAYS', 'NEVER', or 'TERMINATE'. The method also manages the consecutive auto-reply counter
        for the conversation and prints relevant messages based on the human input received.

        Args:
            messages: A list of message dictionaries, representing the conversation history.
            sender: The agent object representing the sender of the message.
            config: Configuration object, defaults to the current instance if not provided.
            iostream (Optional[IOStreamProtocol]): The IOStream object to use for sending messages.

        Returns:
            A tuple containing a boolean indicating if the conversation
            should be terminated, and a human reply which can be a string, a dictionary, or None.
        """
        iostream = iostream or IOStream.get_default()

        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender] if sender else []

        termination_reason = None

        # if there are no messages, continue the conversation
        if not messages:
            return False, None
        message = messages[-1]

        reply = ""
        no_human_input_msg = ""
        sender_name = "the sender" if sender is None else sender.name
        if self.human_input_mode == "ALWAYS":
            reply = self.get_human_input(
                f"Replying as {self.name}. Provide feedback to {sender_name}. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: ",
                iostream=iostream,
            )
            no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
            # if the human input is empty, and the message is a termination message, then we will terminate the conversation
            if not reply and self._is_termination_msg(message):
                termination_reason = f"Termination message condition on agent '{self.name}' met"
            elif reply == "exit":
                termination_reason = "User requested to end the conversation"

            reply = reply if reply or not self._is_termination_msg(message) else "exit"
        else:
            if self._consecutive_auto_reply_counter[sender] >= self._max_consecutive_auto_reply_dict[sender]:
                if self.human_input_mode == "NEVER":
                    termination_reason = "Maximum number of consecutive auto-replies reached"
                    reply = "exit"
                else:
                    # self.human_input_mode == "TERMINATE":
                    terminate = self._is_termination_msg(message)
                    reply = self.get_human_input(
                        f"Please give feedback to {sender_name}. Press enter or type 'exit' to stop the conversation: "
                        if terminate
                        else f"Please give feedback to {sender_name}. Press enter to skip and use auto-reply, or type 'exit' to stop the conversation: ",
                        iostream=iostream,
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    if reply != "exit" and terminate:
                        termination_reason = (
                            f"Termination message condition on agent '{self.name}' met and no human input provided"
                        )
                    elif reply == "exit":
                        termination_reason = "User requested to end the conversation"

                    reply = reply if reply or not terminate else "exit"
            elif self._is_termination_msg(message):
                if self.human_input_mode == "NEVER":
                    termination_reason = f"Termination message condition on agent '{self.name}' met"
                    reply = "exit"
                else:
                    # self.human_input_mode == "TERMINATE":
                    reply = self.get_human_input(
                        f"Please give feedback to {sender_name}. Press enter or type 'exit' to stop the conversation: ",
                        iostream=iostream,
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""

                    # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    if not reply or reply == "exit":
                        termination_reason = (
                            f"Termination message condition on agent '{self.name}' met and no human input provided"
                        )

                    reply = reply or "exit"

        # print the no_human_input_msg
        if no_human_input_msg:
            iostream.send(
                TerminationAndHumanReplyNoInputEvent(
                    no_human_input_msg=no_human_input_msg, sender=sender, recipient=self
                )
            )

        # stop the conversation
        if reply == "exit":
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender] = 0

            if termination_reason:
                iostream.send(TerminationEvent(termination_reason=termination_reason, sender=self, recipient=sender))

            return True, None

        # send the human reply
        if reply or self._max_consecutive_auto_reply_dict[sender] == 0:
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender] = 0
            # User provided a custom response, return function and tool failures indicating user interruption
            tool_returns = []
            if message.get("function_call", False):
                tool_returns.append({
                    "role": "function",
                    "name": message["function_call"].get("name", ""),
                    "content": "USER INTERRUPTED",
                })

            if message.get("tool_calls", False):
                tool_returns.extend([
                    {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": "USER INTERRUPTED"}
                    for tool_call in message["tool_calls"]
                ])

            response = {"role": "user", "content": reply}
            if tool_returns:
                response["tool_responses"] = tool_returns

            return True, response

        # increment the consecutive_auto_reply_counter
        self._consecutive_auto_reply_counter[sender] += 1
        if self.human_input_mode != "NEVER":
            iostream.send(UsingAutoReplyEvent(human_input_mode=self.human_input_mode, sender=sender, recipient=self))

        return False, None

    async def a_check_termination_and_human_reply(
        self: "ConversableAgentBase",
        messages: list[dict[str, Any]] | None = None,
        sender: Agent | None = None,
        config: Any | None = None,
        iostream: AsyncIOStreamProtocol | None = None,
    ) -> tuple[bool, str | None]:
        """(async) Check if the conversation should be terminated, and if human reply is provided.

        This method checks for conditions that require the conversation to be terminated, such as reaching
        a maximum number of consecutive auto-replies or encountering a termination message. Additionally,
        it prompts for and processes human input based on the configured human input mode, which can be
        'ALWAYS', 'NEVER', or 'TERMINATE'. The method also manages the consecutive auto-reply counter
        for the conversation and prints relevant messages based on the human input received.

        Args:
            messages (Optional[List[Dict]]): A list of message dictionaries, representing the conversation history.
            sender (Optional[Agent]): The agent object representing the sender of the message.
            config (Optional[Any]): Configuration object, defaults to the current instance if not provided.
            iostream (Optional[AsyncIOStreamProtocol]): The AsyncIOStreamProtocol object to use for sending messages.

        Returns:
            Tuple[bool, Union[str, Dict, None]]: A tuple containing a boolean indicating if the conversation
            should be terminated, and a human reply which can be a string, a dictionary, or None.
        """
        iostream = iostream or IOStream.get_default()

        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender] if sender else []

        termination_reason = None

        message = messages[-1] if messages else {}
        reply = ""
        no_human_input_msg = ""
        sender_name = "the sender" if sender is None else sender.name
        if self.human_input_mode == "ALWAYS":
            reply = await self.a_get_human_input(
                f"Replying as {self.name}. Provide feedback to {sender_name}. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: ",
                iostream=iostream,
            )
            no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
            # if the human input is empty, and the message is a termination message, then we will terminate the conversation
            if not reply and self._is_termination_msg(message):
                termination_reason = f"Termination message condition on agent '{self.name}' met"
            elif reply == "exit":
                termination_reason = "User requested to end the conversation"

            reply = reply if reply or not self._is_termination_msg(message) else "exit"
        else:
            if self._consecutive_auto_reply_counter[sender] >= self._max_consecutive_auto_reply_dict[sender]:
                if self.human_input_mode == "NEVER":
                    termination_reason = "Maximum number of consecutive auto-replies reached"
                    reply = "exit"
                else:
                    # self.human_input_mode == "TERMINATE":
                    terminate = self._is_termination_msg(message)
                    reply = await self.a_get_human_input(
                        f"Please give feedback to {sender_name}. Press enter or type 'exit' to stop the conversation: "
                        if terminate
                        else f"Please give feedback to {sender_name}. Press enter to skip and use auto-reply, or type 'exit' to stop the conversation: ",
                        iostream=iostream,
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    if reply != "exit" and terminate:
                        termination_reason = (
                            f"Termination message condition on agent '{self.name}' met and no human input provided"
                        )
                    elif reply == "exit":
                        termination_reason = "User requested to end the conversation"

                    reply = reply if reply or not terminate else "exit"
            elif self._is_termination_msg(message):
                if self.human_input_mode == "NEVER":
                    termination_reason = f"Termination message condition on agent '{self.name}' met"
                    reply = "exit"
                else:
                    # self.human_input_mode == "TERMINATE":
                    reply = await self.a_get_human_input(
                        f"Please give feedback to {sender_name}. Press enter or type 'exit' to stop the conversation: ",
                        iostream=iostream,
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""

                    # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    if not reply or reply == "exit":
                        termination_reason = (
                            f"Termination message condition on agent '{self.name}' met and no human input provided"
                        )

                    reply = reply or "exit"

        # print the no_human_input_msg
        if no_human_input_msg:
            iostream.send(
                TerminationAndHumanReplyNoInputEvent(
                    no_human_input_msg=no_human_input_msg, sender=sender, recipient=self
                )
            )

        # stop the conversation
        if reply == "exit":
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender] = 0

            if termination_reason:
                iostream.send(TerminationEvent(termination_reason=termination_reason, sender=self, recipient=sender))

            return True, None

        # send the human reply
        if reply or self._max_consecutive_auto_reply_dict[sender] == 0:
            # User provided a custom response, return function and tool results indicating user interruption
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender] = 0
            tool_returns = []
            if message.get("function_call", False):
                tool_returns.append({
                    "role": "function",
                    "name": message["function_call"].get("name", ""),
                    "content": "USER INTERRUPTED",
                })

            if message.get("tool_calls", False):
                tool_returns.extend([
                    {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": "USER INTERRUPTED"}
                    for tool_call in message["tool_calls"]
                ])

            response = {"role": "user", "content": reply}
            if tool_returns:
                response["tool_responses"] = tool_returns

            return True, response

        # increment the consecutive_auto_reply_counter
        self._consecutive_auto_reply_counter[sender] += 1
        if self.human_input_mode != "NEVER":
            iostream.send(UsingAutoReplyEvent(human_input_mode=self.human_input_mode, sender=sender, recipient=self))

        return False, None
