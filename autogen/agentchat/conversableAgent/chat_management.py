# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import functools
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Optional

from ...cache.cache import AbstractCache, Cache
from ...events.agent_events import (
    ErrorEvent,
    PostCarryoverProcessingEvent,
    RunCompletionEvent,
    TerminationEvent,
)
from ...exception_utils import InvalidCarryOverTypeError
from ...fast_depends.utils import is_coroutine_callable
from ...io.base import IOStream
from ...io.run_response import (
    AsyncRunIterResponse,
    AsyncRunResponse,
    AsyncRunResponseProtocol,
    RunIterResponse,
    RunResponse,
    RunResponseProtocol,
)
from ...io.thread_io_stream import AsyncThreadIOStream, ThreadIOStream
from ...tools import Tool
from ..agent import Agent
from ..chat import (
    ChatResult,
    _post_process_carryover_item,
    _validate_recipients,
    a_initiate_chats,
    initiate_chats,
)
from ..utils import consolidate_chat_info, gather_usage_summary

if TYPE_CHECKING:
    from ...events.base_event import BaseEvent

if TYPE_CHECKING:
    from .base import ConversableAgentBase


class ChatManagementMixin:
    """Mixin class for chat initiation and management functionality"""

    def _prepare_chat(
        self: "ConversableAgentBase",
        recipient: "ConversableAgentBase",
        clear_history: bool,
        prepare_recipient: bool = True,
        reply_at_receive: bool = True,
    ) -> None:
        self.reset_consecutive_auto_reply_counter(recipient)
        self.reply_at_receive[recipient] = reply_at_receive
        if clear_history:
            self.clear_history(recipient)
            self._human_input = []
        if prepare_recipient:
            recipient._prepare_chat(self, clear_history, False, reply_at_receive)

    def _should_terminate_chat(
        self: "ConversableAgentBase", recipient: "ConversableAgentBase", message: dict[str, Any]
    ) -> bool:
        """
        Determines whether the chat should be terminated based on the message content
        and the recipient's termination condition.

        Args:
            recipient (ConversableAgent): The agent to check for termination condition.
            message (dict[str, Any]): The message dictionary to evaluate for termination.

        Returns:
            bool: True if the chat should be terminated, False otherwise.
        """
        return (
            isinstance(recipient, type(self))
            and isinstance(message.get("content"), str)
            and hasattr(recipient, "_is_termination_msg")
            and recipient._is_termination_msg(message)
        )

    def initiate_chat(
        self: "ConversableAgentBase",
        recipient: "ConversableAgentBase",
        clear_history: bool = True,
        silent: bool | None = False,
        cache: AbstractCache | None = None,
        max_turns: int | None = None,
        summary_method: str | Callable[..., Any] | None = "last_msg",
        summary_args: dict[str, Any] | None = {},
        message: dict[str, Any] | str | Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Initiate a chat with the recipient agent.

        Reset the consecutive auto reply counter.
        If `clear_history` is True, the chat history with the recipient agent will be cleared.


        Args:
            recipient: the recipient agent.
            clear_history (bool): whether to clear the chat history with the agent. Default is True.
            silent (bool or None): (Experimental) whether to print the messages for this conversation. Default is False.
            cache (AbstractCache or None): the cache client to be used for this conversation. Default is None.
            max_turns (int or None): the maximum number of turns for the chat between the two agents. One turn means one conversation round trip. Note that this is different from
                `max_consecutive_auto_reply` which is the maximum number of consecutive auto replies; and it is also different from `max_rounds` in GroupChat which is the maximum number of rounds in a group chat session.
                If max_turns is set to None, the chat will continue until a termination condition is met. Default is None.
            summary_method (str or callable): a method to get a summary from the chat. Default is DEFAULT_SUMMARY_METHOD, i.e., "last_msg".
                Supported strings are "last_msg" and "reflection_with_llm":
                    - when set to "last_msg", it returns the last message of the dialog as the summary.
                    - when set to "reflection_with_llm", it returns a summary extracted using an llm client.
                        `llm_config` must be set in either the recipient or sender.

                A callable summary_method should take the recipient and sender agent in a chat as input and return a string of summary. E.g.,

                ```python
                def my_summary_method(
                    sender: ConversableAgent,
                    recipient: ConversableAgent,
                    summary_args: dict,
                ):
                    return recipient.last_message(sender)["content"]
                ```
            summary_args (dict): a dictionary of arguments to be passed to the summary_method.
                One example key is "summary_prompt", and value is a string of text used to prompt a LLM-based agent (the sender or recipient agent) to reflect
                on the conversation and extract a summary when summary_method is "reflection_with_llm".
                The default summary_prompt is DEFAULT_SUMMARY_PROMPT, i.e., "Summarize takeaway from the conversation. Do not add any introductory phrases. If the intended request is NOT properly addressed, please point it out."
                Another available key is "summary_role", which is the role of the message sent to the agent in charge of summarizing. Default is "system".
            message (str, dict or Callable): the initial message to be sent to the recipient. Needs to be provided. Otherwise, input() will be called to get the initial message.
                - If a string or a dict is provided, it will be used as the initial message.        `generate_init_message` is called to generate the initial message for the agent based on this string and the context.
                    If dict, it may contain the following reserved fields (either content or tool_calls need to be provided).

                        1. "content": content of the message, can be None.
                        2. "function_call": a dictionary containing the function name and arguments. (deprecated in favor of "tool_calls")
                        3. "tool_calls": a list of dictionaries containing the function name and arguments.
                        4. "role": role of the message, can be "assistant", "user", "function".
                            This field is only needed to distinguish between "function" or "assistant"/"user".
                        5. "name": In most cases, this field is not needed. When the role is "function", this field is needed to indicate the function name.
                        6. "context" (dict): the context of the message, which will be passed to
                            `OpenAIWrapper.create`.

                - If a callable is provided, it will be called to get the initial message in the form of a string or a dict.
                    If the returned type is dict, it may contain the reserved fields mentioned above.

                    Example of a callable message (returning a string):

                    ```python
                    def my_message(
                        sender: ConversableAgent, recipient: ConversableAgent, context: dict
                    ) -> Union[str, Dict]:
                        carryover = context.get("carryover", "")
                        if isinstance(message, list):
                            carryover = carryover[-1]
                        final_msg = "Write a blogpost." + "\\nContext: \\n" + carryover
                        return final_msg
                    ```

                    Example of a callable message (returning a dict):

                    ```python
                    def my_message(
                        sender: ConversableAgent, recipient: ConversableAgent, context: dict
                    ) -> Union[str, Dict]:
                        final_msg = {}
                        carryover = context.get("carryover", "")
                        if isinstance(message, list):
                            carryover = carryover[-1]
                        final_msg["content"] = "Write a blogpost." + "\\nContext: \\n" + carryover
                        final_msg["context"] = {"prefix": "Today I feel"}
                        return final_msg
                    ```
            **kwargs: any additional information. It has the following reserved fields:
                - "carryover": a string or a list of string to specify the carryover information to be passed to this chat.
                    If provided, we will combine this carryover (by attaching a "context: " string and the carryover content after the message content) with the "message" content when generating the initial chat
                    message in `generate_init_message`.
                - "verbose": a boolean to specify whether to print the message and carryover in a chat. Default is False.

        Raises:
            RuntimeError: if any async reply functions are registered and not ignored in sync chat.

        Returns:
            ChatResult: an ChatResult object.
        """
        iostream = IOStream.get_default()

        cache = Cache.get_current_cache(cache)
        _chat_info = locals().copy()
        _chat_info["sender"] = self
        consolidate_chat_info(_chat_info, uniform_sender=self)
        for agent in [self, recipient]:
            agent._raise_exception_on_async_reply_functions()
            agent.previous_cache = agent.client_cache
            agent.client_cache = cache
        if isinstance(max_turns, int):
            self._prepare_chat(recipient, clear_history, reply_at_receive=False)
            is_termination = False
            for i in range(max_turns):
                # check recipient max consecutive auto reply limit
                if self._consecutive_auto_reply_counter[recipient] >= recipient._max_consecutive_auto_reply:
                    break
                if i == 0:
                    if isinstance(message, Callable):
                        msg2send = message(_chat_info["sender"], _chat_info["recipient"], kwargs)
                    else:
                        msg2send = self.generate_init_message(message, **kwargs)
                else:
                    last_message = self.chat_messages[recipient][-1]
                    if self._should_terminate_chat(recipient, last_message):
                        break
                    msg2send = self.generate_reply(messages=self.chat_messages[recipient], sender=recipient)
                if msg2send is None:
                    break
                self.send(msg2send, recipient, request_reply=True, silent=silent)
            else:  # No breaks in the for loop, so we have reached max turns
                iostream.send(
                    TerminationEvent(
                        termination_reason=f"Maximum turns ({max_turns}) reached", sender=self, recipient=recipient
                    )
                )
        else:
            self._prepare_chat(recipient, clear_history)
            if isinstance(message, Callable):
                msg2send = message(_chat_info["sender"], _chat_info["recipient"], kwargs)
            else:
                msg2send = self.generate_init_message(message, **kwargs)
            self.send(msg2send, recipient, silent=silent)
        summary = self._summarize_chat(
            summary_method,
            summary_args,
            recipient,
            cache=cache,
        )
        for agent in [self, recipient]:
            agent.client_cache = agent.previous_cache
            agent.previous_cache = None
        chat_result = ChatResult(
            chat_history=self.chat_messages[recipient],
            summary=summary,
            cost=gather_usage_summary([self, recipient]),
            human_input=self._human_input,
        )
        return chat_result

    def run(
        self: "ConversableAgentBase",
        recipient: Optional["ConversableAgentBase"] = None,
        clear_history: bool = True,
        silent: bool | None = False,
        cache: AbstractCache | None = None,
        max_turns: int | None = None,
        summary_method: str | Callable[..., Any] | None = "last_msg",
        summary_args: dict[str, Any] | None = {},
        message: dict[str, Any] | str | Callable[..., Any] | None = None,
        executor_kwargs: dict[str, Any] | None = None,
        tools: Tool | Iterable[Tool] | None = None,
        user_input: bool | None = False,
        msg_to: str | None = "agent",
        **kwargs: Any,
    ) -> RunResponseProtocol:
        iostream = ThreadIOStream()
        agents = [self, recipient] if recipient else [self]
        response = RunResponse(iostream, agents=agents)

        if recipient is None:

            def initiate_chat(
                self=self,
                iostream: ThreadIOStream = iostream,
                response: RunResponse = response,
            ) -> None:
                with (
                    IOStream.set_default(iostream),
                    self._create_or_get_executor(
                        executor_kwargs=executor_kwargs,
                        tools=tools,
                        agent_name="user",
                        agent_human_input_mode="ALWAYS" if user_input else "NEVER",
                    ) as executor,
                ):
                    try:
                        if msg_to == "agent":
                            chat_result = executor.initiate_chat(
                                self,
                                message=message,
                                clear_history=clear_history,
                                max_turns=max_turns,
                                summary_method=summary_method,
                            )
                        else:
                            chat_result = self.initiate_chat(
                                executor,
                                message=message,
                                clear_history=clear_history,
                                max_turns=max_turns,
                                summary_method=summary_method,
                            )

                        IOStream.get_default().send(
                            RunCompletionEvent(
                                history=chat_result.chat_history,
                                summary=chat_result.summary,
                                cost=chat_result.cost,
                                last_speaker=self.name,
                            )
                        )
                    except Exception as e:
                        response.iostream.send(ErrorEvent(error=e))

        else:

            def initiate_chat(
                self=self,
                iostream: ThreadIOStream = iostream,
                response: RunResponse = response,
            ) -> None:
                with IOStream.set_default(iostream):  # type: ignore[arg-type]
                    try:
                        chat_result = self.initiate_chat(
                            recipient,
                            clear_history=clear_history,
                            silent=silent,
                            cache=cache,
                            max_turns=max_turns,
                            summary_method=summary_method,
                            summary_args=summary_args,
                            message=message,
                            **kwargs,
                        )

                        response._summary = chat_result.summary
                        response._messages = chat_result.chat_history

                        _last_speaker = recipient if chat_result.chat_history[-1]["name"] == recipient.name else self
                        if hasattr(recipient, "last_speaker"):
                            _last_speaker = recipient.last_speaker

                        IOStream.get_default().send(
                            RunCompletionEvent(
                                history=chat_result.chat_history,
                                summary=chat_result.summary,
                                cost=chat_result.cost,
                                last_speaker=_last_speaker.name,
                            )
                        )
                    except Exception as e:
                        response.iostream.send(ErrorEvent(error=e))

        threading.Thread(
            target=initiate_chat,
        ).start()

        return response

    async def a_initiate_chat(
        self: "ConversableAgentBase",
        recipient: "ConversableAgentBase",
        clear_history: bool = True,
        silent: bool | None = False,
        cache: AbstractCache | None = None,
        max_turns: int | None = None,
        summary_method: str | Callable[..., Any] | None = "last_msg",
        summary_args: dict[str, Any] | None = {},
        message: str | Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """(async) Initiate a chat with the recipient agent.

        Reset the consecutive auto reply counter.
        If `clear_history` is True, the chat history with the recipient agent will be cleared.
        `a_generate_init_message` is called to generate the initial message for the agent.

        Args: Please refer to `initiate_chat`.

        Returns:
            ChatResult: an ChatResult object.
        """
        iostream = IOStream.get_default()

        _chat_info = locals().copy()
        _chat_info["sender"] = self
        consolidate_chat_info(_chat_info, uniform_sender=self)
        for agent in [self, recipient]:
            agent.previous_cache = agent.client_cache
            agent.client_cache = cache
        if isinstance(max_turns, int):
            self._prepare_chat(recipient, clear_history, reply_at_receive=False)
            is_termination = False
            for _ in range(max_turns):
                if _ == 0:
                    if isinstance(message, Callable):
                        msg2send = message(_chat_info["sender"], _chat_info["recipient"], kwargs)
                    else:
                        msg2send = await self.a_generate_init_message(message, **kwargs)
                else:
                    last_message = self.chat_messages[recipient][-1]
                    if self._should_terminate_chat(recipient, last_message):
                        break
                    msg2send = await self.a_generate_reply(messages=self.chat_messages[recipient], sender=recipient)
                    if msg2send is None:
                        break
                await self.a_send(msg2send, recipient, request_reply=True, silent=silent)
            else:  # No breaks in the for loop, so we have reached max turns
                iostream.send(
                    TerminationEvent(
                        termination_reason=f"Maximum turns ({max_turns}) reached", sender=self, recipient=recipient
                    )
                )
        else:
            self._prepare_chat(recipient, clear_history)
            if isinstance(message, Callable):
                msg2send = message(_chat_info["sender"], _chat_info["recipient"], kwargs)
            else:
                msg2send = await self.a_generate_init_message(message, **kwargs)
            await self.a_send(msg2send, recipient, silent=silent)
        summary = self._summarize_chat(
            summary_method,
            summary_args,
            recipient,
            cache=cache,
        )
        for agent in [self, recipient]:
            agent.client_cache = agent.previous_cache
            agent.previous_cache = None
        chat_result = ChatResult(
            chat_history=self.chat_messages[recipient],
            summary=summary,
            cost=gather_usage_summary([self, recipient]),
            human_input=self._human_input,
        )
        return chat_result

    async def a_run(
        self: "ConversableAgentBase",
        recipient: Optional["ConversableAgentBase"] = None,
        clear_history: bool = True,
        silent: bool | None = False,
        cache: AbstractCache | None = None,
        max_turns: int | None = None,
        summary_method: str | Callable[..., Any] | None = "last_msg",
        summary_args: dict[str, Any] | None = {},
        message: dict[str, Any] | str | Callable[..., Any] | None = None,
        executor_kwargs: dict[str, Any] | None = None,
        tools: Tool | Iterable[Tool] | None = None,
        user_input: bool | None = False,
        msg_to: str | None = "agent",
        **kwargs: Any,
    ) -> AsyncRunResponseProtocol:
        iostream = AsyncThreadIOStream()
        agents = [self, recipient] if recipient else [self]
        response = AsyncRunResponse(iostream, agents=agents)

        if recipient is None:

            async def initiate_chat(
                self=self,
                iostream: AsyncThreadIOStream = iostream,
                response: AsyncRunResponse = response,
            ) -> None:
                with (
                    IOStream.set_default(iostream),
                    self._create_or_get_executor(
                        executor_kwargs=executor_kwargs,
                        tools=tools,
                        agent_name="user",
                        agent_human_input_mode="ALWAYS" if user_input else "NEVER",
                    ) as executor,
                ):
                    try:
                        if msg_to == "agent":
                            chat_result = await executor.a_initiate_chat(
                                self,
                                message=message,
                                clear_history=clear_history,
                                max_turns=max_turns,
                                summary_method=summary_method,
                            )
                        else:
                            chat_result = await self.a_initiate_chat(
                                executor,
                                message=message,
                                clear_history=clear_history,
                                max_turns=max_turns,
                                summary_method=summary_method,
                            )

                        IOStream.get_default().send(
                            RunCompletionEvent(
                                history=chat_result.chat_history,
                                summary=chat_result.summary,
                                cost=chat_result.cost,
                                last_speaker=self.name,
                            )
                        )
                    except Exception as e:
                        response.iostream.send(ErrorEvent(error=e))

        else:

            async def initiate_chat(
                self=self,
                iostream: AsyncThreadIOStream = iostream,
                response: AsyncRunResponse = response,
            ) -> None:
                with IOStream.set_default(iostream):  # type: ignore[arg-type]
                    try:
                        chat_result = await self.a_initiate_chat(
                            recipient,
                            clear_history=clear_history,
                            silent=silent,
                            cache=cache,
                            max_turns=max_turns,
                            summary_method=summary_method,
                            summary_args=summary_args,
                            message=message,
                            **kwargs,
                        )

                        last_speaker = recipient if chat_result.chat_history[-1]["name"] == recipient.name else self
                        if hasattr(recipient, "last_speaker"):
                            last_speaker = recipient.last_speaker

                        IOStream.get_default().send(
                            RunCompletionEvent(
                                history=chat_result.chat_history,
                                summary=chat_result.summary,
                                cost=chat_result.cost,
                                last_speaker=last_speaker.name,
                            )
                        )

                    except Exception as e:
                        response.iostream.send(ErrorEvent(error=e))

        asyncio.create_task(initiate_chat())

        return response

    def _summarize_chat(
        self: "ConversableAgentBase",
        summary_method,
        summary_args,
        recipient: Agent | None = None,
        cache: AbstractCache | None = None,
    ) -> str:
        """Get a chat summary from an agent participating in a chat.

        Args:
            summary_method (str or callable): the summary_method to get the summary.
                The callable summary_method should take the recipient and sender agent in a chat as input and return a string of summary. E.g,
                ```python
                def my_summary_method(
                    sender: ConversableAgent,
                    recipient: ConversableAgent,
                    summary_args: dict,
                ):
                    return recipient.last_message(sender)["content"]
                ```
            summary_args (dict): a dictionary of arguments to be passed to the summary_method.
            recipient: the recipient agent in a chat.
            cache: the cache client to be used for this conversation. When provided,
                the cache will be used to store and retrieve LLM responses when generating
                summaries, which can improve performance and reduce API costs for
                repetitive summary requests. The cache is passed to the summary_method
                via summary_args['cache'].

        Returns:
            str: a chat summary from the agent.
        """
        summary = ""
        if summary_method is None:
            return summary
        if "cache" not in summary_args:
            summary_args["cache"] = cache
        if summary_method == "reflection_with_llm":
            summary_method = self._reflection_with_llm_as_summary
        elif summary_method == "last_msg":
            summary_method = self._last_msg_as_summary

        if isinstance(summary_method, Callable):
            summary = summary_method(self, recipient, summary_args)
        else:
            raise ValueError(
                "If not None, the summary_method must be a string from [`reflection_with_llm`, `last_msg`] or a callable."
            )
        if isinstance(summary, dict):
            summary = str(summary.get("content", ""))
        return summary

    @staticmethod
    def _last_msg_as_summary(sender, recipient, summary_args) -> str:
        """Get a chat summary from the last message of the recipient."""
        import warnings

        summary = ""
        try:
            content = recipient.last_message(sender)["content"]
            if isinstance(content, str):
                summary = content.replace("TERMINATE", "")
            elif isinstance(content, list):
                # Remove the `TERMINATE` word in the content list.
                summary = "\n".join(
                    x["text"].replace("TERMINATE", "") for x in content if isinstance(x, dict) and "text" in x
                )
        except (IndexError, AttributeError) as e:
            warnings.warn(f"Cannot extract summary using last_msg: {e}. Using an empty str as summary.", UserWarning)
        return summary

    @staticmethod
    def _reflection_with_llm_as_summary(sender, recipient, summary_args):
        import warnings

        prompt = summary_args.get("summary_prompt")
        prompt = (
            "Summarize the takeaway from the conversation. Do not add any introductory phrases."
            if prompt is None
            else prompt
        )
        if not isinstance(prompt, str):
            raise ValueError("The summary_prompt must be a string.")
        msg_list = recipient.chat_messages_for_summary(sender)
        agent = sender if recipient is None else recipient
        role = summary_args.get("summary_role", None)
        if role and not isinstance(role, str):
            raise ValueError("The summary_role in summary_arg must be a string.")
        try:
            summary = sender._reflection_with_llm(
                prompt, msg_list, llm_agent=agent, cache=summary_args.get("cache"), role=role
            )
        except Exception as e:
            warnings.warn(
                f"Cannot extract summary using reflection_with_llm: {e}. Using an empty str as summary.", UserWarning
            )
            summary = ""
        return summary

    def _reflection_with_llm(
        self: "ConversableAgentBase",
        prompt,
        messages,
        llm_agent: Agent | None = None,
        cache: AbstractCache | None = None,
        role: str | None = None,
    ) -> str:
        """Get a chat summary using reflection with an llm client based on the conversation history.

        Args:
            prompt (str): The prompt (in this method it is used as system prompt) used to get the summary.
            messages (list): The messages generated as part of a chat conversation.
            llm_agent: the agent with an llm client.
            cache (AbstractCache or None): the cache client to be used for this conversation.
            role (str): the role of the message, usually "system" or "user". Default is "system".
        """
        if not role:
            role = "system"

        system_msg = [
            {
                "role": role,
                "content": prompt,
            }
        ]

        messages = messages + system_msg
        if llm_agent and llm_agent.client is not None:
            llm_client = llm_agent.client
        elif self.client is not None:
            llm_client = self.client
        else:
            raise ValueError("No OpenAIWrapper client is found.")
        response = self._generate_oai_reply_from_client(llm_client=llm_client, messages=messages, cache=cache)
        return response

    @staticmethod
    def _get_chats_to_run(
        chat_queue: list[dict[str, Any]],
        recipient: Agent,
        messages: list[dict[str, Any]] | None,
        sender: Agent,
        config: Any,
    ) -> list[dict[str, Any]]:
        """A simple chat reply function.
        This function initiate one or a sequence of chats between the "recipient" and the agents in the
        chat_queue.

        It extracts and returns a summary from the nested chat based on the "summary_method" in each chat in chat_queue.

        Returns:
            Tuple[bool, str]: A tuple where the first element indicates the completion of the chat, and the second element contains the summary of the last chat if any chats were initiated.
        """
        last_msg = messages[-1].get("content")
        chat_to_run = []
        for i, c in enumerate(chat_queue):
            current_c = c.copy()
            if current_c.get("sender") is None:
                current_c["sender"] = recipient
            message = current_c.get("message")
            # If message is not provided in chat_queue, we by default use the last message from the original chat history as the first message in this nested chat (for the first chat in the chat queue).
            # NOTE: This setting is prone to change.
            if message is None and i == 0:
                message = last_msg
            if callable(message):
                message = message(recipient, messages, sender, config)
            # We only run chat that has a valid message. NOTE: This is prone to change depending on applications.
            if message:
                current_c["message"] = message
                chat_to_run.append(current_c)
        return chat_to_run

    @staticmethod
    def _process_nested_chat_carryover(
        chat: dict[str, Any],
        recipient: Agent,
        messages: list[dict[str, Any]],
        sender: Agent,
        config: Any,
        trim_n_messages: int = 0,
    ) -> None:
        """Process carryover messages for a nested chat (typically for the first chat of a group chat)

        The carryover_config key is a dictionary containing:
            "summary_method": The method to use to summarise the messages, can be "all", "last_msg", "reflection_with_llm" or a Callable
            "summary_args": Optional arguments for the summary method

        Supported carryover 'summary_methods' are:
            "all" - all messages will be incorporated
            "last_msg" - the last message will be incorporated
            "reflection_with_llm" - an llm will summarise all the messages and the summary will be incorporated as a single message
            Callable - a callable with the signature: my_method(agent: ConversableAgent, messages: List[Dict[str, Any]]) -> str

        Args:
            chat: The chat dictionary containing the carryover configuration
            recipient: The recipient agent
            messages: The messages from the parent chat
            sender: The sender agent
            config: The LLM configuration
            trim_n_messages: The number of latest messages to trim from the messages list
        """
        def concat_carryover(chat_message: str, carryover_message: str | list[dict[str, Any]]) -> str:
            """Concatenate the carryover message to the chat message."""
            prefix = f"{chat_message}\n" if chat_message else ""
            if isinstance(carryover_message, str):
                content = carryover_message
            elif isinstance(carryover_message, list):
                content = "\n".join(
                    msg["content"] for msg in carryover_message if "content" in msg and msg["content"] is not None
                )
            else:
                raise ValueError("Carryover message must be a string or a list of dictionaries")
            return f"{prefix}Context:\n{content}"

        carryover_config = chat["carryover_config"]
        if "summary_method" not in carryover_config:
            raise ValueError("Carryover configuration must contain a 'summary_method' key")
        carryover_summary_method = carryover_config["summary_method"]
        carryover_summary_args = carryover_config.get("summary_args") or {}
        chat_message = ""
        message = chat.get("message")
        if message:
            chat_message = message(recipient, messages, sender, config) if callable(message) else message
        content_messages = copy.deepcopy(messages)
        content_messages = content_messages[:-trim_n_messages]

        if carryover_summary_method == "all":
            carry_over_message = concat_carryover(chat_message, content_messages)
        elif carryover_summary_method == "last_msg":
            carry_over_message = concat_carryover(chat_message, content_messages[-1]["content"])
        elif carryover_summary_method == "reflection_with_llm":
            chat["recipient"]._oai_messages[sender] = content_messages
            carry_over_message_llm = ChatManagementMixin._reflection_with_llm_as_summary(
                sender=sender,
                recipient=chat["recipient"],
                summary_args=carryover_summary_args,
            )
            recipient._oai_messages[sender] = []
            carry_over_message = concat_carryover(chat_message, carry_over_message_llm)
        elif isinstance(carryover_summary_method, Callable):
            carry_over_message_result = carryover_summary_method(recipient, content_messages, carryover_summary_args)
            carry_over_message = concat_carryover(chat_message, carry_over_message_result)
        chat["message"] = carry_over_message

    def _process_carryover(self: "ConversableAgentBase", content: str, kwargs: dict) -> str:
        # Makes sure there's a carryover
        if not kwargs.get("carryover"):
            return content

        # if carryover is string
        if isinstance(kwargs["carryover"], str):
            content += "\nContext: \n" + kwargs["carryover"]
        elif isinstance(kwargs["carryover"], list):
            content += "\nContext: \n" + ("\n").join([_post_process_carryover_item(t) for t in kwargs["carryover"]])
        else:
            raise InvalidCarryOverTypeError(
                "Carryover should be a string or a list of strings. Not adding carryover to the message."
            )
        return content

    def run_iter(
        self: "ConversableAgentBase",
        recipient: Optional["ConversableAgentBase"] = None,
        clear_history: bool = True,
        silent: bool | None = False,
        cache: AbstractCache | None = None,
        max_turns: int | None = None,
        summary_method: str | Callable[..., Any] | None = None,
        summary_args: dict[str, Any] | None = {},
        message: dict[str, Any] | str | Callable[..., Any] | None = None,
        executor_kwargs: dict[str, Any] | None = None,
        tools: Tool | Iterable[Tool] | None = None,
        user_input: bool | None = False,
        msg_to: str | None = "agent",
        yield_on: Sequence[type["BaseEvent"]] | None = None,
        **kwargs: Any,
    ) -> RunIterResponse:
        """Run a chat with iterator-based stepped execution."""
        if summary_method is None:
            summary_method = self.DEFAULT_SUMMARY_METHOD
        agents = [self, recipient] if recipient else [self]

        def create_thread(iostream: ThreadIOStream) -> threading.Thread:
            if recipient is None:
                def initiate_chat() -> None:
                    with (
                        IOStream.set_default(iostream),
                        self._create_or_get_executor(
                            executor_kwargs=executor_kwargs,
                            tools=tools,
                            agent_name="user",
                            agent_human_input_mode="ALWAYS" if user_input else "NEVER",
                        ) as executor,
                    ):
                        try:
                            if msg_to == "agent":
                                chat_result = executor.initiate_chat(
                                    self,
                                    message=message,
                                    clear_history=clear_history,
                                    max_turns=max_turns,
                                    summary_method=summary_method,
                                )
                            else:
                                chat_result = self.initiate_chat(
                                    executor,
                                    message=message,
                                    clear_history=clear_history,
                                    max_turns=max_turns,
                                    summary_method=summary_method,
                                )
                            IOStream.get_default().send(
                                RunCompletionEvent(
                                    history=chat_result.chat_history,
                                    summary=chat_result.summary,
                                    cost=chat_result.cost,
                                    last_speaker=self.name,
                                )
                            )
                        except Exception as e:
                            iostream.send(ErrorEvent(error=e))
            else:
                def initiate_chat() -> None:
                    with IOStream.set_default(iostream):
                        try:
                            chat_result = self.initiate_chat(
                                recipient,
                                clear_history=clear_history,
                                silent=silent,
                                cache=cache,
                                max_turns=max_turns,
                                summary_method=summary_method,
                                summary_args=summary_args,
                                message=message,
                                **kwargs,
                            )
                            _last_speaker = (
                                recipient if chat_result.chat_history[-1]["name"] == recipient.name else self
                            )
                            if hasattr(recipient, "last_speaker"):
                                _last_speaker = recipient.last_speaker
                            IOStream.get_default().send(
                                RunCompletionEvent(
                                    history=chat_result.chat_history,
                                    summary=chat_result.summary,
                                    cost=chat_result.cost,
                                    last_speaker=_last_speaker.name,
                                )
                            )
                        except Exception as e:
                            iostream.send(ErrorEvent(error=e))
            return threading.Thread(target=initiate_chat, daemon=True)

        return RunIterResponse(
            start_thread_func=create_thread,
            yield_on=yield_on,
            agents=agents,
        )

    def a_run_iter(
        self: "ConversableAgentBase",
        recipient: Optional["ConversableAgentBase"] = None,
        clear_history: bool = True,
        silent: bool | None = False,
        cache: AbstractCache | None = None,
        max_turns: int | None = None,
        summary_method: str | Callable[..., Any] | None = None,
        summary_args: dict[str, Any] | None = {},
        message: dict[str, Any] | str | Callable[..., Any] | None = None,
        executor_kwargs: dict[str, Any] | None = None,
        tools: Tool | Iterable[Tool] | None = None,
        user_input: bool | None = False,
        msg_to: str | None = "agent",
        yield_on: Sequence[type["BaseEvent"]] | None = None,
        **kwargs: Any,
    ) -> AsyncRunIterResponse:
        """Async version of run_iter() for async contexts."""
        if summary_method is None:
            summary_method = self.DEFAULT_SUMMARY_METHOD
        agents = [self, recipient] if recipient else [self]

        def create_thread(iostream: ThreadIOStream) -> threading.Thread:
            if recipient is None:
                async def async_initiate_chat() -> None:
                    with (
                        IOStream.set_default(iostream),
                        self._create_or_get_executor(
                            executor_kwargs=executor_kwargs,
                            tools=tools,
                            agent_name="user",
                            agent_human_input_mode="ALWAYS" if user_input else "NEVER",
                        ) as executor,
                    ):
                        if msg_to == "agent":
                            chat_result = await executor.a_initiate_chat(
                                self,
                                message=message,
                                clear_history=clear_history,
                                max_turns=max_turns,
                                summary_method=summary_method,
                            )
                        else:
                            chat_result = await self.a_initiate_chat(
                                executor,
                                message=message,
                                clear_history=clear_history,
                                max_turns=max_turns,
                                summary_method=summary_method,
                            )
                        iostream.send(
                            RunCompletionEvent(
                                history=chat_result.chat_history,
                                summary=chat_result.summary,
                                cost=chat_result.cost,
                                last_speaker=self.name,
                            )
                        )

                def run_in_thread() -> None:
                    with IOStream.set_default(iostream):
                        try:
                            asyncio.run(async_initiate_chat())
                        except Exception as e:
                            iostream.send(ErrorEvent(error=e))
            else:
                async def async_initiate_chat() -> None:
                    chat_result = await self.a_initiate_chat(
                        recipient,
                        clear_history=clear_history,
                        silent=silent,
                        cache=cache,
                        max_turns=max_turns,
                        summary_method=summary_method,
                        summary_args=summary_args,
                        message=message,
                        **kwargs,
                    )
                    last_speaker = recipient if chat_result.chat_history[-1]["name"] == recipient.name else self
                    if hasattr(recipient, "last_speaker"):
                        last_speaker = recipient.last_speaker
                    iostream.send(
                        RunCompletionEvent(
                            history=chat_result.chat_history,
                            summary=chat_result.summary,
                            cost=chat_result.cost,
                            last_speaker=last_speaker.name,
                        )
                    )

                def run_in_thread() -> None:
                    with IOStream.set_default(iostream):
                        try:
                            asyncio.run(async_initiate_chat())
                        except Exception as e:
                            iostream.send(ErrorEvent(error=e))
            return threading.Thread(target=run_in_thread, daemon=True)

        return AsyncRunIterResponse(
            start_thread_func=create_thread,
            yield_on=yield_on,
            agents=agents,
        )

    def initiate_chats(self: "ConversableAgentBase", chat_queue: list[dict[str, Any]]) -> list[ChatResult]:
        """(Experimental) Initiate chats with multiple agents."""
        _chat_queue = self._check_chat_queue_for_sender(chat_queue)
        self._finished_chats = initiate_chats(_chat_queue)
        return self._finished_chats

    async def a_initiate_chats(self: "ConversableAgentBase", chat_queue: list[dict[str, Any]]) -> dict[int, ChatResult]:
        _chat_queue = self._check_chat_queue_for_sender(chat_queue)
        self._finished_chats = await a_initiate_chats(_chat_queue)
        return self._finished_chats

    def sequential_run(
        self: "ConversableAgentBase",
        chat_queue: list[dict[str, Any]],
    ) -> list[RunResponseProtocol]:
        """(Experimental) Initiate chats with multiple agents sequentially."""
        iostreams = [ThreadIOStream() for _ in range(len(chat_queue))]
        responses = [RunResponse(iostream, agents=[]) for iostream in iostreams]

        def _initiate_chats(
            iostreams: list[ThreadIOStream] = iostreams,
            responses: list[RunResponseProtocol] = responses,
        ) -> None:
            response = responses[0]
            try:
                _chat_queue = self._check_chat_queue_for_sender(chat_queue)
                consolidate_chat_info(_chat_queue)
                _validate_recipients(_chat_queue)
                finished_chats = []
                for chat_info, response, iostream in zip(_chat_queue, responses, iostreams):
                    with IOStream.set_default(iostream):
                        _chat_carryover = chat_info.get("carryover", [])
                        finished_chat_indexes_to_exclude_from_carryover = chat_info.get(
                            "finished_chat_indexes_to_exclude_from_carryover", []
                        )
                        if isinstance(_chat_carryover, str):
                            _chat_carryover = [_chat_carryover]
                        chat_info["carryover"] = _chat_carryover + [
                            r.summary
                            for i, r in enumerate(finished_chats)
                            if i not in finished_chat_indexes_to_exclude_from_carryover
                        ]
                        if not chat_info.get("silent", False):
                            IOStream.get_default().send(PostCarryoverProcessingEvent(chat_info=chat_info))
                        sender = chat_info["sender"]
                        chat_res = sender.initiate_chat(**chat_info)
                        IOStream.get_default().send(
                            RunCompletionEvent(
                                history=chat_res.chat_history,
                                summary=chat_res.summary,
                                cost=chat_res.cost,
                                last_speaker=(self if chat_res.chat_history[-1]["name"] == self.name else sender).name,
                            )
                        )
                        finished_chats.append(chat_res)
            except Exception as e:
                response.iostream.send(ErrorEvent(error=e))

        threading.Thread(target=_initiate_chats, daemon=True).start()
        return responses

    async def a_sequential_run(
        self: "ConversableAgentBase",
        chat_queue: list[dict[str, Any]],
    ) -> list[AsyncRunResponseProtocol]:
        """(Experimental) Initiate chats with multiple agents sequentially."""
        iostreams = [AsyncThreadIOStream() for _ in range(len(chat_queue))]
        responses = [AsyncRunResponse(iostream, agents=[]) for iostream in iostreams]

        async def _a_initiate_chats(
            iostreams: list[AsyncThreadIOStream] = iostreams,
            responses: list[AsyncRunResponseProtocol] = responses,
        ) -> None:
            response = responses[0]
            try:
                _chat_queue = self._check_chat_queue_for_sender(chat_queue)
                consolidate_chat_info(_chat_queue)
                _validate_recipients(_chat_queue)
                finished_chats = []
                for chat_info, response, iostream in zip(_chat_queue, responses, iostreams):
                    with IOStream.set_default(iostream):
                        _chat_carryover = chat_info.get("carryover", [])
                        finished_chat_indexes_to_exclude_from_carryover = chat_info.get(
                            "finished_chat_indexes_to_exclude_from_carryover", []
                        )
                        if isinstance(_chat_carryover, str):
                            _chat_carryover = [_chat_carryover]
                        chat_info["carryover"] = _chat_carryover + [
                            r.summary
                            for i, r in enumerate(finished_chats)
                            if i not in finished_chat_indexes_to_exclude_from_carryover
                        ]
                        if not chat_info.get("silent", False):
                            iostream.send(PostCarryoverProcessingEvent(chat_info=chat_info))
                        sender = chat_info["sender"]
                        chat_res = await sender.a_initiate_chat(**chat_info)
                        iostream.send(
                            RunCompletionEvent(
                                history=chat_res.chat_history,
                                summary=chat_res.summary,
                                cost=chat_res.cost,
                                last_speaker=(self if chat_res.chat_history[-1]["name"] == self.name else sender).name,
                            )
                        )
                        finished_chats.append(chat_res)
            except Exception as e:
                iostream.send(ErrorEvent(error=e))

        asyncio.create_task(_a_initiate_chats())
        return responses

    def _check_chat_queue_for_sender(self: "ConversableAgentBase", chat_queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check the chat queue and add the "sender" key if it's missing."""
        chat_queue_with_sender = []
        for chat_info in chat_queue:
            if chat_info.get("sender") is None:
                chat_info["sender"] = self
            chat_queue_with_sender.append(chat_info)
        return chat_queue_with_sender

    @staticmethod
    def _process_chat_queue_carryover(
        chat_queue: list[dict[str, Any]],
        recipient: Agent,
        messages: str | Callable[..., Any],
        sender: Agent,
        config: Any,
        trim_messages: int = 2,
    ) -> tuple[bool, str | None]:
        """Process carryover configuration for the first chat in the queue."""
        restore_chat_queue_message = False
        original_chat_queue_message = None
        if len(chat_queue) > 0 and "carryover_config" in chat_queue[0]:
            if "message" in chat_queue[0]:
                restore_chat_queue_message = True
                original_chat_queue_message = chat_queue[0]["message"]
            ChatManagementMixin._process_nested_chat_carryover(
                chat=chat_queue[0],
                recipient=recipient,
                messages=messages,
                sender=sender,
                config=config,
                trim_n_messages=trim_messages,
            )
        return restore_chat_queue_message, original_chat_queue_message

    @staticmethod
    def _summary_from_nested_chats(
        chat_queue: list[dict[str, Any]],
        recipient: Agent,
        messages: list[dict[str, Any]] | None,
        sender: Agent,
        config: Any,
    ) -> tuple[bool, str | None]:
        """A simple chat reply function that initiates nested chats and returns summary."""
        restore_chat_queue_message, original_chat_queue_message = ChatManagementMixin._process_chat_queue_carryover(
            chat_queue, recipient, messages, sender, config
        )
        chat_to_run = ChatManagementMixin._get_chats_to_run(chat_queue, recipient, messages, sender, config)
        if not chat_to_run:
            return True, None
        res = initiate_chats(chat_to_run)
        if restore_chat_queue_message:
            chat_queue[0]["message"] = original_chat_queue_message
        return True, res[-1].summary

    @staticmethod
    async def _a_summary_from_nested_chats(
        chat_queue: list[dict[str, Any]],
        recipient: Agent,
        messages: list[dict[str, Any]] | None,
        sender: Agent,
        config: Any,
    ) -> tuple[bool, str | None]:
        """Async version of _summary_from_nested_chats."""
        restore_chat_queue_message, original_chat_queue_message = ChatManagementMixin._process_chat_queue_carryover(
            chat_queue, recipient, messages, sender, config
        )
        chat_to_run = ChatManagementMixin._get_chats_to_run(chat_queue, recipient, messages, sender, config)
        if not chat_to_run:
            return True, None
        res = await a_initiate_chats(chat_to_run)
        index_of_last_chat = chat_to_run[-1]["chat_id"]
        if restore_chat_queue_message:
            chat_queue[0]["message"] = original_chat_queue_message
        return True, res[index_of_last_chat].summary

    def register_nested_chats(
        self: "ConversableAgentBase",
        chat_queue: list[dict[str, Any]],
        trigger: type[Agent] | str | Agent | Callable[[Agent], bool] | list,
        reply_func_from_nested_chats: str | Callable[..., Any] = "summary_from_nested_chats",
        position: int = 2,
        use_async: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Register a nested chat reply function."""
        if use_async:
            for chat in chat_queue:
                if chat.get("chat_id") is None:
                    raise ValueError("chat_id is required for async nested chats")
        if use_async:
            if reply_func_from_nested_chats == "summary_from_nested_chats":
                reply_func_from_nested_chats = self._a_summary_from_nested_chats
            if not callable(reply_func_from_nested_chats) or not is_coroutine_callable(reply_func_from_nested_chats):
                raise ValueError("reply_func_from_nested_chats must be a callable and a coroutine")
            async def wrapped_reply_func(recipient, messages=None, sender=None, config=None):
                return await reply_func_from_nested_chats(chat_queue, recipient, messages, sender, config)
        else:
            if reply_func_from_nested_chats == "summary_from_nested_chats":
                reply_func_from_nested_chats = self._summary_from_nested_chats
            if not callable(reply_func_from_nested_chats):
                raise ValueError("reply_func_from_nested_chats must be a callable")
            def wrapped_reply_func(recipient, messages=None, sender=None, config=None):
                return reply_func_from_nested_chats(chat_queue, recipient, messages, sender, config)
        functools.update_wrapper(wrapped_reply_func, reply_func_from_nested_chats)
        self.register_reply(
            trigger,
            wrapped_reply_func,
            position,
            kwargs.get("config"),
            kwargs.get("reset_config"),
            ignore_async_in_sync_chat=(
                not use_async if use_async is not None else kwargs.get("ignore_async_in_sync_chat")
            ),
        )

    def get_chat_results(self: "ConversableAgentBase", chat_index: int | None = None) -> list[ChatResult] | ChatResult:
        """A summary from the finished chats of particular agents."""
        if chat_index is not None:
            return self._finished_chats[chat_index]
        else:
            return self._finished_chats
