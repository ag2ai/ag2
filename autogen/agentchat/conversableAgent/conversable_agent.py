# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

from ...events.agent_events import (
    ClearConversableAgentHistoryEvent,
    ClearConversableAgentHistoryWarningEvent,
    ConversableAgentUsageSummaryEvent,
    ConversableAgentUsageSummaryNoCostIncurredEvent,
)
from ...io.base import IOStream
from ..chat import ChatResult
from ..group.guardrails import Guardrail
from ..group.on_condition import OnCondition
from ..group.on_context_condition import OnContextCondition
from .base import ConversableAgentBase
from .chat_management import ChatManagementMixin
from .code_execution import CodeExecutionMixin
from .conversable_utils import ConversableUtilsMixin
from .function_execution import FunctionExecutionMixin
from .hooks_and_registry import HooksAndRegistryMixin
from .llm_integration import LLMIntegrationMixin
from .massaging import MessagingMixin
from .reply_handlers import ReplyHandlersMixin

if TYPE_CHECKING:
    from ..agent import Agent


class ConversableAgent(
    ConversableAgentBase,
    MessagingMixin,
    ReplyHandlersMixin,
    FunctionExecutionMixin,
    CodeExecutionMixin,
    ChatManagementMixin,
    HooksAndRegistryMixin,
    LLMIntegrationMixin,
    ConversableUtilsMixin,
):
    """(In preview) A class for generic conversable agents which can be configured as assistant or user proxy.

    After receiving each message, the agent will send a reply to the sender unless the msg is a termination msg.
    For example, AssistantAgent and UserProxyAgent are subclasses of this class,
    configured with different default settings.

    To modify auto reply, override `generate_reply` method. \n
    To disable/enable human response in every turn, set `human_input_mode` to "NEVER" or "ALWAYS". \n
    To modify the way to get human input, override `get_human_input` method. \n
    To modify the way to execute code blocks, single code block, or function call, override `execute_code_blocks`, \n
    `run_code`, and `execute_function` methods respectively. \n
    """

    def get_chat_results(self, chat_index: int | None = None) -> list[ChatResult] | ChatResult:
        """A summary from the finished chats of particular agents."""
        if chat_index is not None:
            return self._finished_chats[chat_index]
        else:
            return self._finished_chats

    def reset(self) -> None:
        """Reset the agent."""
        self.clear_history()
        self.reset_consecutive_auto_reply_counter()
        self.stop_reply_at_receive()
        if self.client is not None:
            self.client.clear_usage_summary()
        for reply_func_tuple in self._reply_func_list:
            if reply_func_tuple["reset_config"] is not None:
                reply_func_tuple["reset_config"](reply_func_tuple["config"])
            else:
                reply_func_tuple["config"] = copy.copy(reply_func_tuple["init_config"])

    def stop_reply_at_receive(self, sender: "Agent" | None = None):
        """Reset the reply_at_receive of the sender."""
        if sender is None:
            self.reply_at_receive.clear()
        else:
            self.reply_at_receive[sender] = False

    def reset_consecutive_auto_reply_counter(self, sender: "Agent" | None = None):
        """Reset the consecutive_auto_reply_counter of the sender."""
        if sender is None:
            self._consecutive_auto_reply_counter.clear()
        else:
            self._consecutive_auto_reply_counter[sender] = 0

    def clear_history(self, recipient: "Agent" | None = None, nr_messages_to_preserve: int | None = None):
        """Clear the chat history of the agent.

        Args:
            recipient: the agent with whom the chat history to clear. If None, clear the chat history with all agents.
            nr_messages_to_preserve: the number of newest messages to preserve in the chat history.
        """
        iostream = IOStream.get_default()
        if recipient is None:
            no_messages_preserved = 0
            if nr_messages_to_preserve:
                for key in self._oai_messages:
                    nr_messages_to_preserve_internal = nr_messages_to_preserve
                    # if breaking history between function call and function response, save function call message
                    # additionally, otherwise openai will return error
                    first_msg_to_save = self._oai_messages[key][-nr_messages_to_preserve_internal]
                    if first_msg_to_save.get("role") == "assistant" and (
                        first_msg_to_save.get("function_call") or first_msg_to_save.get("tool_calls")
                    ):
                        nr_messages_to_preserve_internal += 1
                    self._oai_messages[key] = self._oai_messages[key][-nr_messages_to_preserve_internal:]
                    no_messages_preserved += nr_messages_to_preserve_internal
            else:
                self._oai_messages.clear()
            if no_messages_preserved > 0:
                iostream.send(
                    ClearConversableAgentHistoryWarningEvent(
                        recipient=self, no_messages_preserved=no_messages_preserved
                    )
                )
            else:
                iostream.send(ClearConversableAgentHistoryEvent(recipient=self))
        else:
            if recipient in self._oai_messages:
                if nr_messages_to_preserve:
                    nr_messages_to_preserve_internal = nr_messages_to_preserve
                    # if breaking history between function call and function response, save function call message
                    # additionally, otherwise openai will return error
                    first_msg_to_save = self._oai_messages[recipient][-nr_messages_to_preserve_internal]
                    if first_msg_to_save.get("role") == "assistant" and (
                        first_msg_to_save.get("function_call") or first_msg_to_save.get("tool_calls")
                    ):
                        nr_messages_to_preserve_internal += 1
                    self._oai_messages[recipient] = self._oai_messages[recipient][-nr_messages_to_preserve_internal:]
                else:
                    del self._oai_messages[recipient]

    def print_usage_summary(self, mode: str | list[str] = ["actual", "total"]) -> None:
        """Print the usage summary."""
        iostream = IOStream.get_default()
        if self.client is None:
            iostream.send(ConversableAgentUsageSummaryNoCostIncurredEvent(recipient=self))
        else:
            iostream.send(ConversableAgentUsageSummaryEvent(recipient=self))

        if self.client is not None:
            self.client.print_usage_summary(mode)

    def get_actual_usage(self) -> None | dict[str, int]:
        """Get the actual usage summary."""
        if self.client is None:
            return None
        else:
            return self.client.actual_usage_summary

    def get_total_usage(self) -> None | dict[str, int]:
        """Get the total usage summary."""
        if self.client is None:
            return None
        else:
            return self.client.total_usage_summary

    def register_handoff(self, condition: Union["OnContextCondition", "OnCondition"]) -> None:
        """Register a single handoff condition (OnContextCondition or OnCondition).

        Args:
            condition: The condition to add (OnContextCondition, OnCondition)
        """
        self.handoffs.add(condition)

    def register_handoffs(self, conditions: list[Union["OnContextCondition", "OnCondition"]]) -> None:
        """Register multiple handoff conditions (OnContextCondition or OnCondition).

        Args:
            conditions: List of conditions to add
        """
        for condition in conditions:
            self.register_handoff(condition)

    def register_input_guardrail(self, guardrail: "Guardrail") -> None:
        """Register a guardrail to be used for input validation.

        Args:
            guardrail: The guardrail to register.
        """
        self.input_guardrails.add(guardrail)

    def register_input_guardrails(self, guardrails: list["Guardrail"]) -> None:
        """Register multiple guardrails to be used for input validation.

        Args:
            guardrails: List of guardrails to register.
        """
        for guardrail in guardrails:
            self.register_input_guardrail(guardrail)

    def register_output_guardrail(self, guardrail: "Guardrail") -> None:
        """Register a guardrail to be used for output validation.

        Args:
            guardrail: The guardrail to register.
        """
        self.output_guardrails.add(guardrail)

    def register_output_guardrails(self, guardrails: list["Guardrail"]) -> None:
        """Register multiple guardrails to be used for output validation.

        Args:
            guardrails: List of guardrails to register.
        """
        for guardrail in guardrails:
            self.register_output_guardrail(guardrail)


# Standalone function for backward compatibility
def register_function(
    f: Callable[..., Any],
    *,
    caller: ConversableAgent,
    executor: ConversableAgent,
    name: str | None = None,
    description: str,
) -> None:
    """Register a function to be proposed by an agent and executed for an executor.

    This function can be used instead of function decorators `@ConversationAgent.register_for_llm` and
    `@ConversationAgent.register_for_execution`.

    Args:
        f: the function to be registered.
        caller: the agent calling the function, typically an instance of ConversableAgent.
        executor: the agent executing the function, typically an instance of UserProxy.
        name: name of the function. If None, the function name will be used (default: None).
        description: description of the function. The description is used by LLM to decode whether the function
            is called. Make sure the description is properly describing what the function does or it might not be
            called by LLM when needed.

    """
    f = caller.register_for_llm(name=name, description=description)(f)
    executor.register_for_execution(name=name)(f)
