# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from .... import Agent, ConversableAgent, OpenAIWrapper
from ....doc_utils import export_module

__all__ = ["TimeReplyAgent"]


@export_module("autogen.agents.contrib")  # API Reference: autogen > agents > contrib > TimeReplyAgent
class TimeReplyAgent(ConversableAgent):
    """A simple agent that returns the current time.

    Use it is as a reference for creating new agents with a reply-based approach (as opposed to tool-based).

    This agent will return the date and time whenever it needs to reply."""

    DEFAULT_SYSTEM_MESSAGE = "You are a calendar agent that just returns the date and time."

    def __init__(
        self,
        date_time_format: str = "%Y-%m-%d %H:%M:%S",  # This is a parameter that is unique to this agent
        output_prefix: str = "Tick, tock, the current date/time is ",
        **kwargs: Any,
    ) -> None:
        """Initialize the TimeReplyAgent.

        Args:
            date_time_format: The format in which the date and time should be returned.
            output_prefix: The prefix to add to the output message.
            **kwargs: Additional parameters to pass to the base
        """
        # Here we handle a ConversableAgent parameter through the kwargs
        # We will pass this through when we run init() the base class
        # Note: For this TimeReplyAgent, the LLM is not used so this won't affect the behavior of this agent
        system_message = kwargs.pop("system_message", self.DEFAULT_SYSTEM_MESSAGE)

        # Store the date and time format on the agent, prefixed with an underscore to indicate it's a private variable
        self._date_time_format = date_time_format

        self._output_prefix = output_prefix

        # Initialise the base class, passing through the system_message parameter
        super().__init__(system_message=system_message, **kwargs)

        # Our reply function.
        # This one is simple, but yours will be more complex and
        # may even contain another AG2 workflow inside it
        def get_date_time_reply(
            agent: ConversableAgent,
            messages: Optional[list[dict[str, Any]]] = None,
            sender: Optional[Agent] = None,
            config: Optional[OpenAIWrapper] = None,
        ) -> tuple[bool, dict[str, Any]]:
            from datetime import datetime

            now = datetime.now()

            # Format the date and time as a string (e.g., "2025-02-25 14:30:00")
            current_date_time = now.strftime(self._date_time_format)

            # Final reply, with the date/time as the message
            return True, {"content": f"{self._output_prefix}{current_date_time}."}

        # Register our reply function with the agent
        self.register_reply(
            trigger=[Agent, None],
            reply_func=get_date_time_reply,  # This is the function that will be called when the agent needs to reply
            remove_other_reply_funcs=True,  # Removing all other reply functions so only this one will be used
        )
