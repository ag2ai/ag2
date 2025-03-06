# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from .... import ConversableAgent
from ....doc_utils import export_module
from ....tools.contrib import TimeTool

__all__ = ["TimeToolAgent"]


@export_module("autogen.agents.contrib")  # API Reference: autogen > agents > contrib > TimeToolAgent
class TimeToolAgent(ConversableAgent):
    """A simple agent that returns the current time using tools

    Use it is as a reference for creating new agents with a tool-based approach (as opposed to reply-based).

    This agent will call the TimeTool and return the date and time whenever it needs to reply."""

    DEFAULT_SYSTEM_MESSAGE = (
        "You are a calendar agent that uses tools to return the date and time. "
        "When you reply, say 'Tick, tock, the current date/time is ' followed by the date and time in the exact format the tool provided."
    )

    def __init__(
        self,
        date_time_format: str = "%Y-%m-%d %H:%M:%S",  # This is a parameter that is unique to this agent
        **kwargs: Any,
    ) -> None:
        """Initialize the TimeToolAgent.

        Args:
            date_time_format: The format in which the date and time should be returned.
            **kwargs: Additional keyword arguments passed to the parent ConversableAgent class.
        """
        # Here we handle a ConversableAgent parameter through the kwargs
        # We will pass this through when we run init() the base class
        # Use this to tailor the return message
        system_message = kwargs.pop("system_message", self.DEFAULT_SYSTEM_MESSAGE)

        # Store the date and time format on the agent, prefixed with an underscore to indicate it's a private variable
        self._date_time_format = date_time_format

        self._time_tool = TimeTool(date_time_format=self._date_time_format)

        # Initialise the base class, passing through the system_message parameter
        super().__init__(system_message=system_message, **kwargs)

        self.register_for_llm()(self._time_tool)
