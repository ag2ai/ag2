# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

from pydantic import BaseModel, Field

from ...oai.client import OpenAIWrapper

if TYPE_CHECKING:
    from ...llm_config import LLMConfig
    from .targets.transition_target import TransitionTarget


class GuardrailResult(BaseModel):
    """Represents the outcome of a guardrail check."""

    activated: bool
    justification: str = Field(default="No justification provided")

    def __str__(self) -> str:
        return f"Guardrail Result: {self.activated}\nJustification: {self.justification}"

    @staticmethod
    def parse(text: str) -> "GuardrailResult":
        """Parses a JSON string into a GuardrailResult object.

        Args:
            text (str): The JSON string to parse.

        Returns:
            GuardrailResult: The parsed GuardrailResult object.
        """
        try:
            data = json.loads(text)
            return GuardrailResult(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse GuardrailResult from text: {text}") from e


class Guardrail(ABC):
    """Abstract base class for guardrails."""

    def __init__(
        self, name: str, condition: str, target: "TransitionTarget", activation_message: Optional[str] = None
    ) -> None:
        self.name = name
        self.condition = condition
        self.target = target
        self.activation_message = (
            activation_message if activation_message else f"Guardrail '{name}' has been activated."
        )

    @abstractmethod
    def check(
        self,
        context: Union[str, list[dict[str, Any]]],
    ) -> GuardrailResult:
        """Checks the text against the guardrail and returns a GuardrailResult.

        Args:
            context (Union[str, list[dict[str, Any]]]): The context to check against the guardrail.

        Returns:
            GuardrailResult: The result of the guardrail check.
        """
        pass


class LLMGuardrail(Guardrail):
    """Guardrail that uses an LLM to check the context."""

    def __init__(
        self,
        name: str,
        condition: str,
        target: "TransitionTarget",
        llm_config: "LLMConfig",
        activation_message: Optional[str] = None,
    ) -> None:
        super().__init__(name, condition, target, activation_message)

        if not llm_config:
            raise ValueError("LLMConfig is required.")

        self.llm_config = llm_config.deepcopy()
        setattr(self.llm_config, "response_format", GuardrailResult)
        self.client = OpenAIWrapper(**self.llm_config.model_dump())

        self.check_prompt = f"""You are a guardrail that checks if a condition is met in the conversation you are given.
You will activate the guardrail only if the condition is met.

**Condition: {self.condition}**"""

    def check(
        self,
        context: Union[str, list[dict[str, Any]]],
    ) -> GuardrailResult:
        """Checks the context against the guardrail using an LLM.

        Args:
            context (Union[str, list[dict[str, Any]]]): The context to check against the guardrail.

        Returns:
            GuardrailResult: The result of the guardrail check.
        """
        # Set the check prompt as the system message
        check_messages = [{"role": "system", "content": self.check_prompt}]
        # If context is a string, wrap it in a user message and append it
        if isinstance(context, str):
            check_messages.append({"role": "user", "content": context})
        # If context is a list of messages, append them
        elif isinstance(context, list):
            check_messages.extend(context)
        else:
            raise ValueError("Context must be a string or a list of messages.")
        # Call the LLM with the check messages
        response = self.client.create(messages=check_messages)
        return GuardrailResult.parse(response.choices[0].message.content)  # type: ignore


class RegexGuardrail(Guardrail):
    """Guardrail that checks the context against a regular expression."""

    def __init__(
        self,
        name: str,
        condition: str,
        target: "TransitionTarget",
        activation_message: Optional[str] = None,
    ) -> None:
        super().__init__(name, condition, target, activation_message)
        # Compile the regular expression condition
        try:
            self.regex = re.compile(condition)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{condition}': {str(e)}")

    def check(
        self,
        context: Union[str, list[dict[str, Any]]],
    ) -> GuardrailResult:
        """Checks the context against the guardrail using a regular expression.

        Args:
            context (Union[str, list[dict[str, Any]]]): The context to check against the guardrail.

        Returns:
            GuardrailResult: The result of the guardrail check.
        """
        # Create a list of the messages to check
        if isinstance(context, str):
            messages = [context]
        elif isinstance(context, list):
            messages = [message.get("content", "") for message in context]
        else:
            raise ValueError("Context must be a string or a list of messages.")

        # Check each message against the regex
        for message in messages:
            match = self.regex.search(message)
            # If a match is found, activate the guardrail and return the result
            if match:
                activated = True
                justification = f"Match found -> {match.group(0)}"
                return GuardrailResult(activated=activated, justification=justification)
        return GuardrailResult(activated=False, justification="No match found in the context.")
