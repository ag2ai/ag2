# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union

from pydantic import BaseModel, Field

from ...oai.client import OpenAIWrapper

if TYPE_CHECKING:
    from ...llm_config import LLMConfig


class GuardrailResult(BaseModel):
    """Represents the outcome of a guardrail check."""

    activated: bool
    justification: str = Field(default="No justification provided")

    def __str__(self) -> str:
        return f"Guardrail Result: {self.activated}\nJustification: {self.justification}"

    @staticmethod
    def parse(text: str) -> "GuardrailResult":
        """Parses a JSON string into a GuardrailResult object."""
        try:
            data = json.loads(text)
            return GuardrailResult(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse GuardrailResult from text: {text}") from e


class GuardrailError(Exception):
    """Custom exception for guardrail violations."""

    def __init__(self, message: str, result: GuardrailResult):
        super().__init__(message)
        self.result = result

    def __str__(self) -> str:
        return f"{super().__str__()}\n{self.result}"


class Guardrail(ABC):
    """Abstract base class for guardrails."""

    def __init__(self, name: str, condition: str, activation_message: str) -> None:
        self.name = name
        self.condition = condition
        self.activation_message = activation_message

    @abstractmethod
    def check(
        self,
        context: Union[str, list[dict[str, Any]]],
    ) -> GuardrailResult:
        """Checks the text against the guardrail and returns a GuardrailResult."""
        pass

    def enforce(
        self,
        context: Union[str, list[dict[str, Any]]],
    ) -> None:
        """Runs check and raises GuardrailError if the result is not successful."""
        result = self.check(context)
        if result.activated:
            raise GuardrailError(
                f"Guardrail '{self.name}' check failed.",
                result,
            )


class LLMGuardrail(Guardrail):
    """Guardrail that uses an LLM to check the context."""

    def __init__(
        self,
        name: str,
        condition: str,
        activation_message: str,
        llm_config: "LLMConfig",
    ) -> None:
        super().__init__(name, condition, activation_message)
        self.check_prompt = f"""You are a guardrail that checks if a condition is met in the conversation you are given.
You will activate the guardrail only if the condition is met.

**Condition: {condition}**"""

        if not llm_config:
            raise ValueError("LLMConfig is required.")

        self.llm_config = llm_config.deepcopy()
        setattr(self.llm_config, "response_format", GuardrailResult)
        self.client = OpenAIWrapper(**self.llm_config.model_dump())

    def check(
        self,
        context: Union[str, list[dict[str, Any]]],
    ) -> GuardrailResult:
        check_messages = [{"role": "system", "content": self.check_prompt}]
        if isinstance(context, str):
            check_messages.append({"role": "user", "content": context})
        elif isinstance(context, list):
            check_messages.extend(context)
        response = self.client.create(messages=check_messages)
        assert type(response.choices[0].message.content) is str
        return GuardrailResult.parse(response.choices[0].message.content)


class RegexGuardrail(Guardrail):
    """Guardrail that checks the context against a regular expression."""

    def __init__(
        self,
        name: str,
        condition: str,
        activation_message: str,
    ) -> None:
        super().__init__(name, condition, activation_message)
        self.regex = re.compile(condition)

    def check(
        self,
        context: Union[str, list[dict[str, Any]]],
    ) -> GuardrailResult:
        if isinstance(context, str):
            text = context
        elif isinstance(context, list):
            text = " ".join([msg.get("content", "") for msg in context])
        else:
            raise ValueError("Context must be a string or a list of messages.")

        match = self.regex.search(text)
        activated = bool(match)
        justification = f"Match found: {match.group(0)}" if match else "No match found"

        return GuardrailResult(activated=activated, justification=justification)
