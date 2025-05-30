# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from pydantic import BaseModel, Field

from .oai.client import OpenAIWrapper

if TYPE_CHECKING:
    from .llm_config import LLMConfig


class GuardrailResult(BaseModel):
    """Represents the outcome of a guardrail check."""

    activated: bool
    justification: str = Field(default="No justification provided")

    def __str__(self) -> str:
        return f"Guardrail Result: {self.activated}\nJustification: {self.justification}"


class GuardrailException(Exception):
    """Custom exception for guardrail violations."""

    def __init__(self, message: str, result: GuardrailResult):
        super().__init__(message)
        self.result = result

    def __str__(self) -> str:
        return f"{super().__str__()}\n{self.result}"


class Guardrail(ABC):
    """Abstract base class for guardrails."""

    def __init__(self, name: str, should_check_fn: Optional[Callable[..., bool]] = None) -> None:
        self.name = name
        self.should_check_fn = should_check_fn

    @abstractmethod
    def check(
        self,
        context: Union[str, list[dict[str, Any]]],
    ) -> GuardrailResult:
        """Checks the text against the guardrail and returns a GuardrailResult."""
        pass

    def should_check(
        self,
        context: Union[str, list[dict[str, Any]]],
    ) -> bool:
        """Determines whether the guardrail should be applied to the given text."""
        if self.should_check_fn:
            return self.should_check_fn(context)
        return True  # default to always checking

    def enforce(
        self,
        context: Union[str, list[dict[str, Any]]],
    ) -> None:
        """Runs check and raises GuardrailException if the result is not successful."""
        if not self.should_check(context):
            return
        result = self.check(context)
        if result.activated:
            raise GuardrailException(
                f"Guardrail '{self.name}' check failed.",
                result,
            )


class LLMGuardrail(Guardrail):
    """Guardrail that uses an LLM to check the context."""

    def __init__(
        self,
        name: str,
        check_message: str,
        llm_config: "LLMConfig",
        should_check_fn: Optional[Callable[..., bool]] = None,
    ) -> None:
        super().__init__(name, should_check_fn)
        self.check_message = check_message

        if not llm_config:
            raise ValueError("LLMConfig is required.")

        self.llm_config = llm_config
        setattr(self.llm_config, "response_format", GuardrailResult)
        self.client = OpenAIWrapper(**llm_config.model_dump())

    def check(
        self,
        context: Union[str, list[dict[str, Any]]],
    ) -> GuardrailResult:
        check_messages = [{"role": "system", "content": self.check_message}]
        if isinstance(context, str):
            check_messages.append({"role": "user", "content": context})
        elif isinstance(context, list):
            check_messages.extend(context)
        response = self.client.create(messages=check_messages)
        return self.client.extract_text_or_completion_object(response)[0]


class CustomGuardrail(Guardrail):
    """Custom guardrail that uses a user-defined function to check the context."""

    def __init__(
        self, name: str, check_fn: Callable[..., GuardrailResult], should_check_fn: Optional[Callable[..., bool]] = None
    ) -> None:
        super().__init__(name, should_check_fn)
        self.check_fn = check_fn

    def check(
        self,
        context: Union[str, list[dict[str, Any]]],
    ) -> GuardrailResult:
        return self.check_fn(context)
