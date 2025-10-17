# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, TypedDict

from autogen import ConversableAgent, ModelClient


class TestAgent:
    """A context manager for testing ConversableAgent instances with predefined messages.

    This class allows you to temporarily replace an agent's LLM client with a fake client
    that returns predefined messages. It's useful for testing agent behavior without
    making actual API calls.

    Attributes:
        agent (ConversableAgent): The agent to be tested.
        messages (Iterable[str  |  dict[str, Any]]): An iterable of messages to be returned by the fake client.
        suppress_messages_end (bool): Whether to suppress StopIteration exceptions from the fake client.

    Example:
        >>> with TestAgent(agent, ["Hello", "How are you?"]) as test_agent:
        ...     # Agent will respond with "Hello" then "How are you?"
        ...     pass
    """

    def __init__(
        self,
        agent: ConversableAgent,
        messages: Iterable[str | dict[str, Any]] = (),
        *,
        suppress_messages_end: bool = False,
    ) -> None:
        self.agent = agent

        self.__original_human_input = self.agent.human_input_mode

        self.__original_client = agent.client
        self.__fake_client = FakeClient(messages)

        self.suppress_messages_end = suppress_messages_end

    def __enter__(self) -> None:
        self.agent.human_input_mode = "NEVER"

        self.__original_client = self.agent.client
        self.agent.client = self.__fake_client  # type: ignore[assignment]
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None | bool:
        self.agent.human_input_mode = self.__original_human_input

        self.agent.client = self.__original_client

        if isinstance(exc_value, StopIteration):
            # suppress fake client iterator ending
            return self.suppress_messages_end
        return None


class FakeClient:
    def __init__(self, messages: Iterable[str | dict[str, Any]]) -> None:
        # do not unpack messages to allow endless generators pass
        self.choice_iterator = iter(map(convert_fake_message, messages))

        self.total_usage_summary = None
        self.actual_usage_summary = None

    def create(self, **params: Any) -> ModelClient.ModelClientResponseProtocol:
        choice = next(self.choice_iterator)
        return FakeClientResponse(choices=[choice])

    def extract_text_or_completion_object(
        self,
        response: "FakeClientResponse",
    ) -> list[str] | list["FakeMessage"]:
        return response.message_retrieval_function()


def convert_fake_message(message: str | dict[str, Any]) -> "FakeChoice":
    if isinstance(message, str):
        return FakeChoice({"content": message})
    else:
        return FakeChoice({"role": "assistant", **message})  # type: ignore[typeddict-item]


class FakeMessage(TypedDict):
    content: str | dict[str, Any]


@dataclass
class FakeChoice(ModelClient.ModelClientResponseProtocol.Choice):
    message: FakeMessage  # type: ignore[assignment]


@dataclass
class FakeClientResponse(ModelClient.ModelClientResponseProtocol):
    choices: list[FakeChoice]
    model: str = "fake-model"

    def message_retrieval_function(self) -> list[str] | list[FakeMessage]:
        return [c.message for c in self.choices]
