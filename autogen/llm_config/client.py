# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from collections.abc import Sequence
from typing import Any, Protocol

from ..doc_utils import export_module


@export_module("autogen")
class ModelClient(Protocol):
    """A client class must implement the following methods:
    - create must return a response object that implements the ModelClientResponseProtocol
    - cost must return the cost of the response
    - get_usage must return a dict with the following keys:
        - prompt_tokens
        - completion_tokens
        - total_tokens
        - cost
        - model

    This class is used to create a client that can be used by OpenAIWrapper.
    The response returned from create must adhere to the ModelClientResponseProtocol but can be extended however needed.
    The message_retrieval method must be implemented to return a list of str or a list of messages from the response.
    """

    RESPONSE_USAGE_KEYS: list[str] = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    class ModelClientResponseProtocol(Protocol):
        class Choice(Protocol):
            class Message(Protocol):
                content: str | dict[str, Any] | list[dict[str, Any]] | None

            message: Message

        choices: Sequence[Choice]
        model: str

    def create(self, params: dict[str, Any]) -> ModelClientResponseProtocol: ...  # pragma: no cover

    def message_retrieval(
        self, response: ModelClientResponseProtocol
    ) -> list[str] | list["ModelClient.ModelClientResponseProtocol.Choice.Message"]:
        """Retrieve and return a list of strings or a list of Choice.Message from the response.

        NOTE: if a list of Choice.Message is returned, it currently needs to contain the fields of OpenAI's ChatCompletion Message object,
        since that is expected for function or tool calling in the rest of the codebase at the moment, unless a custom agent is being used.
        """
        ...  # pragma: no cover

    def cost(self, response: ModelClientResponseProtocol) -> float: ...  # pragma: no cover

    @staticmethod
    def get_usage(response: ModelClientResponseProtocol) -> dict[str, Any]:
        """Return usage summary of the response using RESPONSE_USAGE_KEYS."""
        ...  # pragma: no cover
