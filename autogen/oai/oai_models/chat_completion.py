# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

# Taken over from https://github.com/openai/openai-python/blob/3e69750d47df4f0759d4a28ddc68e4b38756d9ca/src/openai/types/chat/chat_completion.py

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Any, Callable, List, Optional

from typing_extensions import Literal

from ._models import BaseModel
from .chat_completion_message import ChatCompletionMessage
from .chat_completion_token_logprob import ChatCompletionTokenLogprob
from .completion_usage import CompletionUsage

__all__ = ["ChatCompletion", "Choice", "ChoiceLogprobs"]


class ChoiceLogprobs(BaseModel):
    content: Optional[List[ChatCompletionTokenLogprob]] = None
    """A list of message content tokens with log probability information."""

    refusal: Optional[List[ChatCompletionTokenLogprob]] = None
    """A list of message refusal tokens with log probability information."""


class Choice(BaseModel):
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    """The reason the model stopped generating tokens.

    This will be `stop` if the model hit a natural stop point or a provided stop
    sequence, `length` if the maximum number of tokens specified in the request was
    reached, `content_filter` if content was omitted due to a flag from our content
    filters, `tool_calls` if the model called a tool, or `function_call`
    (deprecated) if the model called a function.
    """

    index: int
    """The index of the choice in the list of choices."""

    logprobs: Optional[ChoiceLogprobs] = None
    """Log probability information for the choice."""

    message: ChatCompletionMessage
    """A chat completion message generated by the model."""


class ChatCompletion(BaseModel):
    id: str
    """A unique identifier for the chat completion."""

    choices: List[Choice]
    """A list of chat completion choices.

    Can be more than one if `n` is greater than 1.
    """

    created: int
    """The Unix timestamp (in seconds) of when the chat completion was created."""

    model: str
    """The model used for the chat completion."""

    object: Literal["chat.completion"]
    """The object type, which is always `chat.completion`."""

    service_tier: Optional[Literal["auto", "default", "flex"]] = None
    """The service tier used for processing the request."""

    system_fingerprint: Optional[str] = None
    """This fingerprint represents the backend configuration that the model runs with.

    Can be used in conjunction with the `seed` request parameter to understand when
    backend changes have been made that might impact determinism.
    """

    usage: Optional[CompletionUsage] = None
    """Usage statistics for the completion request."""


class ChatCompletionExtended(ChatCompletion):
    message_retrieval_function: Optional[Callable[[Any, "ChatCompletion"], list[ChatCompletionMessage]]] = None
    config_id: Optional[str] = None
    pass_filter: Optional[Callable[..., bool]] = None
    cost: Optional[float] = None
