# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

# Taken over from https://github.com/openai/openai-python/blob/3e69750d47df4f0759d4a28ddc68e4b38756d9ca/src/openai/types/chat/chat_completion_message_tool_call.py

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from pydantic import BaseModel
from typing_extensions import Literal

__all__ = ["ChatCompletionMessageToolCall", "Function"]


class Function(BaseModel):
    arguments: str
    """
    The arguments to call the function with, as generated by the model in JSON
    format. Note that the model does not always generate valid JSON, and may
    hallucinate parameters not defined by your function schema. Validate the
    arguments in your code before calling your function.
    """

    name: str
    """The name of the function to call."""


class ChatCompletionMessageToolCall(BaseModel):
    id: str
    """The ID of the tool call."""

    function: Function
    """The function that the model called."""

    type: Literal["function"]
    """The type of the tool. Currently, only `function` is supported."""
