# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from typing import List

from pydantic import BaseModel


# Perplexity response model
class SearchResponse(BaseModel):
    content: str
    citations: List[str]


class Message(BaseModel):
    role: str
    content: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    search_context_size: str


class Choice(BaseModel):
    index: int
    finish_reason: str
    message: Message


class PerplexityChatCompletionResponse(BaseModel):
    id: str
    model: str
    created: int
    usage: Usage
    citations: List[str]
    object: str
    choices: List[Choice]
