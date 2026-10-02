# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Anthropic splits one answer across several ``TextBlock``s whenever citations
are on, which web search always is. ``ModelResponse.message`` holds a single
``ModelMessage``, so only the last block survives and ``reply.body`` returns a
trailing fragment instead of the answer.

Measured against the live API: asked for the capital of Iceland with
``WebSearchTool``, Anthropic returned 5 text blocks totalling 350 characters
starting "The capital of Iceland is Reykjavík"; ``reply.body`` was the 84-character
block "The city and its suburbs account for about two-thirds of Iceland's total
population."

The assertion below states the requirement, not a fix: no text the provider sent
may be missing from the durable record. How to satisfy it is open — it needs a
representation for a multi-block answer, which no model event currently has
(``ModelMessage.content`` is a single ``str``). The same defect is in
``openai_responses_client.py`` (last ``part`` wins) while ``gemini_client.py``
and ``mistral_client.py`` concatenate in the client instead.
"""

from collections.abc import Iterable
from typing import Any

import pytest
from anthropic.types import Message, TextBlock, Usage

from ag2 import Context, MemoryStream
from ag2.config.anthropic import AnthropicClient
from ag2.events import ModelResponse


async def _process(content: Iterable[Any]) -> ModelResponse:
    client = AnthropicClient(api_key="test", prompt_caching=False)
    message = Message.model_construct(
        id="m1",
        type="message",
        role="assistant",
        model="claude-sonnet-5",
        content=list(content),
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )
    return await client._process_response(message, Context(stream=MemoryStream()))


@pytest.mark.asyncio
async def test_single_text_block() -> None:
    response = await _process([TextBlock(text="Reykjavík.", type="text")])

    assert response.content == "Reykjavík."


@pytest.mark.xfail(reason="no representation for a multi-block answer; see module docstring", strict=True)
@pytest.mark.asyncio
async def test_no_text_the_provider_sent_is_lost() -> None:
    blocks = [
        TextBlock(text="The capital of Iceland is Reykjavík.", type="text"),
        TextBlock(text=" ", type="text"),
        TextBlock(text="It sits on Faxa Bay.", type="text"),
    ]

    response = await _process(blocks)

    assert response.content is not None
    for block in blocks:
        assert block.text.strip() in response.content
