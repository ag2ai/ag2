# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx
import pytest
from dirty_equals import IsPartialDict

from ag2 import Agent
from ag2.config import GeminiConfig

MODEL = "gemini-3.1-flash-lite"


async def _ask(http_client: httpx.AsyncClient, **overrides: Any) -> None:
    config = GeminiConfig(model=MODEL, api_key="test", http_client=http_client, **overrides)
    agent = Agent(name="sampler", prompt="Answer concisely.", config=config)

    await agent.ask("capital of France?")


@pytest.mark.asyncio
class TestSamplingOnTheWire:
    """``temperature``/``top_p``/``top_k`` are ignored on the Gemini 3.x line but
    honoured on 2.x and on Vertex, and the config takes an arbitrary model
    string — so they stay, and stay opt-in."""

    async def test_set_sampling_parameters_reach_the_request(
        self,
        capturing_http_client: httpx.AsyncClient,
        captured_request: dict[str, Any],
    ) -> None:
        await _ask(capturing_http_client, temperature=0.25, top_p=0.5, top_k=1)

        assert captured_request["body"]["generationConfig"] == IsPartialDict({
            "temperature": 0.25,
            "topP": 0.5,
            "topK": 1,
        })

    async def test_unset_sampling_parameters_are_absent_from_the_request(
        self,
        capturing_http_client: httpx.AsyncClient,
        captured_request: dict[str, Any],
    ) -> None:
        await _ask(capturing_http_client)

        assert captured_request["body"]["generationConfig"] == {}
