# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Gemini's token counts as an AG-UI client receives them.

Driven through the public ``config=`` seam with ``GeminiConfig``'s ``http_client``,
for the reason given in ``test/config/anthropic/test_ag_ui_usage.py``: ``TestConfig``
starts downstream of the provider mapper, and what is under test is the whole path
from Gemini's payload to the wire.
"""

from typing import Any

import httpx
import pytest

pytest.importorskip("ag_ui")

from ag_ui.core import RunFinishedEvent, TokenUsage, UserMessage  # noqa: E402

from ag2 import Agent  # noqa: E402
from ag2.ag_ui import AGUIStream  # noqa: E402
from ag2.config.gemini import GeminiConfig  # noqa: E402
from test.ag_ui.harness import dispatch_run, run_input  # noqa: E402

pytestmark = pytest.mark.asyncio


def _agent(usage: dict[str, Any]) -> Agent:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"role": "model", "parts": [{"text": "done"}]}, "finishReason": "STOP"}],
                "usageMetadata": usage,
                "modelVersion": "gemini-3-flash",
            },
        )

    return Agent(
        "test_agent",
        config=GeminiConfig(
            model="gemini-3-flash",
            api_key="test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ),
    )


async def _finished(agent: Agent) -> RunFinishedEvent:
    frames = await dispatch_run(AGUIStream(agent), run_input(UserMessage(id="msg_1", content="go")))
    return RunFinishedEvent.model_validate(frames[-1])


async def test_the_output_a_client_sees_counts_what_gemini_spent_thinking() -> None:
    """Gemini reports its thoughts beside the candidates, so they are added in on the way out."""
    agent = _agent({
        "promptTokenCount": 12,
        "candidatesTokenCount": 10,
        "thoughtsTokenCount": 30,
        "totalTokenCount": 52,
    })

    assert (await _finished(agent)).usage == [
        TokenUsage(
            provider="google",
            model="gemini-3-flash",
            input_tokens=12,
            output_tokens=40,
            total_tokens=52,
            reasoning_tokens=30,
        )
    ]


async def test_a_cached_prompt_is_already_inside_gemini_s_input() -> None:
    agent = _agent({
        "promptTokenCount": 1200,
        "cachedContentTokenCount": 1000,
        "candidatesTokenCount": 10,
        "totalTokenCount": 1210,
    })

    assert (await _finished(agent)).usage == [
        TokenUsage(
            provider="google",
            model="gemini-3-flash",
            input_tokens=1200,
            output_tokens=10,
            total_tokens=1210,
            cached_input_tokens=1000,
        )
    ]
