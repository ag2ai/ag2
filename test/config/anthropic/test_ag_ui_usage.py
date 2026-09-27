# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Anthropic's reasoning count as an AG-UI client receives it.

**A deliberate exception to `test/CLAUDE.md`'s "always use `TestConfig`" rule.**
``TestConfig`` returns a ready-made ``Usage`` verbatim, so it starts *downstream* of the
mapper this ticket changes and cannot observe whether the provider layer reads the count
at all — the rule's own goal, a public-API test, is met here instead by driving the
public ``config=`` seam with ``AnthropicConfig``'s public ``http_client`` field. Nothing
private is patched, which is the behaviour the rule exists to prevent. What that buys is
the whole path under test: the SDK parsing the payload, the mapper, ``UsageEvent``,
``UsageReport``, the AG-UI grouping.

It lives here rather than beside the other AG-UI tests because a top-level
``ag2.config.anthropic`` import outside this package breaks *collection* on the LLM
matrix runs that install one provider at a time.
"""

from typing import Any

import httpx2
import pytest

pytest.importorskip("ag_ui")

from ag_ui.core import RunFinishedEvent, TokenUsage, UserMessage  # noqa: E402

from ag2 import Agent  # noqa: E402
from ag2.ag_ui import AGUIStream  # noqa: E402
from ag2.config.anthropic import AnthropicConfig  # noqa: E402
from test.ag_ui.utils import collect_events, create_run_input  # noqa: E402

pytestmark = pytest.mark.asyncio


def _agent(usage: dict[str, Any]) -> Agent:
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=message)

    return Agent(
        "test_agent",
        config=AnthropicConfig(
            model="claude-haiku-4-5",
            api_key="test",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        ),
    )


async def _finished(agent: Agent) -> RunFinishedEvent:
    """The terminating event of a completed run, parsed by the class that sent it."""
    frames = await collect_events(AGUIStream(agent), create_run_input(UserMessage(id="msg_1", content="go")))
    return RunFinishedEvent.model_validate(frames[-1])


async def test_a_client_sees_the_reasoning_tokens_anthropic_measured() -> None:
    agent = _agent({
        "input_tokens": 100,
        "output_tokens": 300,
        "output_tokens_details": {"thinking_tokens": 214},
    })

    assert (await _finished(agent)).usage == [
        TokenUsage(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=100,
            output_tokens=300,
            total_tokens=400,
            reasoning_tokens=214,
        )
    ]


async def test_a_run_that_did_not_reason_omits_them() -> None:
    agent = _agent({"input_tokens": 100, "output_tokens": 8})

    assert (await _finished(agent)).usage == [
        TokenUsage(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=100,
            output_tokens=8,
            total_tokens=108,
        )
    ]


async def test_the_zeros_anthropic_always_sends_reach_the_client_as_zeros() -> None:
    """The body a live ``claude-opus-5`` call returned, probed 2026-09-19.

    Anthropic reports both cache counts as a measured ``0`` on every call that used no
    caching, so this is what the presence-versus-truthiness fix changes on ordinary
    traffic: the client now sees a zero it can trust instead of an absence it cannot
    distinguish from a cache that was never measured.
    """
    agent = _agent({
        "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "inference_geo": "global",
        "input_tokens": 36,
        "output_tokens": 269,
        "output_tokens_details": {"thinking_tokens": 45},
        "server_tool_use": None,
        "service_tier": "standard",
    })

    assert (await _finished(agent)).usage == [
        TokenUsage(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=36,
            output_tokens=269,
            total_tokens=305,
            reasoning_tokens=45,
            cached_input_tokens=0,
        )
    ]
