# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Live coverage for the web fetch options ag2 sends: the API accepts what it is given.

`claude-haiku-4-5` cannot carry `web_fetch_20260309` or later — it has no programmatic tool
calling, and those versions ask for it through `allowed_callers`, which ag2 does not expose. The
model here is one that does.
"""

import os

import pytest

from ag2 import Agent
from ag2.config import AnthropicConfig
from ag2.tools import WebFetchTool

MODEL = "claude-sonnet-5"
PROMPT = "Fetch https://example.com and quote its first sentence verbatim. Use the web_fetch tool."


@pytest.fixture()
def config() -> AnthropicConfig:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return AnthropicConfig(model=MODEL, api_key=api_key)


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_strict_is_accepted_on_the_oldest_version(config: AnthropicConfig) -> None:
    agent = Agent("fetcher", config=config, tools=[WebFetchTool(max_uses=1, strict=True)])

    reply = await agent.ask(PROMPT)

    assert reply.body is not None
    assert "domain" in reply.body.lower()


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_every_option_is_accepted_on_the_newest_version(config: AnthropicConfig) -> None:
    agent = Agent(
        "fetcher",
        config=config,
        tools=[
            WebFetchTool(
                max_uses=1,
                strict=True,
                use_cache=False,
                response_inclusion="full",
                version="web_fetch_20260318",
            )
        ],
    )

    reply = await agent.ask(PROMPT)

    assert reply.body is not None
    assert "domain" in reply.body.lower()
