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
from dirty_equals import IsPartialDict

from ag2 import Agent, AgentReply
from ag2.config import AnthropicConfig
from ag2.events import BuiltinToolResultEvent
from ag2.tools import tool
from ag2.tools.builtin.web_fetch import OnlyTools, UrlSources, WebFetchTool

MODEL = "claude-sonnet-5"
URL = "https://example.com"
PROMPT = f"Fetch {URL} and quote its first sentence verbatim. Use the web_fetch tool."
TOOL_PROMPT = (
    "Call `find_source` to get a URL, then fetch that URL and quote its first sentence verbatim. "
    "Use the web_fetch tool."
)


@tool
async def find_source() -> str:
    """Return the URL of the document to read."""
    return URL


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


async def _fetch_outcomes(reply: AgentReply) -> list[dict[str, object]]:
    """What the provider did with each web fetch, read off the result events rather than the prose."""
    return [
        e.result.metadata
        for e in await reply.history.get_events()
        if isinstance(e, BuiltinToolResultEvent) and e.name == "web_fetch"
    ]


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_a_url_sources_policy_is_honoured(config: AnthropicConfig) -> None:
    """Shutting user input out leaves the model no URL it is allowed to fetch.

    The URL is only ever in the prompt, so `user_input="none"` removes the one source that could
    supply it and the provider refuses the fetch with `url_not_in_prior_context`.
    """
    agent = Agent(
        "fetcher",
        config=config,
        tools=[WebFetchTool(max_uses=1, url_sources=UrlSources(user_input="none"))],
    )

    reply = await agent.ask(PROMPT)

    outcomes = await _fetch_outcomes(reply)
    assert outcomes[0] == IsPartialDict({"error": True, "error_code": "url_not_in_prior_context"})
    assert all(o.get("error") for o in outcomes)


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_url_sources_admitting_user_input_still_fetches(config: AnthropicConfig) -> None:
    """The other half of the pair: with the source admitted, the same request fetches."""
    agent = Agent(
        "fetcher",
        config=config,
        tools=[WebFetchTool(max_uses=1, url_sources=UrlSources(user_input="all"))],
    )

    reply = await agent.ask(PROMPT)

    assert any("retrieved_at" in o for o in await _fetch_outcomes(reply))


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_an_only_filter_admits_the_tool_it_names(config: AnthropicConfig) -> None:
    """The `only` form is what the name validation exists for, so the API resolves it live.

    `find_source` is the sole route to the URL — the prompt never carries it — and naming that
    tool under `client_tool_results` is what makes the fetch legal.
    """
    agent = Agent(
        "fetcher",
        config=config,
        tools=[
            find_source,
            WebFetchTool(
                max_uses=1,
                url_sources=UrlSources(
                    user_input="none",
                    client_tool_results=OnlyTools(["find_source"]),
                ),
            ),
        ],
    )

    reply = await agent.ask(TOOL_PROMPT)

    assert any("retrieved_at" in o for o in await _fetch_outcomes(reply))


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_shutting_the_client_tool_out_refuses_the_same_fetch(config: AnthropicConfig) -> None:
    """The other half of the pair: the same run with that source closed is refused."""
    agent = Agent(
        "fetcher",
        config=config,
        tools=[
            find_source,
            WebFetchTool(max_uses=1, url_sources=UrlSources(client_tool_results="none")),
        ],
    )

    reply = await agent.ask(TOOL_PROMPT)

    outcomes = await _fetch_outcomes(reply)
    assert outcomes[0] == IsPartialDict({"error": True, "error_code": "url_not_in_prior_context"})
