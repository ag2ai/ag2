# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest
import respx
from dirty_equals import IsPartialDict

pytest.importorskip("firecrawl")

from firecrawl.v2.utils.error_handler import FirecrawlError, PaymentRequiredError
from firecrawl.v2.utils.http_client_async import AsyncHttpClient

from ag2 import Agent, Context, DataInput, Variable
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from ag2.extensions.tools.search.firecrawl import (
    FirecrawlNewsResult,
    FirecrawlScrapeResult,
    FirecrawlSearchResponse,
    FirecrawlSearchResult,
    FirecrawlToolkit,
)
from ag2.testing import TestConfig, TrackingConfig

FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v2/search"
FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"

SAMPLE_SEARCH_RAW: dict[str, Any] = {
    "success": True,
    "data": {
        "web": [
            {
                "url": "https://ag2.ai",
                "title": "AG2 Framework",
                "description": "AG2 is an agent framework.",
                "position": 1,
            },
            {
                "url": "https://github.com/ag2ai/ag2",
                "title": "GitHub - AG2",
                "description": "Open source repo.",
                "position": 2,
            },
        ]
    },
}

SAMPLE_SEARCH_SCRAPED_RAW: dict[str, Any] = {
    "success": True,
    "data": {
        "web": [
            {
                "url": "https://ag2.ai",
                "title": "AG2 Framework",
                "description": "AG2 is an agent framework.",
                "position": 1,
                "markdown": "# AG2\nFull page",
                "metadata": {
                    "title": "AG2 Framework",
                    "description": "AG2 is an agent framework.",
                    "sourceURL": "https://ag2.ai",
                    "url": "https://ag2.ai/",
                    "statusCode": 200,
                },
            }
        ]
    },
}

SAMPLE_SEARCH_NEWS_RAW: dict[str, Any] = {
    "success": True,
    "data": {
        "news": [
            {
                "title": "AG2 1.1 released",
                "url": "https://news.example/ag2",
                "snippet": "A new AG2 release.",
                "date": "2 days ago",
                "position": 1,
            }
        ]
    },
}

SAMPLE_SEARCH_NEWS_SCRAPED_RAW: dict[str, Any] = {
    "success": True,
    "data": {
        "news": [
            {
                "title": "AG2 1.1 released",
                "url": "https://news.example/ag2",
                "snippet": "A new AG2 release.",
                "date": "2 days ago",
                "position": 1,
                "markdown": "# AG2 1.1",
                "metadata": {
                    "title": "AG2 1.1 released",
                    "description": "Release notes.",
                    "sourceURL": "https://news.example/ag2",
                },
            }
        ]
    },
}

SAMPLE_SCRAPE_RAW: dict[str, Any] = {
    "success": True,
    "data": {
        "markdown": "# AG2\nFull text",
        "html": "<h1>AG2</h1>",
        "links": ["https://github.com/ag2ai/ag2"],
        "metadata": {
            "title": "AG2",
            "description": "Agent framework",
            "language": "en",
            "sourceURL": "https://ag2.ai",
            "url": "https://ag2.ai/",
            "statusCode": 200,
        },
    },
}


def _tool_call_config(
    arguments: dict[str, Any],
    *,
    tool_name: str,
    final_reply: str = "done",
) -> TestConfig:
    return TestConfig(
        ModelResponse(tool_calls=ToolCallsEvent([ToolCallEvent(arguments=json.dumps(arguments), name=tool_name)])),
        final_reply,
    )


def _request_body(route: respx.Route) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(route.calls.last.request.content)
    return body


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schemas(self, context: Context) -> None:
        toolkit = FirecrawlToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))

        names = [s.function.name for s in schemas]
        assert names == ["firecrawl_search", "firecrawl_scrape"]

    async def test_search_schema_has_query_param(self, context: Context) -> None:
        toolkit = FirecrawlToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))
        search_schema = next(s for s in schemas if s.function.name == "firecrawl_search")

        assert search_schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })

    async def test_scrape_schema_has_url_param(self, context: Context) -> None:
        toolkit = FirecrawlToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))
        scrape_schema = next(s for s in schemas if s.function.name == "firecrawl_scrape")

        assert scrape_schema.function.parameters == IsPartialDict({
            "required": ["url"],
            "properties": IsPartialDict({"url": IsPartialDict({"type": "string"})}),
        })

    async def test_custom_tool_name_and_description(self, context: Context) -> None:
        toolkit = FirecrawlToolkit(api_key="test")
        custom = toolkit.search(name="web_search", description="Custom Firecrawl search.")

        [schema] = list(await custom.schemas(context))

        assert schema.function.name == "web_search"
        assert schema.function.description == "Custom Firecrawl search."

    async def test_toolkit_builds_without_api_key(self, monkeypatch: pytest.MonkeyPatch, context: Context) -> None:
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)

        toolkit = FirecrawlToolkit()

        assert len(list(await toolkit.schemas(context))) == 2


@pytest.mark.asyncio
class TestSearchExecution:
    @respx.mock
    async def test_returns_structured_results(self) -> None:
        respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = FirecrawlToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "AG2 framework"}, tool_name="firecrawl_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            FirecrawlSearchResponse(
                query="AG2 framework",
                web=[
                    FirecrawlSearchResult(
                        url="https://ag2.ai",
                        title="AG2 Framework",
                        description="AG2 is an agent framework.",
                        position=1,
                    ),
                    FirecrawlSearchResult(
                        url="https://github.com/ag2ai/ag2",
                        title="GitHub - AG2",
                        description="Open source repo.",
                        position=2,
                    ),
                ],
            )
        )

    @respx.mock
    async def test_sends_ag2_origin(self) -> None:
        route = respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = FirecrawlToolkit(api_key="test")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="firecrawl_search"), tools=[toolkit])
        await agent.ask("search")

        assert _request_body(route)["origin"] == "ag2"
        assert route.calls.last.request.headers["authorization"] == "Bearer test"

    @respx.mock
    async def test_default_request_has_no_scrape_options(self) -> None:
        route = respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = FirecrawlToolkit(api_key="test")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="firecrawl_search"), tools=[toolkit])
        await agent.ask("search")

        body = _request_body(route)
        assert body["query"] == "q"
        assert "scrapeOptions" not in body
        assert (
            not {
                "sources",
                "categories",
                "includeDomains",
                "excludeDomains",
                "tbs",
                "location",
                "country",
                "ignoreInvalidURLs",
            }
            & body.keys()
        )

    @respx.mock
    async def test_search_params_forwarded(self) -> None:
        route = respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = FirecrawlToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="firecrawl_search"),
            tools=[
                toolkit.search(
                    limit=3,
                    sources=["web", "news"],
                    categories=["pdf"],
                    include_domains=["ag2.ai"],
                    tbs="qdr:w",
                    location="Germany",
                    country="DE",
                    ignore_invalid_urls=True,
                    timeout=30000,
                )
            ],
        )
        await agent.ask("search")

        assert _request_body(route) == IsPartialDict({
            "query": "q",
            "limit": 3,
            "sources": [{"type": "web"}, {"type": "news"}],
            "categories": [{"type": "pdf"}],
            "includeDomains": ["ag2.ai"],
            "tbs": "qdr:w",
            "location": "Germany",
            "country": "DE",
            "ignoreInvalidURLs": True,
            "timeout": 30000,
        })

    @respx.mock
    async def test_exclude_domains_forwarded(self) -> None:
        route = respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = FirecrawlToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="firecrawl_search"),
            tools=[toolkit.search(exclude_domains=["example.com"])],
        )
        await agent.ask("search")

        assert _request_body(route)["excludeDomains"] == ["example.com"]

    @respx.mock
    async def test_scrape_results_returns_page_markdown(self) -> None:
        route = respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_SCRAPED_RAW))
        toolkit = FirecrawlToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "AG2"}, tool_name="firecrawl_search"))
        agent = Agent("a", config=config, tools=[toolkit.search(scrape_results=True)])
        await agent.ask("search")

        assert _request_body(route)["scrapeOptions"] == IsPartialDict({
            "formats": ["markdown"],
            "onlyMainContent": True,
        })

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            FirecrawlSearchResponse(
                query="AG2",
                web=[
                    FirecrawlSearchResult(
                        url="https://ag2.ai/",
                        title="AG2 Framework",
                        description="AG2 is an agent framework.",
                        position=1,
                        markdown="# AG2\nFull page",
                    )
                ],
            )
        )

    @respx.mock
    async def test_scrape_results_options_forwarded(self) -> None:
        route = respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_SCRAPED_RAW))
        toolkit = FirecrawlToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="firecrawl_search"),
            tools=[toolkit.search(scrape_results=True, scrape_only_main_content=False, scrape_max_age=0)],
        )
        await agent.ask("search")

        assert _request_body(route)["scrapeOptions"] == IsPartialDict({
            "formats": ["markdown"],
            "onlyMainContent": False,
            "maxAge": 0,
        })

    @respx.mock
    async def test_scrape_results_false_sends_no_scrape_options(self) -> None:
        route = respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = FirecrawlToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="firecrawl_search"),
            tools=[toolkit.search(scrape_results=False, scrape_max_age=0)],
        )
        await agent.ask("search")

        assert "scrapeOptions" not in _request_body(route)

    @respx.mock
    async def test_toolkit_level_scrape_results_applied_to_default_search(self) -> None:
        route = respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_SCRAPED_RAW))
        toolkit = FirecrawlToolkit(api_key="test", limit=2, scrape_results=True)

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="firecrawl_search"), tools=[toolkit])
        await agent.ask("search")

        body = _request_body(route)
        assert body["limit"] == 2
        assert "scrapeOptions" in body

    @respx.mock
    async def test_news_results(self) -> None:
        respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_NEWS_RAW))
        toolkit = FirecrawlToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "AG2"}, tool_name="firecrawl_search"))
        agent = Agent("a", config=config, tools=[toolkit.search(sources=["news"])])
        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            FirecrawlSearchResponse(
                query="AG2",
                news=[
                    FirecrawlNewsResult(
                        url="https://news.example/ag2",
                        title="AG2 1.1 released",
                        snippet="A new AG2 release.",
                        date="2 days ago",
                        position=1,
                    )
                ],
            )
        )

    @respx.mock
    async def test_scraped_news_results(self) -> None:
        respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_NEWS_SCRAPED_RAW))
        toolkit = FirecrawlToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "AG2"}, tool_name="firecrawl_search"))
        agent = Agent("a", config=config, tools=[toolkit.search(sources=["news"], scrape_results=True)])
        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            FirecrawlSearchResponse(
                query="AG2",
                news=[
                    FirecrawlNewsResult(
                        url="https://news.example/ag2",
                        title="AG2 1.1 released",
                        snippet="Release notes.",
                        position=1,
                        markdown="# AG2 1.1",
                    )
                ],
            )
        )

    @respx.mock
    async def test_explicit_api_url(self) -> None:
        route = respx.post("https://example.test/v2/search").mock(
            return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW)
        )
        toolkit = FirecrawlToolkit(api_key="test", api_url="https://example.test")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="firecrawl_search"), tools=[toolkit])
        await agent.ask("search")

        assert route.called

    @respx.mock
    async def test_http_timeout_and_max_retries_forwarded(self) -> None:
        route = respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(502))
        toolkit = FirecrawlToolkit(api_key="test", http_timeout=7.5, max_retries=1)

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="firecrawl_search"), tools=[toolkit])
        with pytest.raises(FirecrawlError):
            await agent.ask("search")

        assert route.call_count == 1
        assert route.calls.last.request.extensions["timeout"]["read"] == 7.5

    @respx.mock
    async def test_api_error_surfaces(self) -> None:
        respx.post(FIRECRAWL_SEARCH_URL).mock(
            return_value=httpx.Response(402, json={"success": False, "error": "Insufficient credits"})
        )
        toolkit = FirecrawlToolkit(api_key="test")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="firecrawl_search"), tools=[toolkit])

        with pytest.raises(PaymentRequiredError):
            await agent.ask("search")

    @respx.mock
    async def test_client_closed_after_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        respx.post(FIRECRAWL_SEARCH_URL).mock(
            return_value=httpx.Response(402, json={"success": False, "error": "Insufficient credits"})
        )
        closed: list[AsyncHttpClient] = []
        original_close = AsyncHttpClient.close

        async def tracking_close(self: AsyncHttpClient) -> None:
            closed.append(self)
            await original_close(self)

        monkeypatch.setattr(AsyncHttpClient, "close", tracking_close)
        toolkit = FirecrawlToolkit(api_key="test")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="firecrawl_search"), tools=[toolkit])
        with pytest.raises(PaymentRequiredError):
            await agent.ask("search")

        assert len(closed) == 1

    @respx.mock
    async def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        route = respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = FirecrawlToolkit()

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="firecrawl_search"), tools=[toolkit])

        with pytest.raises(ValueError, match="API key"):
            await agent.ask("search")
        assert not route.called

    @respx.mock
    async def test_api_key_read_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIRECRAWL_API_KEY", "env-key")
        route = respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = FirecrawlToolkit()

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="firecrawl_search"), tools=[toolkit])
        await agent.ask("search")

        assert route.calls.last.request.headers["authorization"] == "Bearer env-key"


@pytest.mark.asyncio
class TestScrapeExecution:
    @respx.mock
    async def test_returns_markdown(self) -> None:
        route = respx.post(FIRECRAWL_SCRAPE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SCRAPE_RAW))
        toolkit = FirecrawlToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"url": "https://ag2.ai"}, tool_name="firecrawl_scrape"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("scrape")

        body = _request_body(route)
        assert body == IsPartialDict({"url": "https://ag2.ai", "formats": ["markdown"], "origin": "ag2"})

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            FirecrawlScrapeResult(
                url="https://ag2.ai/",
                title="AG2",
                description="Agent framework",
                language="en",
                status_code=200,
                markdown="# AG2\nFull text",
                html="<h1>AG2</h1>",
                links=["https://github.com/ag2ai/ag2"],
            )
        )

    @respx.mock
    @pytest.mark.parametrize("url", ["file:///etc/passwd", "javascript:alert(1)", "data:text/plain,hello"])
    async def test_scrape_rejects_unsafe_url_scheme_before_client_call(self, url: str) -> None:
        route = respx.post(FIRECRAWL_SCRAPE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SCRAPE_RAW))
        toolkit = FirecrawlToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"url": url}, tool_name="firecrawl_scrape"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("scrape")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        result = tool_results_event.results[0].result.parts[0].data

        assert result == {"error": f"Only http/https URLs are supported; rejected: {url!r}"}
        assert not route.called

    @respx.mock
    async def test_scrape_params_forwarded(self) -> None:
        route = respx.post(FIRECRAWL_SCRAPE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SCRAPE_RAW))
        toolkit = FirecrawlToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="firecrawl_scrape"),
            tools=[
                toolkit.scrape(
                    formats=["markdown", "links"],
                    only_main_content=False,
                    include_tags=["article"],
                    exclude_tags=["nav"],
                    headers={"Accept-Language": "en"},
                    wait_for=1000,
                    mobile=True,
                    proxy="auto",
                    block_ads=False,
                    max_age=0,
                    store_in_cache=False,
                    timeout=20000,
                )
            ],
        )
        await agent.ask("scrape")

        assert _request_body(route) == IsPartialDict({
            "url": "https://ag2.ai",
            "formats": ["markdown", "links"],
            "onlyMainContent": False,
            "includeTags": ["article"],
            "excludeTags": ["nav"],
            "headers": {"Accept-Language": "en"},
            "waitFor": 1000,
            "mobile": True,
            "proxy": "auto",
            "blockAds": False,
            "maxAge": 0,
            "storeInCache": False,
            "timeout": 20000,
        })

    @respx.mock
    async def test_toolkit_level_scrape_params_applied_to_default_scrape(self) -> None:
        route = respx.post(FIRECRAWL_SCRAPE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SCRAPE_RAW))
        toolkit = FirecrawlToolkit(api_key="test", formats=["html"], only_main_content=False)

        agent = Agent(
            "a", config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="firecrawl_scrape"), tools=[toolkit]
        )
        await agent.ask("scrape")

        assert _request_body(route) == IsPartialDict({"formats": ["html"], "onlyMainContent": False})


@pytest.mark.asyncio
class TestFirecrawlToolkitVariable:
    @respx.mock
    async def test_search_resolved(self) -> None:
        route = respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_SCRAPED_RAW))
        toolkit = FirecrawlToolkit(api_key="test")
        search_tool = toolkit.search(
            limit=Variable("max_results"),
            country=Variable(),
            scrape_results=Variable("read_pages"),
        )

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="firecrawl_search"),
            tools=[search_tool],
            variables={"max_results": 4, "country": "US", "read_pages": True},
        )
        await agent.ask("search")

        body = _request_body(route)
        assert body["limit"] == 4
        assert body["country"] == "US"
        assert "scrapeOptions" in body

    @respx.mock
    async def test_scrape_resolved(self) -> None:
        route = respx.post(FIRECRAWL_SCRAPE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SCRAPE_RAW))
        toolkit = FirecrawlToolkit(api_key="test")
        scrape_tool = toolkit.scrape(formats=Variable("scrape_formats"), max_age=Variable())

        agent = Agent(
            "a",
            config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="firecrawl_scrape"),
            tools=[scrape_tool],
            variables={"scrape_formats": ["markdown", "html"], "max_age": 0},
        )
        await agent.ask("scrape")

        assert _request_body(route) == IsPartialDict({"formats": ["markdown", "html"], "maxAge": 0})

    @respx.mock
    async def test_single_string_variable_wrapped_in_list(self) -> None:
        route = respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_NEWS_RAW))
        toolkit = FirecrawlToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="firecrawl_search"),
            tools=[toolkit.search(sources=Variable())],
            variables={"sources": "news"},
        )
        await agent.ask("search")

        assert _request_body(route)["sources"] == [{"type": "news"}]

    @respx.mock
    async def test_missing_raises(self) -> None:
        respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = FirecrawlToolkit(api_key="test")
        search_tool = toolkit.search(limit=Variable())

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="firecrawl_search"),
            tools=[search_tool],
        )

        with pytest.raises(KeyError, match="limit"):
            await agent.ask("search")


@pytest.mark.asyncio
class TestIndividualTools:
    @respx.mock
    async def test_search_tool_passed_alone(self) -> None:
        route = respx.post(FIRECRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = FirecrawlToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="firecrawl_search"),
            tools=[toolkit.search()],
        )
        await agent.ask("search")

        assert route.called

    @respx.mock
    async def test_scrape_tool_passed_alone(self) -> None:
        route = respx.post(FIRECRAWL_SCRAPE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SCRAPE_RAW))
        toolkit = FirecrawlToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="firecrawl_scrape"),
            tools=[toolkit.scrape()],
        )
        await agent.ask("scrape")

        assert route.called
