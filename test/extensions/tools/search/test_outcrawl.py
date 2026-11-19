# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest
import respx
from dirty_equals import IsPartialDict

from ag2 import Agent, Context, DataInput, Variable
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from ag2.extensions.tools.search import OutcrawlToolkit as ExportedOutcrawlToolkit
from ag2.extensions.tools.search.outcrawl import (
    OutcrawlAPIError,
    OutcrawlScrapeResult,
    OutcrawlSearchResponse,
    OutcrawlSearchResult,
    OutcrawlToolkit,
)
from ag2.testing import TestConfig, TrackingConfig

OUTCRAWL_SEARCH_URL = "https://api.outcrawl.ai/v1/search"
OUTCRAWL_SCRAPE_URL = "https://api.outcrawl.ai/v1/scrape"

SAMPLE_SEARCH_RAW: dict[str, Any] = {
    "hits": [
        {"url": "https://ag2.ai", "title": "AG2 Framework", "snippet": "AG2 is an agent framework.", "rank": 1},
        {"url": "https://github.com/ag2ai/ag2", "title": "GitHub - AG2", "snippet": "Open source repo.", "rank": 2},
    ],
    "usage": {"credits": "1"},
}

SAMPLE_SEARCH_SCRAPED_RAW: dict[str, Any] = {
    "hits": [
        {
            "url": "https://ag2.ai",
            "title": "AG2 Framework",
            "snippet": "AG2 is an agent framework.",
            "rank": 1,
            "markdown": "# AG2\nFull page",
        }
    ],
    "usage": {"credits": "2"},
}

SAMPLE_SCRAPE_RAW: dict[str, Any] = {
    "url": "https://ag2.ai/",
    "statusCode": 200,
    "markdown": "# AG2\nFull text",
    "html": "<h1>AG2</h1>",
    "links": ["https://github.com/ag2ai/ag2"],
    "metadata": {"title": "AG2", "description": "Agent framework", "language": "en"},
    "usage": {"credits": "1"},
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


def test_exported_from_search_package() -> None:
    assert ExportedOutcrawlToolkit is OutcrawlToolkit


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schemas(self, context: Context) -> None:
        toolkit = OutcrawlToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))

        names = [s.function.name for s in schemas]
        assert names == ["outcrawl_search", "outcrawl_scrape"]

    async def test_search_schema_has_query_param(self, context: Context) -> None:
        toolkit = OutcrawlToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))
        search_schema = next(s for s in schemas if s.function.name == "outcrawl_search")

        assert search_schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })

    async def test_scrape_schema_has_url_param(self, context: Context) -> None:
        toolkit = OutcrawlToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))
        scrape_schema = next(s for s in schemas if s.function.name == "outcrawl_scrape")

        assert scrape_schema.function.parameters == IsPartialDict({
            "required": ["url"],
            "properties": IsPartialDict({"url": IsPartialDict({"type": "string"})}),
        })

    async def test_custom_tool_name_and_description(self, context: Context) -> None:
        toolkit = OutcrawlToolkit(api_key="test")
        custom = toolkit.search(name="web_search", description="Custom Outcrawl search.")

        [schema] = list(await custom.schemas(context))

        assert schema.function.name == "web_search"
        assert schema.function.description == "Custom Outcrawl search."

    async def test_toolkit_builds_without_api_key(self, monkeypatch: pytest.MonkeyPatch, context: Context) -> None:
        monkeypatch.delenv("OUTCRAWL_API_KEY", raising=False)

        toolkit = OutcrawlToolkit()

        assert len(list(await toolkit.schemas(context))) == 2


@pytest.mark.asyncio
class TestSearchExecution:
    @respx.mock
    async def test_returns_structured_results(self) -> None:
        respx.post(OUTCRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = OutcrawlToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "AG2 framework"}, tool_name="outcrawl_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            OutcrawlSearchResponse(
                query="AG2 framework",
                results=[
                    OutcrawlSearchResult(
                        url="https://ag2.ai",
                        title="AG2 Framework",
                        snippet="AG2 is an agent framework.",
                        rank=1,
                    ),
                    OutcrawlSearchResult(
                        url="https://github.com/ag2ai/ag2",
                        title="GitHub - AG2",
                        snippet="Open source repo.",
                        rank=2,
                    ),
                ],
            )
        )

    @respx.mock
    async def test_sends_bearer_key_and_user_agent(self) -> None:
        route = respx.post(OUTCRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = OutcrawlToolkit(api_key="test")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="outcrawl_search"), tools=[toolkit])
        await agent.ask("search")

        headers = route.calls.last.request.headers
        assert headers["authorization"] == "Bearer test"
        assert headers["user-agent"] == "ag2-outcrawl-extension"

    @respx.mock
    async def test_default_request_sends_only_query(self) -> None:
        route = respx.post(OUTCRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = OutcrawlToolkit(api_key="test")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="outcrawl_search"), tools=[toolkit])
        await agent.ask("search")

        assert _request_body(route) == {"query": "q"}

    @respx.mock
    async def test_scrape_results_returns_page_markdown(self) -> None:
        route = respx.post(OUTCRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_SCRAPED_RAW))
        toolkit = OutcrawlToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "AG2"}, tool_name="outcrawl_search"))
        agent = Agent("a", config=config, tools=[toolkit.search(limit=3, scrape_results=True)])
        await agent.ask("search")

        assert _request_body(route) == {"query": "AG2", "limit": 3, "scrape": True, "formats": ["markdown"]}

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            OutcrawlSearchResponse(
                query="AG2",
                results=[
                    OutcrawlSearchResult(
                        url="https://ag2.ai",
                        title="AG2 Framework",
                        snippet="AG2 is an agent framework.",
                        rank=1,
                        markdown="# AG2\nFull page",
                    )
                ],
            )
        )

    @respx.mock
    async def test_scrape_results_false_sends_no_scrape(self) -> None:
        route = respx.post(OUTCRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = OutcrawlToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="outcrawl_search"),
            tools=[toolkit.search(scrape_results=False)],
        )
        await agent.ask("search")

        assert not {"scrape", "formats"} & _request_body(route).keys()

    @respx.mock
    async def test_toolkit_level_search_defaults_applied(self) -> None:
        route = respx.post(OUTCRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_SCRAPED_RAW))
        toolkit = OutcrawlToolkit(api_key="test", limit=2, scrape_results=True)

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="outcrawl_search"), tools=[toolkit])
        await agent.ask("search")

        assert _request_body(route) == IsPartialDict({"limit": 2, "scrape": True})

    @respx.mock
    async def test_missing_rank_falls_back_to_position(self) -> None:
        raw = {"hits": [{"url": "https://a.io"}, {"url": "https://b.io", "title": "B"}, "not-a-hit"]}
        respx.post(OUTCRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=raw))
        toolkit = OutcrawlToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "q"}, tool_name="outcrawl_search"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            OutcrawlSearchResponse(
                query="q",
                results=[
                    OutcrawlSearchResult(url="https://a.io", rank=1),
                    OutcrawlSearchResult(url="https://b.io", title="B", rank=2),
                ],
            )
        )

    @respx.mock
    async def test_empty_response_returns_no_results(self) -> None:
        respx.post(OUTCRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=[]))
        toolkit = OutcrawlToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "q"}, tool_name="outcrawl_search"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(OutcrawlSearchResponse(query="q"))

    @respx.mock
    async def test_explicit_base_url(self) -> None:
        route = respx.post("https://example.test/v1/search").mock(
            return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW)
        )
        toolkit = OutcrawlToolkit(api_key="test", base_url="https://example.test/")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="outcrawl_search"), tools=[toolkit])
        await agent.ask("search")

        assert route.called

    @respx.mock
    async def test_timeout_forwarded(self) -> None:
        route = respx.post(OUTCRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = OutcrawlToolkit(api_key="test", timeout=7.5)

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="outcrawl_search"), tools=[toolkit])
        await agent.ask("search")

        assert route.calls.last.request.extensions["timeout"]["read"] == 7.5

    @respx.mock
    async def test_api_error_surfaces(self) -> None:
        respx.post(OUTCRAWL_SEARCH_URL).mock(
            return_value=httpx.Response(402, json={"code": "payment_required", "message": "Out of credits"})
        )
        toolkit = OutcrawlToolkit(api_key="test")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="outcrawl_search"), tools=[toolkit])

        with pytest.raises(OutcrawlAPIError, match=r"402 \(payment_required\): Out of credits") as exc_info:
            await agent.ask("search")
        assert exc_info.value.status_code == 402
        assert exc_info.value.code == "payment_required"

    @respx.mock
    async def test_api_error_without_json_body(self) -> None:
        respx.post(OUTCRAWL_SEARCH_URL).mock(return_value=httpx.Response(502, text="Bad Gateway"))
        toolkit = OutcrawlToolkit(api_key="test")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="outcrawl_search"), tools=[toolkit])

        with pytest.raises(OutcrawlAPIError, match="Outcrawl API error 502: Bad Gateway") as exc_info:
            await agent.ask("search")
        assert exc_info.value.code is None

    @respx.mock
    async def test_api_key_read_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OUTCRAWL_API_KEY", "env-key")
        route = respx.post(OUTCRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = OutcrawlToolkit()

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="outcrawl_search"), tools=[toolkit])
        await agent.ask("search")

        assert route.calls.last.request.headers["authorization"] == "Bearer env-key"

    @respx.mock
    async def test_missing_api_key_raises_before_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OUTCRAWL_API_KEY", raising=False)
        route = respx.post(OUTCRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = OutcrawlToolkit()

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="outcrawl_search"), tools=[toolkit])

        with pytest.raises(ValueError, match="OUTCRAWL_API_KEY"):
            await agent.ask("search")
        assert not route.called


@pytest.mark.asyncio
class TestScrapeExecution:
    @respx.mock
    async def test_returns_markdown(self) -> None:
        route = respx.post(OUTCRAWL_SCRAPE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SCRAPE_RAW))
        toolkit = OutcrawlToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"url": "https://ag2.ai"}, tool_name="outcrawl_scrape"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("scrape")

        assert _request_body(route) == {"url": "https://ag2.ai", "formats": ["markdown"]}

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            OutcrawlScrapeResult(
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
    async def test_sparse_response_falls_back_to_requested_url(self) -> None:
        respx.post(OUTCRAWL_SCRAPE_URL).mock(return_value=httpx.Response(200, json={"markdown": "# AG2"}))
        toolkit = OutcrawlToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"url": "https://ag2.ai"}, tool_name="outcrawl_scrape"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("scrape")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            OutcrawlScrapeResult(url="https://ag2.ai", markdown="# AG2")
        )

    @respx.mock
    @pytest.mark.parametrize("url", ["file:///etc/passwd", "javascript:alert(1)", "data:text/plain,hello"])
    async def test_scrape_rejects_unsafe_url_scheme_before_request(self, url: str) -> None:
        route = respx.post(OUTCRAWL_SCRAPE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SCRAPE_RAW))
        toolkit = OutcrawlToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"url": url}, tool_name="outcrawl_scrape"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("scrape")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        result = tool_results_event.results[0].result.parts[0].data

        assert result == {"error": f"Only http/https URLs are supported; rejected: {url!r}"}
        assert not route.called

    @respx.mock
    async def test_scrape_params_forwarded(self) -> None:
        route = respx.post(OUTCRAWL_SCRAPE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SCRAPE_RAW))
        toolkit = OutcrawlToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="outcrawl_scrape"),
            tools=[toolkit.scrape(formats=["markdown", "links"], only_main_content=False)],
        )
        await agent.ask("scrape")

        assert _request_body(route) == {
            "url": "https://ag2.ai",
            "formats": ["markdown", "links"],
            "onlyMainContent": False,
        }

    @respx.mock
    async def test_toolkit_level_scrape_params_applied_to_default_scrape(self) -> None:
        route = respx.post(OUTCRAWL_SCRAPE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SCRAPE_RAW))
        toolkit = OutcrawlToolkit(api_key="test", formats=["html"], only_main_content=False)

        agent = Agent(
            "a", config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="outcrawl_scrape"), tools=[toolkit]
        )
        await agent.ask("scrape")

        assert _request_body(route) == IsPartialDict({"formats": ["html"], "onlyMainContent": False})

    @respx.mock
    async def test_api_error_surfaces(self) -> None:
        respx.post(OUTCRAWL_SCRAPE_URL).mock(
            return_value=httpx.Response(400, json={"code": "bad_request", "message": "scrape: url is not a url"})
        )
        toolkit = OutcrawlToolkit(api_key="test")

        agent = Agent(
            "a", config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="outcrawl_scrape"), tools=[toolkit]
        )

        with pytest.raises(OutcrawlAPIError, match="bad_request"):
            await agent.ask("scrape")


@pytest.mark.asyncio
class TestMaxCharacters:
    @respx.mock
    async def test_search_markdown_truncated(self) -> None:
        respx.post(OUTCRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_SCRAPED_RAW))
        toolkit = OutcrawlToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "AG2"}, tool_name="outcrawl_search"))
        agent = Agent("a", config=config, tools=[toolkit.search(scrape_results=True, max_characters=5)])
        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            OutcrawlSearchResponse(
                query="AG2",
                results=[
                    OutcrawlSearchResult(
                        url="https://ag2.ai",
                        title="AG2 Framework",
                        snippet="AG2 is an agent framework.",
                        rank=1,
                        markdown="# AG2",
                        truncated=True,
                    )
                ],
            )
        )

    @respx.mock
    async def test_short_content_not_marked_truncated(self) -> None:
        respx.post(OUTCRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_SCRAPED_RAW))
        toolkit = OutcrawlToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "AG2"}, tool_name="outcrawl_search"))
        agent = Agent("a", config=config, tools=[toolkit.search(scrape_results=True, max_characters=10_000)])
        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        [result] = tool_results_event.results[0].result.parts[0].data.results
        assert result.markdown == "# AG2\nFull page"
        assert result.truncated is False

    @respx.mock
    async def test_scrape_markdown_and_html_truncated(self) -> None:
        respx.post(OUTCRAWL_SCRAPE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SCRAPE_RAW))
        toolkit = OutcrawlToolkit(api_key="test", formats=["markdown", "html"], max_characters=4)

        config = TrackingConfig(_tool_call_config({"url": "https://ag2.ai"}, tool_name="outcrawl_scrape"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("scrape")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            OutcrawlScrapeResult(
                url="https://ag2.ai/",
                title="AG2",
                description="Agent framework",
                language="en",
                status_code=200,
                markdown="# AG",
                html="<h1>",
                links=[],
                truncated=True,
            )
        )

    @respx.mock
    async def test_max_characters_resolved_from_variable(self) -> None:
        respx.post(OUTCRAWL_SCRAPE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SCRAPE_RAW))
        toolkit = OutcrawlToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"url": "https://ag2.ai"}, tool_name="outcrawl_scrape"))
        agent = Agent(
            "a",
            config=config,
            tools=[toolkit.scrape(max_characters=Variable())],
            variables={"max_characters": 3},
        )
        await agent.ask("scrape")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        result = tool_results_event.results[0].result.parts[0].data
        assert (result.markdown, result.html, result.links, result.truncated) == ("# A", "<h1", [], True)

    @respx.mock
    @pytest.mark.parametrize(
        ("max_characters", "expected_links", "truncated"),
        [
            (20, ["https://a.io"], True),
            (24, ["https://a.io", "https://b.io"], False),
        ],
    )
    async def test_scrape_links_truncated(
        self, max_characters: int, expected_links: list[str], truncated: bool
    ) -> None:
        raw = {**SAMPLE_SCRAPE_RAW, "markdown": None, "html": None, "links": ["https://a.io", "https://b.io"]}
        respx.post(OUTCRAWL_SCRAPE_URL).mock(return_value=httpx.Response(200, json=raw))
        toolkit = OutcrawlToolkit(api_key="test", formats=["links"], max_characters=max_characters)

        config = TrackingConfig(_tool_call_config({"url": "https://ag2.ai"}, tool_name="outcrawl_scrape"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("scrape")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        result = tool_results_event.results[0].result.parts[0].data
        assert result.links == expected_links
        assert result.truncated is truncated


@pytest.mark.asyncio
class TestOutcrawlToolkitVariable:
    @respx.mock
    async def test_search_resolved(self) -> None:
        route = respx.post(OUTCRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_SCRAPED_RAW))
        toolkit = OutcrawlToolkit(api_key="test")
        search_tool = toolkit.search(limit=Variable("max_results"), scrape_results=Variable("read_pages"))

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="outcrawl_search"),
            tools=[search_tool],
            variables={"max_results": 4, "read_pages": True},
        )
        await agent.ask("search")

        assert _request_body(route) == {"query": "q", "limit": 4, "scrape": True, "formats": ["markdown"]}

    @respx.mock
    async def test_scrape_resolved(self) -> None:
        route = respx.post(OUTCRAWL_SCRAPE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SCRAPE_RAW))
        toolkit = OutcrawlToolkit(api_key="test")
        scrape_tool = toolkit.scrape(formats=Variable("scrape_formats"), only_main_content=Variable())

        agent = Agent(
            "a",
            config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="outcrawl_scrape"),
            tools=[scrape_tool],
            variables={"scrape_formats": ["markdown", "html"], "only_main_content": True},
        )
        await agent.ask("scrape")

        assert _request_body(route) == IsPartialDict({"formats": ["markdown", "html"], "onlyMainContent": True})

    @respx.mock
    async def test_single_string_variable_wrapped_in_list(self) -> None:
        route = respx.post(OUTCRAWL_SCRAPE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SCRAPE_RAW))
        toolkit = OutcrawlToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="outcrawl_scrape"),
            tools=[toolkit.scrape(formats=Variable())],
            variables={"formats": "links"},
        )
        await agent.ask("scrape")

        assert _request_body(route)["formats"] == ["links"]

    @respx.mock
    async def test_missing_raises(self) -> None:
        respx.post(OUTCRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = OutcrawlToolkit(api_key="test")
        search_tool = toolkit.search(limit=Variable())

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="outcrawl_search"),
            tools=[search_tool],
        )

        with pytest.raises(KeyError, match="limit"):
            await agent.ask("search")


@pytest.mark.asyncio
class TestIndividualTools:
    @respx.mock
    async def test_search_tool_passed_alone(self) -> None:
        route = respx.post(OUTCRAWL_SEARCH_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SEARCH_RAW))
        toolkit = OutcrawlToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="outcrawl_search"),
            tools=[toolkit.search()],
        )
        await agent.ask("search")

        assert route.called

    @respx.mock
    async def test_scrape_tool_passed_alone(self) -> None:
        route = respx.post(OUTCRAWL_SCRAPE_URL).mock(return_value=httpx.Response(200, json=SAMPLE_SCRAPE_RAW))
        toolkit = OutcrawlToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="outcrawl_scrape"),
            tools=[toolkit.scrape()],
        )
        await agent.ask("scrape")

        assert route.called
