# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest
import respx
from dirty_equals import IsPartialDict

from ag2 import Agent, Context, DataInput, Variable
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from ag2.extensions.tools.search.searchapi import (
    SearchApiNewsResult,
    SearchApiNewsSearchResponse,
    SearchApiScholarResult,
    SearchApiScholarSearchResponse,
    SearchApiSearchToolkit,
    SearchApiWebResult,
    SearchApiWebSearchResponse,
)
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.final.function_tool import FunctionToolSchema

SEARCH_URL = "https://www.searchapi.io/api/v1/search"


def _tool_call_config(
    arguments: dict[str, object],
    *,
    tool_name: str = "searchapi_web_search",
    final_reply: str = "done",
) -> TestConfig:
    return TestConfig(
        ModelResponse(
            tool_calls=ToolCallsEvent([
                ToolCallEvent(arguments=json.dumps(arguments), name=tool_name),
            ]),
        ),
        final_reply,
    )


def _page(count: int) -> dict[str, object]:
    return {"organic_results": [{"title": f"Result {i}", "link": f"https://example.com/{i}"} for i in range(count)]}


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schemas(self, context: Context) -> None:
        toolkit = SearchApiSearchToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))

        names = []
        for schema in schemas:
            assert isinstance(schema, FunctionToolSchema)
            names.append(schema.function.name)
            assert schema.function.parameters == IsPartialDict({
                "required": ["query"],
                "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
            })
        assert names == ["searchapi_web_search", "searchapi_news_search", "searchapi_scholar_search"]

    async def test_custom_name_and_description(self, context: Context) -> None:
        toolkit = SearchApiSearchToolkit(api_key="test")
        custom = toolkit.scholar(name="find_papers", description="Find academic papers.")

        [schema] = list(await custom.schemas(context))

        assert schema.function.name == "find_papers"
        assert schema.function.description == "Find academic papers."

    async def test_empty_api_key_raises_at_construction(self) -> None:
        with pytest.raises(ValueError, match="api_key is required"):
            SearchApiSearchToolkit("")


@pytest.mark.asyncio
class TestWebSearch:
    @respx.mock
    async def test_returns_structured_results(self) -> None:
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "search_metadata": {"id": "search_abc", "status": "Success"},
                    "organic_results": [
                        {
                            "position": 1,
                            "title": "AG2 docs",
                            "link": "https://docs.ag2.ai",
                            "source": "AG2",
                            "domain": "docs.ag2.ai",
                            "snippet": "AG2 documentation.",
                            "favicon": "data:image/png;base64,...",
                        },
                        {"title": "No snippet", "link": "https://example.com"},
                        "not-a-result",
                    ],
                    "related_searches": [],
                },
            )
        )
        toolkit = SearchApiSearchToolkit(api_key="test")
        config = TrackingConfig(_tool_call_config({"query": "AG2"}))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SearchApiWebSearchResponse(
                query="AG2",
                results=[
                    SearchApiWebResult(
                        title="AG2 docs",
                        link="https://docs.ag2.ai",
                        snippet="AG2 documentation.",
                        position=1,
                    ),
                    SearchApiWebResult(title="No snippet", link="https://example.com"),
                ],
            )
        )

    @respx.mock
    async def test_forwards_search_defaults_and_headers(self) -> None:
        route = respx.get(SEARCH_URL).mock(return_value=httpx.Response(200, json=_page(0)))
        toolkit = SearchApiSearchToolkit(api_key="test-key", num=5, gl="us", hl="en")
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        await agent.ask("search")

        request = route.calls.last.request
        # Google no longer honours `num` on web search, so it is not sent; the page is cut client-side.
        assert dict(request.url.params) == {"engine": "google", "q": "AG2", "link": "resolved", "gl": "us", "hl": "en"}
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["Accept"] == "application/json"
        assert request.headers["User-Agent"] == "ag2"

    @respx.mock
    async def test_key_travels_in_the_header_only(self) -> None:
        route = respx.get(SEARCH_URL).mock(return_value=httpx.Response(200, json=_page(0)))
        toolkit = SearchApiSearchToolkit(api_key="secret-key")
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        await agent.ask("search")

        assert "secret-key" not in str(route.calls.last.request.url)

    @respx.mock
    async def test_omits_unset_params(self) -> None:
        route = respx.get(SEARCH_URL).mock(return_value=httpx.Response(200, json=_page(0)))
        toolkit = SearchApiSearchToolkit(api_key="test")
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        await agent.ask("search")

        assert dict(route.calls.last.request.url.params) == {"engine": "google", "q": "AG2", "link": "resolved"}

    @respx.mock
    async def test_num_cuts_the_page_client_side(self) -> None:
        respx.get(SEARCH_URL).mock(return_value=httpx.Response(200, json=_page(10)))
        toolkit = SearchApiSearchToolkit(api_key="test", num=3)
        config = TrackingConfig(_tool_call_config({"query": "AG2"}))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SearchApiWebSearchResponse(
                query="AG2",
                results=[SearchApiWebResult(title=f"Result {i}", link=f"https://example.com/{i}") for i in range(3)],
            )
        )

    @respx.mock
    async def test_whole_page_when_num_is_unset(self) -> None:
        respx.get(SEARCH_URL).mock(return_value=httpx.Response(200, json=_page(10)))
        toolkit = SearchApiSearchToolkit(api_key="test")
        config = TrackingConfig(_tool_call_config({"query": "AG2"}))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SearchApiWebSearchResponse(
                query="AG2",
                results=[SearchApiWebResult(title=f"Result {i}", link=f"https://example.com/{i}") for i in range(10)],
            )
        )

    @respx.mock
    async def test_custom_base_url(self) -> None:
        route = respx.get("https://proxy.example.com/searchapi/api/v1/search").mock(
            return_value=httpx.Response(200, json=_page(0))
        )
        toolkit = SearchApiSearchToolkit(api_key="test", base_url="https://proxy.example.com/searchapi/")
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        await agent.ask("search")

        assert route.called

    @respx.mock
    async def test_http_error_raises(self) -> None:
        respx.get(SEARCH_URL).mock(return_value=httpx.Response(401, json={"error": "Invalid API key."}))
        toolkit = SearchApiSearchToolkit(api_key="bad")
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        with pytest.raises(httpx.HTTPStatusError):
            await agent.ask("search")


@pytest.mark.asyncio
class TestNewsSearch:
    @respx.mock
    async def test_returns_structured_results(self) -> None:
        route = respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "organic_results": [
                        {
                            "position": 1,
                            "title": "AG2 1.0 released",
                            "link": "https://news.example.com/ag2",
                            "source": "Example News",
                            "date": "2 hours ago",
                            "snippet": "The agent framework ships its first stable release.",
                            "thumbnail": "data:image/jpeg;base64,...",
                        },
                        {"title": "Bare article", "link": "https://news.example.com/bare", "source": {"name": "x"}},
                    ],
                    "top_stories": [{"title": "Ignored", "link": "https://news.example.com/top"}],
                },
            )
        )
        toolkit = SearchApiSearchToolkit(api_key="test", gl="us")
        config = TrackingConfig(_tool_call_config({"query": "AG2"}, tool_name="searchapi_news_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SearchApiNewsSearchResponse(
                query="AG2",
                results=[
                    SearchApiNewsResult(
                        title="AG2 1.0 released",
                        link="https://news.example.com/ag2",
                        snippet="The agent framework ships its first stable release.",
                        source="Example News",
                        date="2 hours ago",
                    ),
                    SearchApiNewsResult(title="Bare article", link="https://news.example.com/bare"),
                ],
            )
        )
        assert dict(route.calls.last.request.url.params) == {
            "engine": "google_news",
            "q": "AG2",
            "link": "resolved",
            "gl": "us",
        }

    @respx.mock
    async def test_num_cuts_the_page_client_side(self) -> None:
        route = respx.get(SEARCH_URL).mock(return_value=httpx.Response(200, json=_page(10)))
        toolkit = SearchApiSearchToolkit(api_key="test", num=3)
        config = TrackingConfig(_tool_call_config({"query": "AG2"}, tool_name="searchapi_news_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SearchApiNewsSearchResponse(
                query="AG2",
                results=[SearchApiNewsResult(title=f"Result {i}", link=f"https://example.com/{i}") for i in range(3)],
            )
        )
        # Google News does not take `num`, so it must not be sent.
        assert dict(route.calls.last.request.url.params) == {"engine": "google_news", "q": "AG2", "link": "resolved"}

    @respx.mock
    async def test_num_resolves_from_a_variable(self) -> None:
        respx.get(SEARCH_URL).mock(return_value=httpx.Response(200, json=_page(10)))
        toolkit = SearchApiSearchToolkit(api_key="test")
        config = TrackingConfig(_tool_call_config({"query": "AG2"}, tool_name="searchapi_news_search"))
        agent = Agent("a", config=config, tools=[toolkit.news(num=Variable("cap"))], variables={"cap": 2})

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SearchApiNewsSearchResponse(
                query="AG2",
                results=[SearchApiNewsResult(title=f"Result {i}", link=f"https://example.com/{i}") for i in range(2)],
            )
        )


@pytest.mark.asyncio
class TestScholarSearch:
    @respx.mock
    async def test_returns_structured_results(self) -> None:
        route = respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "organic_results": [
                        {
                            "position": 1,
                            "title": "JADE - A FIPA-compliant agent framework",
                            "data_cid": "abc123",
                            "link": "https://example.org/jade",
                            "publication": "F Bellifemine, A Poggi, G Rimassa - Proceedings of PAAM, 1999 - example.org",
                            "snippet": "JADE is a software framework for agent applications.",
                            "inline_links": {
                                "cited_by": {"cites_id": "1", "total": 986, "link": "https://example.org/cites"},
                                "cached_page_link": "https://example.org/cache",
                            },
                            "resource": {
                                "name": "example.org",
                                "format": "PDF",
                                "link": "https://example.org/jade.pdf",
                            },
                            "authors": [
                                {"name": "F Bellifemine", "id": "a1", "link": "https://example.org/a1"},
                                {"name": "A Poggi", "id": "a2"},
                                {"id": "a3"},
                                "not-an-author",
                            ],
                        },
                        {"title": "Untracked paper", "link": "https://example.org/paper"},
                    ]
                },
            )
        )
        toolkit = SearchApiSearchToolkit(api_key="test", num=10, gl="us", hl="en")
        config = TrackingConfig(_tool_call_config({"query": "agent frameworks"}, tool_name="searchapi_scholar_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SearchApiScholarSearchResponse(
                query="agent frameworks",
                results=[
                    SearchApiScholarResult(
                        title="JADE - A FIPA-compliant agent framework",
                        link="https://example.org/jade",
                        snippet="JADE is a software framework for agent applications.",
                        publication="F Bellifemine, A Poggi, G Rimassa - Proceedings of PAAM, 1999 - example.org",
                        authors=["F Bellifemine", "A Poggi"],
                        pdf_link="https://example.org/jade.pdf",
                        citations=986,
                    ),
                    SearchApiScholarResult(title="Untracked paper", link="https://example.org/paper"),
                ],
            )
        )
        # Scholar takes `num` server-side and has no `gl`.
        assert dict(route.calls.last.request.url.params) == {
            "engine": "google_scholar",
            "q": "agent frameworks",
            "num": "10",
            "hl": "en",
        }


@pytest.mark.asyncio
class TestVariables:
    @respx.mock
    async def test_resolves_runtime_values(self) -> None:
        route = respx.get(SEARCH_URL).mock(return_value=httpx.Response(200, json=_page(0)))
        toolkit = SearchApiSearchToolkit(api_key="test")
        search_tool = toolkit.scholar(hl=Variable(), num=Variable("result_limit"))
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "AG2"}, tool_name="searchapi_scholar_search"),
            tools=[search_tool],
            variables={"hl": "de", "result_limit": 10},
        )

        await agent.ask("search")

        assert dict(route.calls.last.request.url.params) == {
            "engine": "google_scholar",
            "q": "AG2",
            "num": "10",
            "hl": "de",
        }

    async def test_missing_variable_raises(self) -> None:
        toolkit = SearchApiSearchToolkit(api_key="test")
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "AG2"}),
            tools=[toolkit.web(num=Variable("result_limit"))],
        )

        with pytest.raises(KeyError):
            await agent.ask("search")


@pytest.mark.asyncio
class TestPayloadHandling:
    @respx.mock
    async def test_boolean_citation_count_is_dropped(self) -> None:
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "organic_results": [
                        {
                            "title": "A paper",
                            "link": "https://example.org/paper",
                            "inline_links": {"cited_by": {"total": True}},
                            "resource": "not-a-resource",
                            "authors": "not-a-list",
                        }
                    ]
                },
            )
        )
        toolkit = SearchApiSearchToolkit(api_key="test")
        config = TrackingConfig(_tool_call_config({"query": "papers"}, tool_name="searchapi_scholar_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SearchApiScholarSearchResponse(
                query="papers",
                results=[SearchApiScholarResult(title="A paper", link="https://example.org/paper")],
            )
        )

    @respx.mock
    async def test_javascript_resource_stub_is_dropped(self) -> None:
        # Google sometimes serves a `javascript:void(0)` stub glued to its host as the resource link.
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "organic_results": [
                        {
                            "title": "A paper",
                            "link": "https://example.org/paper",
                            "resource": {"name": "example.org", "link": "https://scholar.google.comjavascript:void(0)"},
                        }
                    ]
                },
            )
        )
        toolkit = SearchApiSearchToolkit(api_key="test")
        config = TrackingConfig(_tool_call_config({"query": "papers"}, tool_name="searchapi_scholar_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SearchApiScholarSearchResponse(
                query="papers",
                results=[SearchApiScholarResult(title="A paper", link="https://example.org/paper")],
            )
        )

    @respx.mock
    async def test_non_object_payload_yields_no_results(self) -> None:
        respx.get(SEARCH_URL).mock(return_value=httpx.Response(200, json=["unexpected"]))
        toolkit = SearchApiSearchToolkit(api_key="test")
        config = TrackingConfig(_tool_call_config({"query": "AG2"}))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SearchApiWebSearchResponse(query="AG2", results=[])
        )

    @respx.mock
    async def test_missing_results_key_yields_no_results(self) -> None:
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"search_metadata": {"status": "Success"}, "organic_results": None})
        )
        toolkit = SearchApiSearchToolkit(api_key="test")
        config = TrackingConfig(_tool_call_config({"query": "AG2"}))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SearchApiWebSearchResponse(query="AG2", results=[])
        )

    @respx.mock
    async def test_non_positive_num_returns_the_whole_page(self) -> None:
        respx.get(SEARCH_URL).mock(return_value=httpx.Response(200, json=_page(4)))
        toolkit = SearchApiSearchToolkit(api_key="test", num=0)
        config = TrackingConfig(_tool_call_config({"query": "AG2"}))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SearchApiWebSearchResponse(
                query="AG2",
                results=[SearchApiWebResult(title=f"Result {i}", link=f"https://example.com/{i}") for i in range(4)],
            )
        )


@pytest.mark.asyncio
class TestConnection:
    @respx.mock
    async def test_duplicate_trailing_slashes_are_stripped_from_base_url(self) -> None:
        route = respx.get("https://proxy.example.com/searchapi/api/v1/search").mock(
            return_value=httpx.Response(200, json=_page(0))
        )
        toolkit = SearchApiSearchToolkit(api_key="test", base_url="https://proxy.example.com/searchapi//")
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        await agent.ask("search")

        assert route.called

    @respx.mock
    async def test_accepts_proxy_and_tls_verify_settings(self) -> None:
        route = respx.get(SEARCH_URL).mock(return_value=httpx.Response(200, json=_page(0)))
        toolkit = SearchApiSearchToolkit(api_key="test", proxy="http://proxy.example.com:3128", verify=False)
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        await agent.ask("search")

        assert route.called
