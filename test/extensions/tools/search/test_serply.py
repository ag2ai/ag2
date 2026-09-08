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
from ag2.extensions.tools.search.serply import (
    SerplyNewsResult,
    SerplyNewsSearchResponse,
    SerplyScholarResult,
    SerplyScholarSearchResponse,
    SerplySearchToolkit,
    SerplyWebResult,
    SerplyWebSearchResponse,
)
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.final.function_tool import FunctionToolSchema

SERPLY_BASE_URL = "https://api.serply.io"


def _tool_call_config(
    arguments: dict[str, object],
    *,
    tool_name: str = "serply_web_search",
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


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schemas(self, context: Context) -> None:
        toolkit = SerplySearchToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))

        names = []
        for schema in schemas:
            assert isinstance(schema, FunctionToolSchema)
            names.append(schema.function.name)
            assert schema.function.parameters == IsPartialDict({
                "required": ["query"],
                "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
            })
        assert names == ["serply_web_search", "serply_news_search", "serply_scholar_search"]

    async def test_custom_name_and_description(self, context: Context) -> None:
        toolkit = SerplySearchToolkit(api_key="test")
        custom = toolkit.scholar(name="find_papers", description="Find academic papers.")

        [schema] = list(await custom.schemas(context))

        assert schema.function.name == "find_papers"
        assert schema.function.description == "Find academic papers."

    async def test_empty_api_key_raises_at_construction(self) -> None:
        with pytest.raises(ValueError, match="api_key is required"):
            SerplySearchToolkit("")


@pytest.mark.asyncio
class TestWebSearch:
    @respx.mock
    async def test_returns_structured_results(self) -> None:
        respx.get(f"{SERPLY_BASE_URL}/v1/search/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "AG2 docs",
                            "link": "https://docs.ag2.ai",
                            "description": "AG2 documentation.",
                            "position": 1,
                            "metadata": {"display_url": "docs.ag2.ai"},
                        },
                        {"title": "No description", "link": "https://example.com"},
                        "not-a-result",
                    ],
                    "related_searches": [],
                },
            )
        )
        toolkit = SerplySearchToolkit(api_key="test")
        config = TrackingConfig(_tool_call_config({"query": "AG2"}))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SerplyWebSearchResponse(
                query="AG2",
                results=[
                    SerplyWebResult(
                        title="AG2 docs",
                        link="https://docs.ag2.ai",
                        description="AG2 documentation.",
                        position=1,
                    ),
                    SerplyWebResult(title="No description", link="https://example.com"),
                ],
            )
        )

    @respx.mock
    async def test_forwards_search_defaults_and_headers(self) -> None:
        route = respx.get(f"{SERPLY_BASE_URL}/v1/search/").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = SerplySearchToolkit(api_key="test-key", num=5, gl="us", hl="en")
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        await agent.ask("search")

        assert dict(route.calls.last.request.url.params) == {"q": "AG2", "num": "5", "gl": "us", "hl": "en"}
        assert route.calls.last.request.headers["X-Api-Key"] == "test-key"
        assert route.calls.last.request.headers["User-Agent"] == "ag2-serply-extension"

    @respx.mock
    async def test_omits_unset_params(self) -> None:
        route = respx.get(f"{SERPLY_BASE_URL}/v1/search/").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = SerplySearchToolkit(api_key="test")
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        await agent.ask("search")

        assert dict(route.calls.last.request.url.params) == {"q": "AG2"}

    @respx.mock
    async def test_custom_base_url(self) -> None:
        route = respx.get("https://proxy.example.com/serply/v1/search/").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        toolkit = SerplySearchToolkit(api_key="test", base_url="https://proxy.example.com/serply/")
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        await agent.ask("search")

        assert route.called

    @respx.mock
    async def test_http_error_raises(self) -> None:
        respx.get(f"{SERPLY_BASE_URL}/v1/search/").mock(
            return_value=httpx.Response(401, json={"detail": "Invalid API key"})
        )
        toolkit = SerplySearchToolkit(api_key="bad")
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        with pytest.raises(httpx.HTTPStatusError):
            await agent.ask("search")


@pytest.mark.asyncio
class TestNewsSearch:
    @respx.mock
    async def test_returns_structured_results(self) -> None:
        route = respx.get(f"{SERPLY_BASE_URL}/v1/news/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "entries": [
                        {
                            "title": "AG2 1.0 released - Example News",
                            "link": "https://news.example.com/ag2",
                            "published": "Wed, 26 Aug 2026 12:00:00 GMT",
                            "source": {"href": "https://news.example.com", "title": "Example News"},
                        }
                    ]
                },
            )
        )
        toolkit = SerplySearchToolkit(api_key="test", gl="us")
        config = TrackingConfig(_tool_call_config({"query": "AG2"}, tool_name="serply_news_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SerplyNewsSearchResponse(
                query="AG2",
                results=[
                    SerplyNewsResult(
                        title="AG2 1.0 released - Example News",
                        link="https://news.example.com/ag2",
                        published="Wed, 26 Aug 2026 12:00:00 GMT",
                        source="Example News",
                    )
                ],
            )
        )
        assert dict(route.calls.last.request.url.params) == {"q": "AG2", "gl": "us"}


@pytest.mark.asyncio
class TestNewsResultCap:
    @staticmethod
    def _feed(count: int) -> dict[str, object]:
        return {"entries": [{"title": f"Article {i}", "link": f"https://news.example.com/{i}"} for i in range(count)]}

    @respx.mock
    async def test_caps_the_feed_when_num_is_unset(self) -> None:
        respx.get(f"{SERPLY_BASE_URL}/v1/news/").mock(return_value=httpx.Response(200, json=self._feed(60)))
        toolkit = SerplySearchToolkit(api_key="test")
        config = TrackingConfig(_tool_call_config({"query": "AG2"}, tool_name="serply_news_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SerplyNewsSearchResponse(
                query="AG2",
                results=[
                    SerplyNewsResult(title=f"Article {i}", link=f"https://news.example.com/{i}") for i in range(10)
                ],
            )
        )

    @respx.mock
    async def test_num_truncates_the_feed_client_side(self) -> None:
        route = respx.get(f"{SERPLY_BASE_URL}/v1/news/").mock(return_value=httpx.Response(200, json=self._feed(60)))
        toolkit = SerplySearchToolkit(api_key="test", num=3)
        config = TrackingConfig(_tool_call_config({"query": "AG2"}, tool_name="serply_news_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SerplyNewsSearchResponse(
                query="AG2",
                results=[
                    SerplyNewsResult(title=f"Article {i}", link=f"https://news.example.com/{i}") for i in range(3)
                ],
            )
        )
        # Google News ignores `num` server-side, so it must not be sent.
        assert dict(route.calls.last.request.url.params) == {"q": "AG2"}

    @respx.mock
    async def test_num_resolves_from_a_variable(self) -> None:
        respx.get(f"{SERPLY_BASE_URL}/v1/news/").mock(return_value=httpx.Response(200, json=self._feed(60)))
        toolkit = SerplySearchToolkit(api_key="test")
        config = TrackingConfig(_tool_call_config({"query": "AG2"}, tool_name="serply_news_search"))
        agent = Agent("a", config=config, tools=[toolkit.news(num=Variable("cap"))], variables={"cap": 2})

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SerplyNewsSearchResponse(
                query="AG2",
                results=[
                    SerplyNewsResult(title=f"Article {i}", link=f"https://news.example.com/{i}") for i in range(2)
                ],
            )
        )


@pytest.mark.asyncio
class TestScholarSearch:
    @respx.mock
    async def test_returns_structured_results(self) -> None:
        respx.get(f"{SERPLY_BASE_URL}/v1/scholar/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "articles": [
                        {
                            "title": "JADE - A FIPA-compliant agent framework",
                            "link": "https://example.org/jade.pdf",
                            "author": {
                                "names": "F. Bellifemine, A. Poggi, G. Rimassa - 1999",
                                "authors": [{"name": "F. Bellifemine", "link": "https://openalex.org/A1"}],
                            },
                            "description": "F. Bellifemine, A. Poggi, G. Rimassa - 1999",
                            "doc": {"link": "https://example.org/jade-oa.pdf", "type": "PDF"},
                            "extras": {"citations": {"count": 986, "link": "https://example.org/cites"}},
                        },
                        {"title": "Untracked paper", "link": "https://example.org/paper"},
                    ]
                },
            )
        )
        toolkit = SerplySearchToolkit(api_key="test")
        config = TrackingConfig(_tool_call_config({"query": "agent frameworks"}, tool_name="serply_scholar_search"))
        agent = Agent("a", config=config, tools=[toolkit.scholar()])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SerplyScholarSearchResponse(
                query="agent frameworks",
                results=[
                    SerplyScholarResult(
                        title="JADE - A FIPA-compliant agent framework",
                        link="https://example.org/jade.pdf",
                        authors="F. Bellifemine, A. Poggi, G. Rimassa - 1999",
                        pdf_link="https://example.org/jade-oa.pdf",
                        citations=986,
                    ),
                    SerplyScholarResult(title="Untracked paper", link="https://example.org/paper"),
                ],
            )
        )


@pytest.mark.asyncio
class TestVariables:
    @respx.mock
    async def test_resolves_runtime_values(self) -> None:
        route = respx.get(f"{SERPLY_BASE_URL}/v1/search/").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = SerplySearchToolkit(api_key="test")
        search_tool = toolkit.web(gl=Variable(), num=Variable("result_limit"))
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "AG2"}),
            tools=[search_tool],
            variables={"gl": "de", "result_limit": 10},
        )

        await agent.ask("search")

        assert dict(route.calls.last.request.url.params) == {"q": "AG2", "gl": "de", "num": "10"}

    async def test_missing_variable_raises(self) -> None:
        toolkit = SerplySearchToolkit(api_key="test")
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
        respx.get(f"{SERPLY_BASE_URL}/v1/scholar/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "articles": [
                        {
                            "title": "A paper",
                            "link": "https://example.org/paper",
                            "extras": {"citations": {"count": True}},
                        }
                    ]
                },
            )
        )
        toolkit = SerplySearchToolkit(api_key="test")
        config = TrackingConfig(_tool_call_config({"query": "papers"}, tool_name="serply_scholar_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SerplyScholarSearchResponse(
                query="papers",
                results=[SerplyScholarResult(title="A paper", link="https://example.org/paper")],
            )
        )

    @respx.mock
    async def test_non_object_payload_yields_no_results(self) -> None:
        respx.get(f"{SERPLY_BASE_URL}/v1/search/").mock(return_value=httpx.Response(200, json=["unexpected"]))
        toolkit = SerplySearchToolkit(api_key="test")
        config = TrackingConfig(_tool_call_config({"query": "AG2"}))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SerplyWebSearchResponse(query="AG2", results=[])
        )


@pytest.mark.asyncio
class TestConnection:
    @respx.mock
    async def test_duplicate_trailing_slashes_are_stripped_from_base_url(self) -> None:
        route = respx.get("https://proxy.example.com/serply/v1/search/").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        toolkit = SerplySearchToolkit(api_key="test", base_url="https://proxy.example.com/serply//")
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        await agent.ask("search")

        assert route.called

    @respx.mock
    async def test_accepts_proxy_and_tls_verify_settings(self) -> None:
        route = respx.get(f"{SERPLY_BASE_URL}/v1/search/").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = SerplySearchToolkit(api_key="test", proxy="http://proxy.example.com:3128", verify=False)
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        await agent.ask("search")

        assert route.called

    @respx.mock
    async def test_forwards_proxy_location_header(self) -> None:
        route = respx.get(f"{SERPLY_BASE_URL}/v1/news/").mock(return_value=httpx.Response(200, json={"entries": []}))
        toolkit = SerplySearchToolkit(api_key="test", proxy_location="US")
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "AG2"}, tool_name="serply_news_search"),
            tools=[toolkit],
        )

        await agent.ask("search")

        assert route.calls.last.request.headers["X-Proxy-Location"] == "US"

    @respx.mock
    async def test_omits_unset_proxy_location_header(self) -> None:
        route = respx.get(f"{SERPLY_BASE_URL}/v1/search/").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = SerplySearchToolkit(api_key="test")
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        await agent.ask("search")

        assert "X-Proxy-Location" not in route.calls.last.request.headers

    @respx.mock
    async def test_resolves_header_variable(self) -> None:
        route = respx.get(f"{SERPLY_BASE_URL}/v1/scholar/").mock(
            return_value=httpx.Response(200, json={"articles": []})
        )
        toolkit = SerplySearchToolkit(api_key="test")
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "AG2"}, tool_name="serply_scholar_search"),
            tools=[toolkit.scholar(proxy_location=Variable("region"))],
            variables={"region": "DE"},
        )

        await agent.ask("search")

        assert route.calls.last.request.headers["X-Proxy-Location"] == "DE"
