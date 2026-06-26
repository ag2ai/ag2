# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest
import respx
from dirty_equals import IsPartialDict

pytest.importorskip("exa_py")

from ag2 import Agent, Context, DataInput, Variable
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from ag2.extensions.tools.search.exa import (
    ExaAnswerCitation,
    ExaAnswerResult,
    ExaContentResult,
    ExaSearchResponse,
    ExaSearchResult,
    ExaToolkit,
)
from ag2.testing import TestConfig, TrackingConfig

EXA_BASE_URL = "https://api.exa.ai"


def _tool_call_config(
    arguments: dict,
    *,
    tool_name: str,
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


def _exa_result(
    *,
    title: str = "AG2 Framework",
    url: str = "https://ag2.ai",
    score: float | None = 0.95,
    published_date: str | None = "2024-01-01",
    author: str | None = "ag2ai",
    text: str | None = None,
) -> dict[str, Any]:
    return {
        "title": title,
        "url": url,
        "score": score,
        "publishedDate": published_date,
        "author": author,
        "text": text,
    }


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schemas(self, context: Context) -> None:
        toolkit = ExaToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))

        names = [s.function.name for s in schemas]
        assert names == ["exa_search", "exa_find_similar", "exa_get_contents", "exa_answer"]

    async def test_search_schema_has_query_param(self, context: Context) -> None:
        toolkit = ExaToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))
        search_schema = next(s for s in schemas if s.function.name == "exa_search")

        assert search_schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })

    async def test_custom_tool_name_and_description(self, context: Context) -> None:
        toolkit = ExaToolkit(api_key="test")
        custom = toolkit.search(name="neural_search", description="Custom neural search.")

        [schema] = list(await custom.schemas(context))

        assert schema.function.name == "neural_search"
        assert schema.function.description == "Custom neural search."


@pytest.mark.asyncio
class TestSearchExecution:
    @respx.mock
    async def test_returns_structured_results(self) -> None:
        respx.post(f"{EXA_BASE_URL}/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        _exa_result(title="AG2 Framework", url="https://ag2.ai", score=0.95),
                        _exa_result(title="GitHub - AG2", url="https://github.com/ag2ai/ag2", score=0.82),
                    ],
                },
            )
        )
        toolkit = ExaToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "AG2 framework"}, tool_name="exa_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            ExaSearchResponse(
                query="AG2 framework",
                results=[
                    ExaSearchResult(
                        title="AG2 Framework",
                        url="https://ag2.ai",
                        score=0.95,
                        published_date="2024-01-01",
                        author="ag2ai",
                        text=None,
                    ),
                    ExaSearchResult(
                        title="GitHub - AG2",
                        url="https://github.com/ag2ai/ag2",
                        score=0.82,
                        published_date="2024-01-01",
                        author="ag2ai",
                        text=None,
                    ),
                ],
            )
        )

    @respx.mock
    async def test_empty_results(self) -> None:
        respx.post(f"{EXA_BASE_URL}/search").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = ExaToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "nothing"}, tool_name="exa_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(ExaSearchResponse(query="nothing"))

    @respx.mock
    async def test_none_params_omitted(self) -> None:
        route = respx.post(f"{EXA_BASE_URL}/search").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = ExaToolkit(api_key="test")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="exa_search"), tools=[toolkit])
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({"query": "q"})

    @respx.mock
    async def test_all_params_forwarded(self) -> None:
        route = respx.post(f"{EXA_BASE_URL}/search").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = ExaToolkit(api_key="test")
        search_tool = toolkit.search(
            num_results=7,
            search_type="neural",
            category="research paper",
            include_domains=["arxiv.org"],
            exclude_domains=["medium.com"],
            start_published_date="2024-01-01",
            end_published_date="2024-12-31",
            start_crawl_date="2024-01-01",
            end_crawl_date="2024-12-31",
            livecrawl="always",
            user_location="US",
            moderation=True,
        )

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="exa_search"), tools=[search_tool])
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({
            "query": "q",
            "numResults": 7,
            "type": "neural",
            "category": "research paper",
            "includeDomains": ["arxiv.org"],
            "excludeDomains": ["medium.com"],
            "startPublishedDate": "2024-01-01",
            "endPublishedDate": "2024-12-31",
            "startCrawlDate": "2024-01-01",
            "endCrawlDate": "2024-12-31",
            "contents": IsPartialDict({"livecrawl": "always"}),
        })

    @respx.mock
    async def test_search_and_contents_when_max_characters_set_on_method(self) -> None:
        route = respx.post(f"{EXA_BASE_URL}/search").mock(
            return_value=httpx.Response(200, json={"results": [_exa_result(text="full article text")]})
        )
        toolkit = ExaToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="exa_search"),
            tools=[toolkit.search(max_characters=500)],
        )
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({
            "query": "q",
            "contents": IsPartialDict({"text": IsPartialDict({"maxCharacters": 500})}),
        })

    @respx.mock
    async def test_toolkit_level_max_characters_applied_to_default_search(self) -> None:
        route = respx.post(f"{EXA_BASE_URL}/search").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = ExaToolkit(api_key="test", max_characters=800)

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="exa_search"), tools=[toolkit])
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({
            "query": "q",
            "contents": IsPartialDict({"text": IsPartialDict({"maxCharacters": 800})}),
        })

    @respx.mock
    async def test_toolkit_level_num_results_applied_to_default_search(self) -> None:
        route = respx.post(f"{EXA_BASE_URL}/search").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = ExaToolkit(api_key="test", num_results=7)

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="exa_search"), tools=[toolkit])
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({"query": "q", "numResults": 7})


@pytest.mark.asyncio
class TestFindSimilar:
    @respx.mock
    async def test_num_results_forwarded_from_method(self) -> None:
        route = respx.post(f"{EXA_BASE_URL}/findSimilar").mock(
            return_value=httpx.Response(200, json={"results": [_exa_result(title="Similar page")]})
        )
        toolkit = ExaToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="exa_find_similar"),
            tools=[toolkit.find_similar(num_results=3)],
        )
        await agent.ask("find similar")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({"url": "https://ag2.ai", "numResults": 3})

    @respx.mock
    async def test_num_results_forwarded_from_toolkit(self) -> None:
        route = respx.post(f"{EXA_BASE_URL}/findSimilar").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = ExaToolkit(api_key="test", num_results=4)

        agent = Agent(
            "a",
            config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="exa_find_similar"),
            tools=[toolkit],
        )
        await agent.ask("find similar")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({"url": "https://ag2.ai", "numResults": 4})

    @respx.mock
    async def test_exclude_source_domain_forwarded(self) -> None:
        route = respx.post(f"{EXA_BASE_URL}/findSimilar").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = ExaToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="exa_find_similar"),
            tools=[toolkit.find_similar(num_results=2, exclude_source_domain=True)],
        )
        await agent.ask("find similar")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({
            "url": "https://ag2.ai",
            "numResults": 2,
            "excludeSourceDomain": True,
        })


@pytest.mark.asyncio
class TestGetContents:
    @respx.mock
    async def test_returns_content(self) -> None:
        route = respx.post(f"{EXA_BASE_URL}/contents").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://ag2.ai",
                            "title": "AG2",
                            "text": "full text",
                            "author": "ag2ai",
                            "publishedDate": "2024-01-01",
                        }
                    ]
                },
            )
        )
        toolkit = ExaToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"urls": ["https://ag2.ai"]}, tool_name="exa_get_contents"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("get contents")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({"urls": ["https://ag2.ai"], "text": True})

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput([
            ExaContentResult(
                url="https://ag2.ai",
                title="AG2",
                text="full text",
                author="ag2ai",
                published_date="2024-01-01",
            )
        ])


@pytest.mark.asyncio
class TestAnswer:
    @respx.mock
    async def test_returns_answer_with_citations(self) -> None:
        route = respx.post(f"{EXA_BASE_URL}/answer").mock(
            return_value=httpx.Response(
                200,
                json={
                    "answer": "AG2 is an open-source multi-agent framework.",
                    "citations": [
                        {"url": "https://ag2.ai", "title": "AG2", "text": "About AG2"},
                        {"url": "https://github.com/ag2ai/ag2", "title": "GitHub", "text": "Source code"},
                    ],
                },
            )
        )
        toolkit = ExaToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "What is AG2?"}, tool_name="exa_answer"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("answer")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({"query": "What is AG2?", "text": True})

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            ExaAnswerResult(
                answer="AG2 is an open-source multi-agent framework.",
                citations=[
                    ExaAnswerCitation(url="https://ag2.ai", title="AG2", text="About AG2"),
                    ExaAnswerCitation(url="https://github.com/ag2ai/ag2", title="GitHub", text="Source code"),
                ],
            )
        )


@pytest.mark.asyncio
class TestExaToolkitVariable:
    @respx.mock
    async def test_resolved(self) -> None:
        route = respx.post(f"{EXA_BASE_URL}/search").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = ExaToolkit(api_key="test")
        search_tool = toolkit.search(
            num_results=Variable("user_limit"),
            search_type=Variable(),
        )

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="exa_search"),
            tools=[search_tool],
            variables={"user_limit": 10, "search_type": "neural"},
        )
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({"query": "q", "numResults": 10, "type": "neural"})

    @respx.mock
    async def test_missing_raises(self) -> None:
        respx.post(f"{EXA_BASE_URL}/search").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = ExaToolkit(api_key="test")
        search_tool = toolkit.search(search_type=Variable())

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="exa_search"),
            tools=[search_tool],
        )

        with pytest.raises(KeyError, match="search_type"):
            await agent.ask("search")


@pytest.mark.asyncio
class TestIntegrationHeader:
    @respx.mock
    async def test_search_sets_header(self) -> None:
        route = respx.post(f"{EXA_BASE_URL}/search").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = ExaToolkit(api_key="test")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="exa_search"), tools=[toolkit])
        await agent.ask("search")

        assert route.calls.last.request.headers["x-exa-integration"] == "ag2"

    @respx.mock
    async def test_find_similar_sets_header(self) -> None:
        route = respx.post(f"{EXA_BASE_URL}/findSimilar").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = ExaToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="exa_find_similar"),
            tools=[toolkit],
        )
        await agent.ask("find similar")

        assert route.calls.last.request.headers["x-exa-integration"] == "ag2"

    @respx.mock
    async def test_get_contents_sets_header(self) -> None:
        route = respx.post(f"{EXA_BASE_URL}/contents").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = ExaToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"urls": ["https://ag2.ai"]}, tool_name="exa_get_contents"),
            tools=[toolkit],
        )
        await agent.ask("get contents")

        assert route.calls.last.request.headers["x-exa-integration"] == "ag2"

    @respx.mock
    async def test_answer_sets_header(self) -> None:
        route = respx.post(f"{EXA_BASE_URL}/answer").mock(
            return_value=httpx.Response(200, json={"answer": "ok", "citations": []})
        )
        toolkit = ExaToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "What is AG2?"}, tool_name="exa_answer"),
            tools=[toolkit],
        )
        await agent.ask("answer")

        assert route.calls.last.request.headers["x-exa-integration"] == "ag2"


@pytest.mark.asyncio
class TestIndividualTools:
    @respx.mock
    async def test_search_tool_passed_alone(self) -> None:
        route = respx.post(f"{EXA_BASE_URL}/search").mock(
            return_value=httpx.Response(200, json={"results": [_exa_result()]})
        )
        toolkit = ExaToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="exa_search"),
            tools=[toolkit.search()],
        )
        await agent.ask("search")

        assert route.called

    @respx.mock
    async def test_pick_two_tools_from_toolkit(self) -> None:
        search_route = respx.post(f"{EXA_BASE_URL}/search").mock(return_value=httpx.Response(200, json={"results": []}))
        respx.post(f"{EXA_BASE_URL}/answer").mock(
            return_value=httpx.Response(200, json={"answer": "ok", "citations": []})
        )
        toolkit = ExaToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="exa_search"),
            tools=[toolkit.search(), toolkit.answer()],
        )
        await agent.ask("search")

        assert search_route.called
