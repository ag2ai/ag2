# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest
import respx
from dirty_equals import IsPartialDict

pytest.importorskip("perplexity")

from ag2 import Agent, Context, DataInput, ImageInput, Variable
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.search.perplexity import (
    PerplexityImageMeta,
    PerplexitySearchResponse,
    PerplexitySearchResult,
    PerplexitySearchToolkit,
)

PERPLEXITY_BASE_URL = "https://api.perplexity.ai"


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


def _search_response(results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"id": "search-xxx", "results": results or []}


def _chat_response(
    content: str = "",
    *,
    search_results: list[dict[str, Any]] | None = None,
    citations: list[str] | None = None,
    images: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-xxx",
        "model": "sonar",
        "created": 1700000000,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
    if search_results is not None:
        body["search_results"] = search_results
    if citations is not None:
        body["citations"] = citations
    if images is not None:
        body["images"] = images
    return body


SAMPLE_SEARCH_RESULTS = [
    {
        "title": "AG2 Framework",
        "url": "https://ag2.ai",
        "snippet": "AG2 is an agent framework.",
        "date": "2026-01-01",
    },
    {
        "title": "GitHub - AG2",
        "url": "https://github.com/ag2ai/ag2",
        "snippet": "Open source repo.",
        "date": None,
    },
]


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schemas(self, context: Context) -> None:
        toolkit = PerplexitySearchToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))

        names = [s.function.name for s in schemas]
        assert names == ["perplexity_search", "perplexity_answer"]

    async def test_search_schema_has_query_param(self, context: Context) -> None:
        toolkit = PerplexitySearchToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))
        search_schema = next(s for s in schemas if s.function.name == "perplexity_search")

        assert search_schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })

    async def test_answer_schema_has_query_param(self, context: Context) -> None:
        toolkit = PerplexitySearchToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))
        answer_schema = next(s for s in schemas if s.function.name == "perplexity_answer")

        assert answer_schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })

    async def test_custom_search_name_and_description(self, context: Context) -> None:
        toolkit = PerplexitySearchToolkit(api_key="test")
        custom = toolkit.search(name="web_search", description="Custom search.")

        [schema] = list(await custom.schemas(context))

        assert schema.function.name == "web_search"
        assert schema.function.description == "Custom search."

    async def test_custom_answer_name_and_description(self, context: Context) -> None:
        toolkit = PerplexitySearchToolkit(api_key="test")
        custom = toolkit.answer(name="ask_perplexity", description="Custom answer.")

        [schema] = list(await custom.schemas(context))

        assert schema.function.name == "ask_perplexity"
        assert schema.function.description == "Custom answer."


@pytest.mark.asyncio
class TestSearch:
    @respx.mock
    async def test_returns_structured_results(self) -> None:
        respx.post(f"{PERPLEXITY_BASE_URL}/search").mock(
            return_value=httpx.Response(200, json=_search_response(SAMPLE_SEARCH_RESULTS))
        )
        toolkit = PerplexitySearchToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "AG2 framework"}, tool_name="perplexity_search"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            PerplexitySearchResponse(
                query="AG2 framework",
                results=[
                    PerplexitySearchResult(
                        title="AG2 Framework",
                        url="https://ag2.ai",
                        snippet="AG2 is an agent framework.",
                        date="2026-01-01",
                    ),
                    PerplexitySearchResult(
                        title="GitHub - AG2",
                        url="https://github.com/ag2ai/ag2",
                        snippet="Open source repo.",
                        date=None,
                    ),
                ],
            )
        )

    @respx.mock
    async def test_empty_results(self) -> None:
        respx.post(f"{PERPLEXITY_BASE_URL}/search").mock(return_value=httpx.Response(200, json=_search_response()))
        toolkit = PerplexitySearchToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "nothing"}, tool_name="perplexity_search"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            PerplexitySearchResponse(query="nothing", results=[])
        )

    @respx.mock
    async def test_none_params_omitted(self) -> None:
        route = respx.post(f"{PERPLEXITY_BASE_URL}/search").mock(
            return_value=httpx.Response(200, json=_search_response())
        )
        toolkit = PerplexitySearchToolkit(api_key="test")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="perplexity_search"), tools=[toolkit])
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == {"query": "q"}

    @respx.mock
    async def test_all_params_forwarded(self) -> None:
        route = respx.post(f"{PERPLEXITY_BASE_URL}/search").mock(
            return_value=httpx.Response(200, json=_search_response())
        )
        toolkit = PerplexitySearchToolkit(api_key="test")
        search_tool = toolkit.search(
            max_results=5,
            max_tokens_per_page=512,
            search_domain_filter=["arxiv.org", "-medium.com"],
            search_recency_filter="week",
            search_after_date_filter="1/1/2025",
            search_before_date_filter="12/31/2025",
        )

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="perplexity_search"),
            tools=[search_tool],
        )
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == {
            "query": "q",
            "max_results": 5,
            "max_tokens_per_page": 512,
            "search_domain_filter": ["arxiv.org", "-medium.com"],
            "search_recency_filter": "week",
            "search_after_date_filter": "1/1/2025",
            "search_before_date_filter": "12/31/2025",
        }

    @respx.mock
    async def test_client_kwargs_forwarded_to_sdk(self) -> None:
        custom_url = "https://custom.perplexity.example"
        route = respx.post(f"{custom_url}/search").mock(return_value=httpx.Response(200, json=_search_response()))
        toolkit = PerplexitySearchToolkit(api_key="test", base_url=custom_url)

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="perplexity_search"),
            tools=[toolkit],
        )
        await agent.ask("search")

        assert route.called

    @respx.mock
    async def test_custom_tool_name_in_agent(self) -> None:
        route = respx.post(f"{PERPLEXITY_BASE_URL}/search").mock(
            return_value=httpx.Response(200, json=_search_response())
        )
        toolkit = PerplexitySearchToolkit(api_key="test")
        search_tool = toolkit.search(name="web_search")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="web_search"),
            tools=[search_tool],
        )
        await agent.ask("search")

        assert route.called


@pytest.mark.asyncio
class TestAnswer:
    @respx.mock
    async def test_returns_structured_results(self) -> None:
        respx.post(f"{PERPLEXITY_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json=_chat_response(
                    content="AG2 is an open-source multi-agent framework.",
                    search_results=SAMPLE_SEARCH_RESULTS,
                    citations=["https://ag2.ai", "https://github.com/ag2ai/ag2"],
                ),
            )
        )
        toolkit = PerplexitySearchToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "AG2 framework"}, tool_name="perplexity_answer"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("answer")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            PerplexitySearchResponse(
                query="AG2 framework",
                results=[
                    PerplexitySearchResult(
                        title="AG2 Framework",
                        url="https://ag2.ai",
                        snippet="AG2 is an agent framework.",
                        date="2026-01-01",
                    ),
                    PerplexitySearchResult(
                        title="GitHub - AG2",
                        url="https://github.com/ag2ai/ag2",
                        snippet="Open source repo.",
                        date=None,
                    ),
                ],
                content="AG2 is an open-source multi-agent framework.",
                citations=["https://ag2.ai", "https://github.com/ag2ai/ag2"],
            )
        )

    @respx.mock
    async def test_empty_results(self) -> None:
        respx.post(f"{PERPLEXITY_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_response())
        )
        toolkit = PerplexitySearchToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "nothing"}, tool_name="perplexity_answer"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("answer")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            PerplexitySearchResponse(query="nothing", results=[], content="", citations=[])
        )

    @respx.mock
    async def test_defaults_applied_when_params_omitted(self) -> None:
        route = respx.post(f"{PERPLEXITY_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_response())
        )
        toolkit = PerplexitySearchToolkit(api_key="test")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="perplexity_answer"), tools=[toolkit])
        await agent.ask("answer")

        body = json.loads(route.calls.last.request.content)
        assert body == {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": "Be precise and concise."},
                {"role": "user", "content": "q"},
            ],
            "max_tokens": 1000,
            "web_search_options": {"search_context_size": "high"},
        }

    @respx.mock
    async def test_all_params_forwarded(self) -> None:
        route = respx.post(f"{PERPLEXITY_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_response())
        )
        toolkit = PerplexitySearchToolkit(api_key="test")
        answer_tool = toolkit.answer(
            model="sonar-pro",
            max_tokens=2000,
            search_domain_filter=["arxiv.org", "-medium.com"],
            search_context_size="medium",
            search_mode="academic",
            search_recency_filter="week",
            return_images=True,
            return_related_questions=True,
        )

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="perplexity_answer"),
            tools=[answer_tool],
        )
        await agent.ask("answer")

        body = json.loads(route.calls.last.request.content)
        assert body == {
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": "Be precise and concise."},
                {"role": "user", "content": "q"},
            ],
            "max_tokens": 2000,
            "web_search_options": {"search_context_size": "medium"},
            "search_domain_filter": ["arxiv.org", "-medium.com"],
            "search_mode": "academic",
            "search_recency_filter": "week",
            "return_images": True,
            "return_related_questions": True,
        }

    @respx.mock
    async def test_client_kwargs_forwarded_to_sdk(self) -> None:
        custom_url = "https://custom.perplexity.example"
        route = respx.post(f"{custom_url}/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_response())
        )
        toolkit = PerplexitySearchToolkit(api_key="test", base_url=custom_url)

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="perplexity_answer"),
            tools=[toolkit],
        )
        await agent.ask("answer")

        assert route.called

    @respx.mock
    async def test_custom_tool_name_in_agent(self) -> None:
        route = respx.post(f"{PERPLEXITY_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_response())
        )
        toolkit = PerplexitySearchToolkit(api_key="test")
        answer_tool = toolkit.answer(name="ask_perplexity")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="ask_perplexity"),
            tools=[answer_tool],
        )
        await agent.ask("answer")

        assert route.called

    @respx.mock
    async def test_returns_image_parts_when_api_yields_images(self) -> None:
        respx.post(f"{PERPLEXITY_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json=_chat_response(
                    content="See attached images.",
                    images=[
                        {
                            "image_url": "https://example.com/a.jpg",
                            "origin_url": "https://example.com/a",
                            "title": "Image A",
                            "width": 800,
                            "height": 600,
                        },
                        {
                            "image_url": "https://example.com/b.png",
                            "origin_url": "https://example.com/b",
                            "title": "Image B",
                            "width": 1024,
                            "height": 768,
                        },
                    ],
                ),
            )
        )
        toolkit = PerplexitySearchToolkit(api_key="test")
        answer_tool = toolkit.answer(return_images=True)

        config = TrackingConfig(_tool_call_config({"query": "show me images"}, tool_name="perplexity_answer"))
        agent = Agent("a", config=config, tools=[answer_tool])
        await agent.ask("answer")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        data_part, image_a, image_b = tool_results_event.results[0].result.parts

        assert data_part == DataInput(
            PerplexitySearchResponse(
                query="show me images",
                results=[],
                content="See attached images.",
                citations=[],
                images=[
                    PerplexityImageMeta(
                        image_url="https://example.com/a.jpg",
                        origin_url="https://example.com/a",
                        title="Image A",
                        width=800,
                        height=600,
                    ),
                    PerplexityImageMeta(
                        image_url="https://example.com/b.png",
                        origin_url="https://example.com/b",
                        title="Image B",
                        width=1024,
                        height=768,
                    ),
                ],
            )
        )
        assert image_a == ImageInput(url="https://example.com/a.jpg")
        assert image_b == ImageInput(url="https://example.com/b.png")

    @respx.mock
    async def test_skips_image_entries_without_image_url(self) -> None:
        respx.post(f"{PERPLEXITY_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json=_chat_response(
                    images=[
                        {"origin_url": None, "title": None, "width": None, "height": None},
                        {"image_url": "https://example.com/ok.jpg"},
                    ],
                ),
            )
        )
        toolkit = PerplexitySearchToolkit(api_key="test")
        answer_tool = toolkit.answer(return_images=True)

        config = TrackingConfig(_tool_call_config({"query": "q"}, tool_name="perplexity_answer"))
        agent = Agent("a", config=config, tools=[answer_tool])
        await agent.ask("answer")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        _data_part, image = tool_results_event.results[0].result.parts
        assert image == ImageInput(url="https://example.com/ok.jpg")


@pytest.mark.asyncio
class TestSearchVariable:
    @respx.mock
    async def test_resolved(self) -> None:
        route = respx.post(f"{PERPLEXITY_BASE_URL}/search").mock(
            return_value=httpx.Response(200, json=_search_response())
        )
        toolkit = PerplexitySearchToolkit(api_key="test")
        search_tool = toolkit.search(
            max_results=Variable("user_max"),
            search_recency_filter=Variable(),
        )

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "test query"}, tool_name="perplexity_search"),
            tools=[search_tool],
            variables={"user_max": 7, "search_recency_filter": "day"},
        )
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == {
            "query": "test query",
            "max_results": 7,
            "search_recency_filter": "day",
        }

    @respx.mock
    async def test_missing_raises(self) -> None:
        respx.post(f"{PERPLEXITY_BASE_URL}/search").mock(return_value=httpx.Response(200, json=_search_response()))
        toolkit = PerplexitySearchToolkit(api_key="test")
        search_tool = toolkit.search(max_results=Variable())

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "test query"}, tool_name="perplexity_search"),
            tools=[search_tool],
        )

        with pytest.raises(KeyError, match="max_results"):
            await agent.ask("search")


@pytest.mark.asyncio
class TestAnswerVariable:
    @respx.mock
    async def test_resolved(self) -> None:
        route = respx.post(f"{PERPLEXITY_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_response())
        )
        toolkit = PerplexitySearchToolkit(api_key="test")
        answer_tool = toolkit.answer(
            model=Variable("user_model"),
            search_recency_filter=Variable(),
        )

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "test query"}, tool_name="perplexity_answer"),
            tools=[answer_tool],
            variables={"user_model": "sonar-pro", "search_recency_filter": "day"},
        )
        await agent.ask("answer")

        body = json.loads(route.calls.last.request.content)
        assert body == {
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": "Be precise and concise."},
                {"role": "user", "content": "test query"},
            ],
            "max_tokens": 1000,
            "web_search_options": {"search_context_size": "high"},
            "search_recency_filter": "day",
        }

    @respx.mock
    async def test_missing_raises(self) -> None:
        respx.post(f"{PERPLEXITY_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_response())
        )
        toolkit = PerplexitySearchToolkit(api_key="test")
        answer_tool = toolkit.answer(search_mode=Variable())

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "test query"}, tool_name="perplexity_answer"),
            tools=[answer_tool],
        )

        with pytest.raises(KeyError, match="search_mode"):
            await agent.ask("answer")


@pytest.mark.asyncio
class TestIndividualTools:
    @respx.mock
    async def test_search_tool_passed_alone(self) -> None:
        route = respx.post(f"{PERPLEXITY_BASE_URL}/search").mock(
            return_value=httpx.Response(200, json=_search_response())
        )
        toolkit = PerplexitySearchToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="perplexity_search"),
            tools=[toolkit.search()],
        )
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == {"query": "q"}

    @respx.mock
    async def test_answer_tool_passed_alone(self) -> None:
        route = respx.post(f"{PERPLEXITY_BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_response())
        )
        toolkit = PerplexitySearchToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="perplexity_answer"),
            tools=[toolkit.answer()],
        )
        await agent.ask("answer")

        assert route.called

    @respx.mock
    async def test_whole_toolkit_registers_both_tools(self) -> None:
        route = respx.post(f"{PERPLEXITY_BASE_URL}/search").mock(
            return_value=httpx.Response(200, json=_search_response())
        )
        toolkit = PerplexitySearchToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="perplexity_search"),
            tools=[toolkit],
        )
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == {"query": "q"}
