# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest
import respx
from dirty_equals import IsPartialDict

pytest.importorskip("linkup")

from linkup import LinkupAuthenticationError

from ag2 import Agent, Context, DataInput, Variable
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from ag2.extensions.tools.search.linkup import (
    LinkupAnswerResult,
    LinkupAnswerSource,
    LinkupFetchResult,
    LinkupSearchResponse,
    LinkupSearchResult,
    LinkupToolkit,
)
from ag2.testing import TestConfig, TrackingConfig

LINKUP_BASE_URL = "https://api.linkup.so/v1"


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


def _linkup_result(
    *,
    name: str = "AG2 Framework",
    url: str = "https://ag2.ai",
    content: str = "AG2 is an open-source agent framework.",
) -> dict[str, Any]:
    return {"type": "text", "name": name, "url": url, "content": content, "favicon": ""}


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schemas(self, context: Context) -> None:
        toolkit = LinkupToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))

        names = [s.function.name for s in schemas]
        assert names == ["linkup_search", "linkup_fetch"]

    async def test_search_schema_has_query_param(self, context: Context) -> None:
        toolkit = LinkupToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))
        search_schema = next(s for s in schemas if s.function.name == "linkup_search")

        assert search_schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })

    async def test_fetch_schema_has_url_param(self, context: Context) -> None:
        toolkit = LinkupToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))
        fetch_schema = next(s for s in schemas if s.function.name == "linkup_fetch")

        assert fetch_schema.function.parameters == IsPartialDict({
            "required": ["url"],
            "properties": IsPartialDict({"url": IsPartialDict({"type": "string"})}),
        })

    async def test_custom_tool_name_and_description(self, context: Context) -> None:
        toolkit = LinkupToolkit(api_key="test")
        custom = toolkit.search(name="web_search", description="Custom web search.")

        [schema] = list(await custom.schemas(context))

        assert schema.function.name == "web_search"
        assert schema.function.description == "Custom web search."


@pytest.mark.asyncio
class TestSearchExecution:
    @respx.mock
    async def test_returns_structured_results(self) -> None:
        respx.post(f"{LINKUP_BASE_URL}/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        _linkup_result(name="AG2 Framework", url="https://ag2.ai", content="About AG2"),
                        _linkup_result(name="GitHub - AG2", url="https://github.com/ag2ai/ag2", content="Source"),
                        {"type": "image", "name": "AG2 logo", "url": "https://ag2.ai/logo.png"},
                    ],
                },
            )
        )
        toolkit = LinkupToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "AG2 framework"}, tool_name="linkup_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            LinkupSearchResponse(
                query="AG2 framework",
                results=[
                    LinkupSearchResult(name="AG2 Framework", url="https://ag2.ai", content="About AG2"),
                    LinkupSearchResult(name="GitHub - AG2", url="https://github.com/ag2ai/ag2", content="Source"),
                ],
            )
        )

    @respx.mock
    async def test_empty_results(self) -> None:
        respx.post(f"{LINKUP_BASE_URL}/search").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = LinkupToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "nothing"}, tool_name="linkup_search"))
        agent = Agent("a", config=config, tools=[toolkit])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(LinkupSearchResponse(query="nothing"))

    @respx.mock
    async def test_default_params(self) -> None:
        route = respx.post(f"{LINKUP_BASE_URL}/search").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = LinkupToolkit(api_key="test")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="linkup_search"), tools=[toolkit])
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({"q": "q", "depth": "standard", "outputType": "searchResults"})
        assert "maxResults" not in body
        assert "includeDomains" not in body
        assert "excludeDomains" not in body
        assert "fromDate" not in body
        assert "toDate" not in body

    @respx.mock
    async def test_all_params_forwarded(self) -> None:
        route = respx.post(f"{LINKUP_BASE_URL}/search").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = LinkupToolkit(api_key="test")
        search_tool = toolkit.search(
            depth="deep",
            max_results=7,
            include_domains=["arxiv.org"],
            exclude_domains=["medium.com"],
            from_date="2024-01-01",
            to_date="2024-12-31",
        )

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="linkup_search"), tools=[search_tool])
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({
            "q": "q",
            "depth": "deep",
            "outputType": "searchResults",
            "maxResults": 7,
            "includeDomains": ["arxiv.org"],
            "excludeDomains": ["medium.com"],
            "fromDate": "2024-01-01",
            "toDate": "2024-12-31",
        })

    @respx.mock
    async def test_toolkit_level_defaults_applied_to_default_search(self) -> None:
        route = respx.post(f"{LINKUP_BASE_URL}/search").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = LinkupToolkit(api_key="test", depth="fast", max_results=3)

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="linkup_search"), tools=[toolkit])
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({"q": "q", "depth": "fast", "maxResults": 3})

    @respx.mock
    async def test_sets_bearer_auth(self) -> None:
        route = respx.post(f"{LINKUP_BASE_URL}/search").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = LinkupToolkit(api_key="test-key")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="linkup_search"), tools=[toolkit])
        await agent.ask("search")

        assert route.calls.last.request.headers["authorization"] == "Bearer test-key"


@pytest.mark.asyncio
class TestSourcedAnswer:
    @respx.mock
    async def test_returns_answer_with_sources(self) -> None:
        route = respx.post(f"{LINKUP_BASE_URL}/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "answer": "AG2 is an open-source multi-agent framework.",
                    "sources": [
                        {"name": "AG2", "url": "https://ag2.ai", "snippet": "About AG2", "favicon": ""},
                        {"name": "GitHub", "url": "https://github.com/ag2ai/ag2", "snippet": "Source", "favicon": ""},
                    ],
                },
            )
        )
        toolkit = LinkupToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"query": "What is AG2?"}, tool_name="linkup_search"))
        agent = Agent("a", config=config, tools=[toolkit.search(output_type="sourcedAnswer")])
        await agent.ask("answer")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({"q": "What is AG2?", "outputType": "sourcedAnswer"})

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            LinkupAnswerResult(
                answer="AG2 is an open-source multi-agent framework.",
                sources=[
                    LinkupAnswerSource(name="AG2", url="https://ag2.ai", snippet="About AG2"),
                    LinkupAnswerSource(name="GitHub", url="https://github.com/ag2ai/ag2", snippet="Source"),
                ],
            )
        )


@pytest.mark.asyncio
class TestFetch:
    @respx.mock
    async def test_returns_markdown(self) -> None:
        route = respx.post(f"{LINKUP_BASE_URL}/fetch").mock(
            return_value=httpx.Response(200, json={"markdown": "# AG2\n\nfull text", "favicon": ""})
        )
        toolkit = LinkupToolkit(api_key="test")

        config = TrackingConfig(_tool_call_config({"url": "https://ag2.ai"}, tool_name="linkup_fetch"))
        agent = Agent("a", config=config, tools=[toolkit])
        await agent.ask("fetch")

        body = json.loads(route.calls.last.request.content)
        assert body == {"url": "https://ag2.ai"}

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            LinkupFetchResult(url="https://ag2.ai", markdown="# AG2\n\nfull text")
        )

    @respx.mock
    async def test_render_js_forwarded(self) -> None:
        route = respx.post(f"{LINKUP_BASE_URL}/fetch").mock(
            return_value=httpx.Response(200, json={"markdown": "", "favicon": ""})
        )
        toolkit = LinkupToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"url": "https://ag2.ai"}, tool_name="linkup_fetch"),
            tools=[toolkit.fetch(render_js=True)],
        )
        await agent.ask("fetch")

        body = json.loads(route.calls.last.request.content)
        assert body == {"url": "https://ag2.ai", "renderJs": True}


@pytest.mark.asyncio
class TestErrors:
    @respx.mock
    async def test_authentication_error_raised(self) -> None:
        respx.post(f"{LINKUP_BASE_URL}/search").mock(
            return_value=httpx.Response(
                401,
                json={
                    "statusCode": 401,
                    "error": {"code": "UNAUTHORIZED", "message": "Invalid API key", "details": []},
                },
            )
        )
        toolkit = LinkupToolkit(api_key="bad")

        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="linkup_search"), tools=[toolkit])

        with pytest.raises(LinkupAuthenticationError):
            await agent.ask("search")


@pytest.mark.asyncio
class TestLinkupToolkitVariable:
    @respx.mock
    async def test_resolved(self) -> None:
        route = respx.post(f"{LINKUP_BASE_URL}/search").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = LinkupToolkit(api_key="test")
        search_tool = toolkit.search(
            max_results=Variable("user_limit"),
            depth=Variable(),
        )

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="linkup_search"),
            tools=[search_tool],
            variables={"user_limit": 10, "depth": "deep"},
        )
        await agent.ask("search")

        body = json.loads(route.calls.last.request.content)
        assert body == IsPartialDict({"q": "q", "maxResults": 10, "depth": "deep"})

    @respx.mock
    async def test_missing_raises(self) -> None:
        respx.post(f"{LINKUP_BASE_URL}/search").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = LinkupToolkit(api_key="test")
        search_tool = toolkit.search(depth=Variable())

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="linkup_search"),
            tools=[search_tool],
        )

        with pytest.raises(KeyError, match="depth"):
            await agent.ask("search")


@pytest.mark.asyncio
class TestIndividualTools:
    @respx.mock
    async def test_search_tool_passed_alone(self) -> None:
        route = respx.post(f"{LINKUP_BASE_URL}/search").mock(
            return_value=httpx.Response(200, json={"results": [_linkup_result()]})
        )
        toolkit = LinkupToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="linkup_search"),
            tools=[toolkit.search()],
        )
        await agent.ask("search")

        assert route.called
