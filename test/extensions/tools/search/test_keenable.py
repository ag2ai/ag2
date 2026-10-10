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
from ag2.extensions.tools.search.keenable import (
    KeenableFetchResult,
    KeenableSearchResponse,
    KeenableSearchResult,
    KeenableSearchToolkit,
)
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.final.function_tool import FunctionToolSchema

KEENABLE_BASE_URL = "https://api.keenable.ai"

# Shape of a real /v1/search result: the page text is in `snippet`, `description` is empty.
SEARCH_PAYLOAD = {
    "query": "AG2 agent framework",
    "results": [
        {
            "title": "GitHub - ag2ai/ag2: AG2 (formerly AutoGen): The Open-Source AgentOS",
            "url": "https://github.com/ag2ai/ag2",
            "description": "",
            "snippet": "AG2 is an open-source programming framework for building AI agents.",
            "acquired_at": "2026-09-13T02:40:48Z",
        },
        {
            "title": "AG2 documentation",
            "url": "https://docs.ag2.ai/latest/",
            "description": "",
            "snippet": "Build production-ready AI agents in minutes, not months.",
            "acquired_at": "2026-09-20T11:05:12Z",
        },
        "not-a-result",
    ],
}

# Shape of a real /v1/fetch response.
FETCH_PAYLOAD = {
    "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "title": "Python (programming language)",
    "content": "Python is a high-level, general-purpose programming language.",
    "description": "",
    "author": "Contributors to Wikimedia projects",
    "published_at": 1790359578,
}


@pytest.fixture(autouse=True)
def _no_ambient_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)


def _tool_call_config(
    arguments: dict[str, object],
    *,
    tool_name: str = "keenable_search",
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


def _fetch_config(url: str) -> TestConfig:
    return _tool_call_config({"url": url}, tool_name="keenable_fetch")


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schemas(self, context: Context) -> None:
        toolkit = KeenableSearchToolkit()

        schemas = list(await toolkit.schemas(context))

        search_schema, fetch_schema = schemas
        assert isinstance(search_schema, FunctionToolSchema)
        assert isinstance(fetch_schema, FunctionToolSchema)
        assert search_schema.function.name == "keenable_search"
        assert search_schema.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })
        assert fetch_schema.function.name == "keenable_fetch"
        assert fetch_schema.function.parameters == IsPartialDict({
            "required": ["url"],
            "properties": IsPartialDict({"url": IsPartialDict({"type": "string"})}),
        })

    async def test_custom_name_and_description(self, context: Context) -> None:
        toolkit = KeenableSearchToolkit()
        custom = toolkit.fetch(name="read_page", description="Read a web page.")

        [schema] = list(await custom.schemas(context))

        assert schema.function.name == "read_page"
        assert schema.function.description == "Read a web page."


@pytest.mark.asyncio
class TestSearch:
    @respx.mock
    async def test_returns_snippet_text(self) -> None:
        respx.post(f"{KEENABLE_BASE_URL}/v1/search/public").mock(return_value=httpx.Response(200, json=SEARCH_PAYLOAD))
        config = TrackingConfig(_tool_call_config({"query": "AG2 agent framework"}))
        agent = Agent("a", config=config, tools=[KeenableSearchToolkit()])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            KeenableSearchResponse(
                query="AG2 agent framework",
                results=[
                    KeenableSearchResult(
                        title="GitHub - ag2ai/ag2: AG2 (formerly AutoGen): The Open-Source AgentOS",
                        url="https://github.com/ag2ai/ag2",
                        snippet="AG2 is an open-source programming framework for building AI agents.",
                        acquired_at="2026-09-13T02:40:48Z",
                    ),
                    KeenableSearchResult(
                        title="AG2 documentation",
                        url="https://docs.ag2.ai/latest/",
                        snippet="Build production-ready AI agents in minutes, not months.",
                        acquired_at="2026-09-20T11:05:12Z",
                    ),
                ],
            )
        )

    @respx.mock
    async def test_keyless_by_default(self) -> None:
        route = respx.post(f"{KEENABLE_BASE_URL}/v1/search/public").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[KeenableSearchToolkit()])

        await agent.ask("search")

        request = route.calls.last.request
        assert json.loads(request.content) == {"query": "AG2", "snippet_max_length": 1000}
        assert request.headers["X-Keenable-Title"] == "ag2"
        assert "X-API-Key" not in request.headers

    @respx.mock
    async def test_api_key_uses_keyed_endpoint(self) -> None:
        route = respx.post(f"{KEENABLE_BASE_URL}/v1/search").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "AG2"}),
            tools=[KeenableSearchToolkit(api_key="test-key")],
        )

        await agent.ask("search")

        request = route.calls.last.request
        assert request.headers["X-API-Key"] == "test-key"
        assert request.headers["X-Keenable-Title"] == "ag2"

    @respx.mock
    async def test_api_key_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KEENABLE_API_KEY", "env-key")
        route = respx.post(f"{KEENABLE_BASE_URL}/v1/search").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[KeenableSearchToolkit()])

        await agent.ask("search")

        assert route.calls.last.request.headers["X-API-Key"] == "env-key"

    @respx.mock
    async def test_forwards_search_defaults(self) -> None:
        route = respx.post(f"{KEENABLE_BASE_URL}/v1/search/public").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        toolkit = KeenableSearchToolkit(
            max_results=5,
            site="docs.python.org",
            published_after="2026-01-01",
            snippet_max_length=500,
        )
        agent = Agent("a", config=_tool_call_config({"query": "asyncio"}), tools=[toolkit])

        await agent.ask("search")

        assert json.loads(route.calls.last.request.content) == {
            "query": "asyncio",
            "max_results": 5,
            "site": "docs.python.org",
            "published_after": "2026-01-01",
            "snippet_max_length": 500,
        }

    @respx.mock
    async def test_resolves_variables(self) -> None:
        route = respx.post(f"{KEENABLE_BASE_URL}/v1/search/public").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        toolkit = KeenableSearchToolkit()
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "AG2"}),
            tools=[toolkit.search(site=Variable(), max_results=Variable("limit"))],
            variables={"site": "github.com", "limit": 3},
        )

        await agent.ask("search")

        assert json.loads(route.calls.last.request.content) == {
            "query": "AG2",
            "site": "github.com",
            "max_results": 3,
            "snippet_max_length": 1000,
        }

    async def test_missing_variable_raises(self) -> None:
        toolkit = KeenableSearchToolkit()
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "AG2"}),
            tools=[toolkit.search(max_results=Variable("limit"))],
        )

        with pytest.raises(KeyError):
            await agent.ask("search")

    @respx.mock
    async def test_non_object_payload_yields_no_results(self) -> None:
        respx.post(f"{KEENABLE_BASE_URL}/v1/search/public").mock(return_value=httpx.Response(200, json=["unexpected"]))
        config = TrackingConfig(_tool_call_config({"query": "AG2"}))
        agent = Agent("a", config=config, tools=[KeenableSearchToolkit()])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(KeenableSearchResponse(query="AG2"))


@pytest.mark.asyncio
class TestFetch:
    @respx.mock
    async def test_returns_page_content(self) -> None:
        route = respx.get(f"{KEENABLE_BASE_URL}/v1/fetch/public").mock(
            return_value=httpx.Response(200, json=FETCH_PAYLOAD)
        )
        page_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        config = TrackingConfig(_fetch_config(page_url))
        agent = Agent("a", config=config, tools=[KeenableSearchToolkit()])

        await agent.ask("fetch")

        request = route.calls.last.request
        assert dict(request.url.params) == {"url": page_url}
        assert request.headers["X-Keenable-Title"] == "ag2"
        assert "X-API-Key" not in request.headers
        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            KeenableFetchResult(
                url=page_url,
                title="Python (programming language)",
                content="Python is a high-level, general-purpose programming language.",
                author="Contributors to Wikimedia projects",
                published_at="2026-09-25T18:06:18+00:00",
            )
        )

    @respx.mock
    async def test_api_key_uses_keyed_endpoint(self) -> None:
        route = respx.get(f"{KEENABLE_BASE_URL}/v1/fetch").mock(return_value=httpx.Response(200, json=FETCH_PAYLOAD))
        agent = Agent(
            "a",
            config=_fetch_config("https://example.com"),
            tools=[KeenableSearchToolkit(api_key="test-key")],
        )

        await agent.ask("fetch")

        assert route.calls.last.request.headers["X-API-Key"] == "test-key"

    @respx.mock
    async def test_missing_fields_fall_back_to_defaults(self) -> None:
        respx.get(f"{KEENABLE_BASE_URL}/v1/fetch/public").mock(
            return_value=httpx.Response(200, json={"title": "Example", "content": "# Example", "description": ""})
        )
        config = TrackingConfig(_fetch_config("https://example.com"))
        agent = Agent("a", config=config, tools=[KeenableSearchToolkit().fetch()])

        await agent.ask("fetch")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            KeenableFetchResult(url="https://example.com", title="Example", content="# Example")
        )


@pytest.mark.asyncio
class TestErrors:
    @respx.mock
    async def test_error_carries_api_message(self) -> None:
        respx.get(f"{KEENABLE_BASE_URL}/v1/fetch/public").mock(
            return_value=httpx.Response(
                404,
                json={"error": "Not found", "message": "The page does not exist at this URL."},
            )
        )
        agent = Agent("a", config=_fetch_config("https://example.com/missing"), tools=[KeenableSearchToolkit()])

        with pytest.raises(httpx.HTTPStatusError, match="HTTP 404: The page does not exist at this URL"):
            await agent.ask("fetch")

    @respx.mock
    async def test_keyless_rate_limit_reports_retry_after(self) -> None:
        respx.post(f"{KEENABLE_BASE_URL}/v1/search/public").mock(
            return_value=httpx.Response(
                429,
                headers={"Retry-After": "30"},
                json={"error": "Too many requests", "message": "Rate limit exceeded."},
            )
        )
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[KeenableSearchToolkit()])

        with pytest.raises(httpx.HTTPStatusError, match=r"Retry-After: 30.*KEENABLE_API_KEY"):
            await agent.ask("search")

    @respx.mock
    async def test_error_does_not_echo_api_key(self) -> None:
        respx.post(f"{KEENABLE_BASE_URL}/v1/search").mock(return_value=httpx.Response(401, text="Unauthorized"))
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "AG2"}),
            tools=[KeenableSearchToolkit(api_key="secret-key")],
        )

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await agent.ask("search")

        assert str(exc_info.value) == "Keenable API returned HTTP 401"
        assert "secret-key" not in str(exc_info.value.request.url)


@pytest.mark.asyncio
class TestConnection:
    @respx.mock
    async def test_custom_base_url_strips_trailing_slashes(self) -> None:
        route = respx.post("https://proxy.example.com/keenable/v1/search/public").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        toolkit = KeenableSearchToolkit(base_url="https://proxy.example.com/keenable//")
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        await agent.ask("search")

        assert route.called

    @respx.mock
    async def test_accepts_proxy_and_tls_verify_settings(self) -> None:
        route = respx.post(f"{KEENABLE_BASE_URL}/v1/search/public").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        toolkit = KeenableSearchToolkit(proxy="http://proxy.example.com:3128", verify=False)
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[toolkit])

        await agent.ask("search")

        assert route.called
