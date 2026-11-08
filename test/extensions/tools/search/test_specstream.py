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
from ag2.extensions.tools.search.specstream import (
    SpecStreamExtractResult,
    SpecStreamSearchResponse,
    SpecStreamSearchToolkit,
    SpecStreamSource,
)
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.final.function_tool import FunctionToolSchema

BASE_URL = "https://api.specstream.ai"

SEARCH_PAYLOAD = {
    "kind": "fact",
    "answer": {"value": "Guido van Rossum", "page": "https://en.wikipedia.org/wiki/Python_(programming_language)"},
    "sources": [
        {
            "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "title": "Python (programming language)",
            "site": "en.wikipedia.org",
            "snippet": "Python was created by Guido van Rossum.",
            "published": "2026-01-02",
            "cited": True,
        },
        "not-a-source",
    ],
    "passages": [{"url": "https://python.org", "title": "Python", "text": "Python is a language."}],
}

EXTRACT_PAYLOAD = {
    "results": [
        {
            "url": "https://example.com/post",
            "status": "ok",
            "title": "A post",
            "text": "# A post\n\nBody.",
            "published": "2026-02-03",
        }
    ]
}


@pytest.fixture(autouse=True)
def _no_ambient_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPECSTREAM_API_KEY", raising=False)


def _tool_call_config(arguments: dict[str, object], *, tool_name: str = "specstream_search") -> TestConfig:
    return TestConfig(
        ModelResponse(
            tool_calls=ToolCallsEvent([ToolCallEvent(arguments=json.dumps(arguments), name=tool_name)]),
        ),
        "done",
    )


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schemas(self, context: Context) -> None:
        schemas = list(await SpecStreamSearchToolkit(api_key="k").schemas(context))

        search_schema, extract_schema = schemas
        assert isinstance(search_schema, FunctionToolSchema)
        assert isinstance(extract_schema, FunctionToolSchema)
        assert search_schema.function.name == "specstream_search"
        assert search_schema.function.parameters == IsPartialDict({"required": ["query"]})
        assert extract_schema.function.name == "specstream_extract"
        assert extract_schema.function.parameters == IsPartialDict({"required": ["url"]})


@pytest.mark.asyncio
class TestSearch:
    @respx.mock
    async def test_returns_answer_and_sources(self) -> None:
        route = respx.get(f"{BASE_URL}/v1/web-search").mock(return_value=httpx.Response(200, json=SEARCH_PAYLOAD))
        config = TrackingConfig(_tool_call_config({"query": "Who created Python?"}))
        agent = Agent("a", config=config, tools=[SpecStreamSearchToolkit(api_key="test-key")])

        await agent.ask("search")

        request = route.calls.last.request
        assert request.headers["Authorization"] == "Bearer test-key"
        assert dict(request.url.params) == {"q": "Who created Python?"}
        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SpecStreamSearchResponse(
                query="Who created Python?",
                kind="fact",
                answer="Guido van Rossum",
                answer_page="https://en.wikipedia.org/wiki/Python_(programming_language)",
                sources=[
                    SpecStreamSource(
                        url="https://en.wikipedia.org/wiki/Python_(programming_language)",
                        title="Python (programming language)",
                        site="en.wikipedia.org",
                        snippet="Python was created by Guido van Rossum.",
                        published="2026-01-02",
                    )
                ],
                passages=[{"url": "https://python.org", "title": "Python", "text": "Python is a language."}],
            )
        )

    @respx.mock
    async def test_api_key_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SPECSTREAM_API_KEY", "env-key")
        route = respx.get(f"{BASE_URL}/v1/web-search").mock(return_value=httpx.Response(200, json={}))
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[SpecStreamSearchToolkit()])

        await agent.ask("search")

        assert route.calls.last.request.headers["Authorization"] == "Bearer env-key"

    @respx.mock
    async def test_passages_default_and_variable(self) -> None:
        route = respx.get(f"{BASE_URL}/v1/web-search").mock(return_value=httpx.Response(200, json={}))
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "AG2"}),
            tools=[SpecStreamSearchToolkit(api_key="k").search(passages=Variable("want"))],
            variables={"want": True},
        )

        await agent.ask("search")

        assert dict(route.calls.last.request.url.params) == {"q": "AG2", "passages": "true"}

    async def test_missing_key_raises(self) -> None:
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[SpecStreamSearchToolkit()])

        with pytest.raises(ValueError, match="SPECSTREAM_API_KEY"):
            await agent.ask("search")

    @respx.mock
    async def test_non_object_payload_yields_empty_response(self) -> None:
        respx.get(f"{BASE_URL}/v1/web-search").mock(return_value=httpx.Response(200, json=["unexpected"]))
        config = TrackingConfig(_tool_call_config({"query": "AG2"}))
        agent = Agent("a", config=config, tools=[SpecStreamSearchToolkit(api_key="k")])

        await agent.ask("search")

        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(SpecStreamSearchResponse(query="AG2"))

    @respx.mock
    async def test_http_error_carries_api_message(self) -> None:
        respx.get(f"{BASE_URL}/v1/web-search").mock(
            return_value=httpx.Response(402, json={"message": "Balance is empty."})
        )
        agent = Agent("a", config=_tool_call_config({"query": "AG2"}), tools=[SpecStreamSearchToolkit(api_key="k")])

        with pytest.raises(httpx.HTTPStatusError, match="HTTP 402: Balance is empty."):
            await agent.ask("search")


@pytest.mark.asyncio
class TestExtract:
    @respx.mock
    async def test_returns_page_text(self) -> None:
        route = respx.post(f"{BASE_URL}/v1/extract").mock(return_value=httpx.Response(200, json=EXTRACT_PAYLOAD))
        config = TrackingConfig(_tool_call_config({"url": "https://example.com/post"}, tool_name="specstream_extract"))
        agent = Agent("a", config=config, tools=[SpecStreamSearchToolkit(api_key="k")])

        await agent.ask("read")

        assert json.loads(route.calls.last.request.content) == {"urls": ["https://example.com/post"]}
        tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert tool_results_event.results[0].result.parts[0] == DataInput(
            SpecStreamExtractResult(
                url="https://example.com/post",
                status="ok",
                title="A post",
                text="# A post\n\nBody.",
                published="2026-02-03",
            )
        )

    @respx.mock
    async def test_format_default(self) -> None:
        route = respx.post(f"{BASE_URL}/v1/extract").mock(return_value=httpx.Response(200, json={"results": []}))
        toolkit = SpecStreamSearchToolkit(api_key="k", extract_format="text")
        agent = Agent(
            "a",
            config=_tool_call_config({"url": "https://example.com"}, tool_name="specstream_extract"),
            tools=[toolkit],
        )

        await agent.ask("read")

        assert json.loads(route.calls.last.request.content) == {"urls": ["https://example.com"], "format": "text"}
