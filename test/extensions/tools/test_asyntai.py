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
from ag2.extensions.tools.asyntai import (
    AsyntaiAnswer,
    AsyntaiConversation,
    AsyntaiKnowledgeAdded,
    AsyntaiKnowledgeEntries,
    AsyntaiLeads,
    AsyntaiToolkit,
    AsyntaiWebsites,
)
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.final.function_tool import FunctionToolSchema

BASE_URL = "https://asyntai.com/api/v1"
API_KEY = "test-key"  # pragma: allowlist secret

ANSWER = {"success": True, "response": "Shipping costs 19 euros.", "session_id": "api_abc"}
WEBSITES = {"success": True, "websites": [{"id": 12, "domain": "example.com"}, "not-a-website"]}
ADDED_TEXT = {"success": True, "id": "6624c7b8-f270-4996-b5f1-7ddb114db0bf", "title": "Shipping", "chunks_created": 2}
ADDED_URL = {
    "success": True,
    "id": "72140d97-2e52-4e95-9044-374833e8a9c5",
    "title": "Home",
    "url": "https://example.com/",
    "chunks_created": 3,
}
ENTRIES = {"success": True, "entries": [{"id": "6624c7b8-f270-4996-b5f1-7ddb114db0bf", "title": "Shipping"}]}
LEADS = {"success": True, "leads": [{"email": "ann@example.com", "session_id": "s1"}]}
MESSAGES = {"success": True, "messages": [{"role": "user", "content": "Hi"}]}


@pytest.fixture(autouse=True)
def _no_ambient_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASYNTAI_API_KEY", raising=False)


def _config(tool_name: str, arguments: dict[str, object]) -> TestConfig:
    return TestConfig(
        ModelResponse(
            tool_calls=ToolCallsEvent([ToolCallEvent(arguments=json.dumps(arguments), name=tool_name)]),
        ),
        "done",
    )


async def _run_event(
    tools: object, tool_name: str, arguments: dict[str, object], variables: dict[str, object] | None = None
) -> ToolResultsEvent:
    """Run one tool call through a real agent and return the tool results event."""
    config = TrackingConfig(_config(tool_name, arguments))
    agent = Agent("a", config=config, tools=[tools])
    await agent.ask("go", variables=variables or {})
    return config.mock.call_args_list[1].args[0]


async def _run_tool(
    tools: object, tool_name: str, arguments: dict[str, object], variables: dict[str, object] | None = None
) -> object:
    """Run one tool call through a real agent and return the tool's result part."""
    event = await _run_event(tools, tool_name, arguments, variables)
    return event.results[0].result.parts[0]


class TestConstructor:
    def test_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="ASYNTAI_API_KEY"):
            AsyntaiToolkit()

    def test_reads_key_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASYNTAI_API_KEY", "from-env")
        assert AsyntaiToolkit()._api_key == "from-env"

    def test_argument_beats_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASYNTAI_API_KEY", "from-env")
        assert AsyntaiToolkit(api_key="explicit")._api_key == "explicit"

    def test_strips_trailing_slash(self) -> None:
        assert AsyntaiToolkit(API_KEY, base_url="https://x.test/api/v1/")._base_url == "https://x.test/api/v1"


@pytest.mark.asyncio
class TestSchema:
    async def test_default_tools_are_read_only(self, context: Context) -> None:
        schemas = list(await AsyntaiToolkit(API_KEY).schemas(context))

        assert [s.function.name for s in schemas] == ["asyntai_ask_website", "asyntai_list_websites"]

    async def test_all_tools(self, context: Context) -> None:
        schemas = list(await AsyntaiToolkit(API_KEY, all_tools=True).schemas(context))

        assert [s.function.name for s in schemas] == [
            "asyntai_ask_website",
            "asyntai_list_websites",
            "asyntai_add_text",
            "asyntai_add_url",
            "asyntai_list_knowledge",
            "asyntai_get_leads",
            "asyntai_get_conversation",
        ]

    async def test_ask_schema_hides_context(self, context: Context) -> None:
        [schema] = list(await AsyntaiToolkit(API_KEY).ask().schemas(context))

        assert isinstance(schema, FunctionToolSchema)
        assert schema.function.parameters == IsPartialDict({
            "required": ["question"],
            "properties": IsPartialDict({"question": IsPartialDict({"type": "string"})}),
        })
        assert "ctx" not in schema.function.parameters["properties"]

    async def test_custom_name_and_description(self, context: Context) -> None:
        custom = AsyntaiToolkit(API_KEY).ask(name="ask_docs", description="Ask our docs.")

        [schema] = list(await custom.schemas(context))

        assert schema.function.name == "ask_docs"
        assert schema.function.description == "Ask our docs."


@pytest.mark.asyncio
class TestAsk:
    @respx.mock
    async def test_returns_answer(self) -> None:
        route = respx.post(f"{BASE_URL}/chat/").mock(return_value=httpx.Response(200, json=ANSWER))

        part = await _run_tool(AsyntaiToolkit(API_KEY), "asyntai_ask_website", {"question": "Shipping?"})

        assert part == DataInput(AsyntaiAnswer(answer="Shipping costs 19 euros.", session_id="api_abc"))
        assert json.loads(route.calls.last.request.content) == {"message": "Shipping?"}

    @respx.mock
    async def test_sends_bearer_key(self) -> None:
        route = respx.post(f"{BASE_URL}/chat/").mock(return_value=httpx.Response(200, json=ANSWER))

        await _run_tool(AsyntaiToolkit(API_KEY), "asyntai_ask_website", {"question": "Q"})

        assert route.calls.last.request.headers["Authorization"] == f"Bearer {API_KEY}"

    @respx.mock
    async def test_sends_session_id(self) -> None:
        route = respx.post(f"{BASE_URL}/chat/").mock(return_value=httpx.Response(200, json=ANSWER))

        await _run_tool(AsyntaiToolkit(API_KEY), "asyntai_ask_website", {"question": "Q", "session_id": "api_abc"})

        assert json.loads(route.calls.last.request.content)["session_id"] == "api_abc"

    @respx.mock
    async def test_toolkit_website_is_the_default(self) -> None:
        route = respx.post(f"{BASE_URL}/chat/").mock(return_value=httpx.Response(200, json=ANSWER))

        await _run_tool(AsyntaiToolkit(API_KEY, website_id=12), "asyntai_ask_website", {"question": "Q"})

        assert json.loads(route.calls.last.request.content)["website_id"] == 12

    @respx.mock
    async def test_model_website_beats_the_default(self) -> None:
        route = respx.post(f"{BASE_URL}/chat/").mock(return_value=httpx.Response(200, json=ANSWER))

        await _run_tool(AsyntaiToolkit(API_KEY, website_id=12), "asyntai_ask_website", {"question": "Q", "website": 34})

        assert json.loads(route.calls.last.request.content)["website_id"] == 34

    @respx.mock
    async def test_website_from_variable(self) -> None:
        route = respx.post(f"{BASE_URL}/chat/").mock(return_value=httpx.Response(200, json=ANSWER))
        toolkit = AsyntaiToolkit(API_KEY, website_id=Variable("site"))

        await _run_tool(toolkit, "asyntai_ask_website", {"question": "Q"}, variables={"site": 56})

        assert json.loads(route.calls.last.request.content)["website_id"] == 56


@pytest.mark.asyncio
class TestOtherTools:
    @respx.mock
    async def test_list_websites_skips_bad_items(self) -> None:
        respx.get(f"{BASE_URL}/websites/").mock(return_value=httpx.Response(200, json=WEBSITES))

        part = await _run_tool(AsyntaiToolkit(API_KEY), "asyntai_list_websites", {})

        assert part == DataInput(AsyntaiWebsites(websites=[{"id": 12, "domain": "example.com"}]))

    @respx.mock
    async def test_add_text(self) -> None:
        route = respx.post(f"{BASE_URL}/knowledge/text/").mock(return_value=httpx.Response(200, json=ADDED_TEXT))

        part = await _run_tool(
            AsyntaiToolkit(API_KEY, all_tools=True, website_id=12),
            "asyntai_add_text",
            {"title": "Shipping", "content": "Shipping costs 19 euros."},
        )

        assert part == DataInput(
            AsyntaiKnowledgeAdded(entry_id="6624c7b8-f270-4996-b5f1-7ddb114db0bf", title="Shipping", chunks_created=2)
        )
        assert json.loads(route.calls.last.request.content) == {
            "title": "Shipping",
            "content": "Shipping costs 19 euros.",
            "website_id": 12,
        }

    @respx.mock
    async def test_add_url(self) -> None:
        route = respx.post(f"{BASE_URL}/knowledge/url/").mock(return_value=httpx.Response(200, json=ADDED_URL))

        part = await _run_tool(AsyntaiToolkit(API_KEY).add_url(), "asyntai_add_url", {"url": "https://example.com/"})

        assert part == DataInput(
            AsyntaiKnowledgeAdded(
                entry_id="72140d97-2e52-4e95-9044-374833e8a9c5",
                title="Home",
                url="https://example.com/",
                chunks_created=3,
            )
        )
        assert json.loads(route.calls.last.request.content) == {"url": "https://example.com/"}

    @respx.mock
    async def test_list_knowledge(self) -> None:
        route = respx.get(f"{BASE_URL}/knowledge/").mock(return_value=httpx.Response(200, json=ENTRIES))

        part = await _run_tool(AsyntaiToolkit(API_KEY).list_knowledge(), "asyntai_list_knowledge", {"limit": 5})

        assert part == DataInput(
            AsyntaiKnowledgeEntries(entries=[{"id": "6624c7b8-f270-4996-b5f1-7ddb114db0bf", "title": "Shipping"}])
        )
        assert route.calls.last.request.url.params["limit"] == "5"

    @respx.mock
    async def test_get_leads_drops_empty_filters(self) -> None:
        route = respx.get(f"{BASE_URL}/leads/").mock(return_value=httpx.Response(200, json=LEADS))

        part = await _run_tool(AsyntaiToolkit(API_KEY).get_leads(), "asyntai_get_leads", {"since": "2026-01-31"})

        assert part == DataInput(AsyntaiLeads(leads=[{"email": "ann@example.com", "session_id": "s1"}]))
        params = route.calls.last.request.url.params
        assert params["since"] == "2026-01-31"
        assert "email" not in params
        assert "website_id" not in params

    @respx.mock
    async def test_get_conversation(self) -> None:
        route = respx.get(f"{BASE_URL}/conversations/").mock(return_value=httpx.Response(200, json=MESSAGES))

        part = await _run_tool(
            AsyntaiToolkit(API_KEY).get_conversation(), "asyntai_get_conversation", {"session_id": "s1"}
        )

        assert part == DataInput(AsyntaiConversation(session_id="s1", messages=[{"role": "user", "content": "Hi"}]))
        assert route.calls.last.request.url.params["session_id"] == "s1"


@pytest.mark.asyncio
class TestErrors:
    @pytest.mark.parametrize(
        ("status", "error"),
        [(401, "Invalid API key"), (403, "API access requires a paid plan"), (429, "Message limit reached")],
    )
    @respx.mock
    async def test_error_carries_asyntai_message(self, status: int, error: str) -> None:
        respx.post(f"{BASE_URL}/chat/").mock(
            return_value=httpx.Response(status, json={"success": False, "error": error})
        )
        config = _config("asyntai_ask_website", {"question": "Q"})
        agent = Agent("a", config=config, tools=[AsyntaiToolkit(API_KEY)])

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await agent.ask("go")

        assert str(exc_info.value) == f"Asyntai API returned HTTP {status}: {error}"
        assert API_KEY not in str(exc_info.value.request.url)

    @respx.mock
    async def test_non_json_error_reports_the_status(self) -> None:
        respx.get(f"{BASE_URL}/websites/").mock(return_value=httpx.Response(502, text="<html>Bad gateway</html>"))
        config = _config("asyntai_list_websites", {})
        agent = Agent("a", config=config, tools=[AsyntaiToolkit(API_KEY)])

        with pytest.raises(httpx.HTTPStatusError, match=r"HTTP 502$"):
            await agent.ask("go")

    @respx.mock
    async def test_non_object_success_body_gives_empty_result(self) -> None:
        respx.get(f"{BASE_URL}/websites/").mock(return_value=httpx.Response(200, json=["unexpected"]))

        part = await _run_tool(AsyntaiToolkit(API_KEY), "asyntai_list_websites", {})

        assert part == DataInput(AsyntaiWebsites(websites=[]))
