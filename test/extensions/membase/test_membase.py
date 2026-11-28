# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from dirty_equals import IsPartialDict

pytest.importorskip("membase")

from membase.errors import PermissionDeniedError

from ag2 import Agent, Context, DataInput, Variable
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from ag2.extensions.membase import (
    MembasePassage,
    MembaseProfile,
    MembaseSaveResult,
    MembaseSearchResponse,
    MembaseToolkit,
)
from ag2.testing import TestConfig, TrackingConfig

MEMBASE_BASE_URL = "https://api.app.membase.com"

SEARCH_PAYLOAD = {
    "items": [
        {
            "type": "memory",
            "id": "mem_eA",
            "content": "We chose Postgres for the ledger.",
            "memory_name": "Project decisions",
            "score": 0.9,
        },
        {"type": "memory", "content": "   ", "memory_name": "Project decisions"},
    ],
    "partial": [
        {"memory_name": "Inbox", "reason": "error"},
        {"memory_name": "Notes", "reason": "truncated"},
    ],
}


@pytest.fixture
def context() -> Context:
    return Context(stream=MagicMock())


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMBASE_API_KEY", raising=False)
    monkeypatch.delenv("MEMBASE_BASE_URL", raising=False)


def _tool_call_config(arguments: dict[str, object], *, tool_name: str, final_reply: str = "done") -> TestConfig:
    return TestConfig(
        ModelResponse(tool_calls=ToolCallsEvent([ToolCallEvent(arguments=json.dumps(arguments), name=tool_name)])),
        final_reply,
    )


def _first_result(config: TrackingConfig) -> object:
    tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
    return tool_results_event.results[0]


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schemas(self, context: Context) -> None:
        toolkit = MembaseToolkit(api_key="mbk_test")

        schemas = list(await toolkit.schemas(context))

        assert [s.function.name for s in schemas] == ["membase_search", "membase_save", "membase_profile"]
        search = schemas[0]
        assert search.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({"query": IsPartialDict({"type": "string"})}),
        })

    async def test_custom_name_and_description(self, context: Context) -> None:
        tool = MembaseToolkit(api_key="mbk_test").search(name="recall", description="Recall the user's memory.")

        [schema] = list(await tool.schemas(context))

        assert schema.function.name == "recall"
        assert schema.function.description == "Recall the user's memory."


@pytest.mark.asyncio
class TestSearch:
    @respx.mock
    async def test_returns_passages_citing_their_memory(self) -> None:
        route = respx.post(f"{MEMBASE_BASE_URL}/v1/search").mock(return_value=httpx.Response(200, json=SEARCH_PAYLOAD))
        config = TrackingConfig(_tool_call_config({"query": "ledger database"}, tool_name="membase_search"))
        agent = Agent("a", config=config, tools=[MembaseToolkit(api_key="mbk_test", limit=5)])

        await agent.ask("Which database did we pick?")

        assert _first_result(config).result.parts[0] == DataInput(
            MembaseSearchResponse(
                query="ledger database",
                passages=[
                    MembasePassage(
                        memory_name="Project decisions",
                        content="We chose Postgres for the ledger.",
                        id="mem_eA",
                        score=0.9,
                    )
                ],
                unavailable=["Inbox"],
            )
        )
        request = route.calls.last.request
        assert request.headers["authorization"] == "Bearer mbk_test"
        assert json.loads(request.content) == {"q": "ledger database", "limit": 5, "mode": "full"}

    @respx.mock
    async def test_api_key_and_limit_resolve_from_variables(self) -> None:
        route = respx.post(f"{MEMBASE_BASE_URL}/v1/search").mock(return_value=httpx.Response(200, json=SEARCH_PAYLOAD))
        toolkit = MembaseToolkit(api_key=Variable("membase_api_key"))
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "ledger"}, tool_name="membase_search"),
            tools=[toolkit.search(limit=Variable())],
            variables={"membase_api_key": "mbk_alice", "limit": 3},
        )

        await agent.ask("search")

        request = route.calls.last.request
        assert request.headers["authorization"] == "Bearer mbk_alice"
        assert json.loads(request.content) == IsPartialDict({"limit": 3})

    @respx.mock
    async def test_api_key_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMBASE_API_KEY", "mbk_env")
        route = respx.post(f"{MEMBASE_BASE_URL}/v1/search").mock(return_value=httpx.Response(200, json=SEARCH_PAYLOAD))
        agent = Agent(
            "a", config=_tool_call_config({"query": "q"}, tool_name="membase_search"), tools=[MembaseToolkit()]
        )

        await agent.ask("search")

        assert route.calls.last.request.headers["authorization"] == "Bearer mbk_env"

    @respx.mock
    async def test_base_url(self) -> None:
        route = respx.post("http://127.0.0.1:8787/v1/search").mock(
            return_value=httpx.Response(200, json=SEARCH_PAYLOAD)
        )
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="membase_search"),
            tools=[MembaseToolkit(api_key="local", base_url="http://127.0.0.1:8787")],
        )

        await agent.ask("search")

        assert route.called

    @respx.mock
    async def test_refused_key_raises_the_sdk_error(self) -> None:
        respx.post(f"{MEMBASE_BASE_URL}/v1/search").mock(
            return_value=httpx.Response(403, json={"error": {"code": "unauthorized", "message": "unauthorized"}})
        )
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="membase_search"),
            tools=[MembaseToolkit(api_key="mbk_revoked")],
        )

        with pytest.raises(PermissionDeniedError, match="403"):
            await agent.ask("search")

    async def test_missing_variable_raises(self) -> None:
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="membase_search"),
            tools=[MembaseToolkit(api_key=Variable("membase_api_key"))],
        )

        with pytest.raises(KeyError):
            await agent.ask("search")


@pytest.mark.asyncio
class TestSaveAndProfile:
    @respx.mock
    async def test_save_reports_where_the_note_landed(self) -> None:
        route = respx.post(f"{MEMBASE_BASE_URL}/v1/memories/add").mock(
            return_value=httpx.Response(
                202,
                json={
                    "id": "doc_bm90ZQ",
                    "kind": "note",
                    "status": "processing",
                    "learning": [{"memory_name": "Project decisions", "status": "processing"}],
                },
            )
        )
        config = TrackingConfig(_tool_call_config({"content": "I prefer dark mode."}, tool_name="membase_save"))
        agent = Agent("a", config=config, tools=[MembaseToolkit(api_key="mbk_test")])

        await agent.ask("Remember that I prefer dark mode.")

        assert _first_result(config).result.parts[0] == DataInput(
            MembaseSaveResult(id="doc_bm90ZQ", status="processing", memory_names=["Project decisions"])
        )
        assert json.loads(route.calls.last.request.content) == {"content": "I prefer dark mode.", "title": ""}

    @respx.mock
    async def test_profile_lists_the_facts(self) -> None:
        respx.get(f"{MEMBASE_BASE_URL}/v1/profile-facts").mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        {"id": "pf_MQ", "content": "Works in Berlin", "source_kind": "user_stated"},
                        {"id": "pf_Mg", "content": " ", "source_kind": "user_stated"},
                    ],
                    "has_more": False,
                    "partial": [],
                },
            )
        )
        config = TrackingConfig(_tool_call_config({}, tool_name="membase_profile"))
        agent = Agent("a", config=config, tools=[MembaseToolkit(api_key="mbk_test")])

        await agent.ask("Who am I?")

        assert _first_result(config).result.parts[0] == DataInput(MembaseProfile(facts=["Works in Berlin"]))
