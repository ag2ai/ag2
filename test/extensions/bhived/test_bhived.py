# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from dirty_equals import IsPartialDict, IsStr

pytest.importorskip("bhived")

from bhived import APIError, AuthenticationError

from ag2 import Agent, Context, DataInput, Variable
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from ag2.extensions.bhived import (
    BhivedFailedApproach,
    BhivedLesson,
    BhivedQueryResponse,
    BhivedToolkit,
    BhivedWarning,
    BhivedWriteResult,
)
from ag2.testing import TestConfig, TrackingConfig

BHIVED_API_URL = "https://mcp.bhived.ai"

API_KEY = "bhv_test"  # pragma: allowlist secret
TEAM_KEY = "bhv_team"  # pragma: allowlist secret
ENV_KEY = "bhv_env"  # pragma: allowlist secret
REVOKED_KEY = "bhv_revoked"  # pragma: allowlist secret

QUERY_PAYLOAD = {
    "query_id": "q-123",
    "team_recommendations": [
        {
            "id": "t-1",
            "title": "Our CI pins uv 0.11",
            "text": "Pin uv 0.11 in CI; 0.12 changed the lockfile format.",
            "type": "instruction",
            "status": "active",
            "score": 0.9,
        }
    ],
    "public_recommendations": [
        {
            "id": "m-1",
            "title": "Pin httpx below 1.0 with respx 0.21",
            "text": "respx 0.21 breaks with httpx 1.0 betas. Pin httpx<1 until respx ships support.",
            "type": "instruction",
            "status": "active",
            "score": 0.83,
            "corroboration_count": 2,
            "contradiction_count": 1,
        },
        {
            "id": "m-skill",
            "title": "pytest helper skill",
            "text": "A skill bundle",
            "type": "note",
            "status": "active",
            "score": 0.5,
            "has_skill": True,
            "skill_meta": {"name": "pytest-helper", "description": "Runs and triages pytest suites."},
        },
    ],
    "team_warnings": [],
    "public_warnings": [
        {
            "id": "w-1",
            "title": "Do NOT patch httpx.Client.send",
            "text": "Patching send skips transport hooks and hides timeout bugs.",
            "type": "mistake",
            "confidence": 0.72,
        }
    ],
    "public_disputed": [],
    "public_failed_approaches": [{"id": "f-1", "title": "Monkeypatching sockets", "reason": "Flaky on Windows."}],
}

WRITE_PAYLOAD = {
    "memory_id": "new-1",
    "action_performed": "created",
    "query_id_linked": "q-123",
    "supersedes_id_linked": None,
    "entities_created": 3,
    "corroborations_created": 1,
    "contradictions_created": 0,
}


@pytest.fixture
def context() -> Context:
    return Context(stream=MagicMock())


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("BHIVED_API_KEY", "BHIVED_API_URL", "BHIVED_TIMEOUT", "BHIVED_WARMUP_RETRIES"):
        monkeypatch.delenv(name, raising=False)
    # The SDK falls back to ~/.bhived/config.json; point home at an empty directory.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))


def _tool_call_config(arguments: dict[str, object], *, tool_name: str, final_reply: str = "done") -> TestConfig:
    return TestConfig(
        ModelResponse(tool_calls=ToolCallsEvent([ToolCallEvent(arguments=json.dumps(arguments), name=tool_name)])),
        final_reply,
    )


def _first_result(config: TrackingConfig) -> object:
    tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
    return tool_results_event.results[0].result.parts[0]


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schemas(self, context: Context) -> None:
        toolkit = BhivedToolkit(api_key=API_KEY)

        schemas = list(await toolkit.schemas(context))

        assert [s.function.name for s in schemas] == [
            "bhived_query",
            "bhived_write_instruction",
            "bhived_write_mistake",
            "bhived_write_update",
        ]
        query, *writes = schemas
        assert query.function.parameters == IsPartialDict({
            "required": ["query"],
            "properties": IsPartialDict({
                "query": IsPartialDict({"type": "string"}),
                "context": IsPartialDict({"default": None}),
            }),
        })
        for write in writes:
            assert write.function.parameters == IsPartialDict({
                "required": ["text", "title"],
                "properties": IsPartialDict({
                    "text": IsPartialDict({"type": "string"}),
                    "title": IsPartialDict({"type": "string"}),
                    "query_id": IsPartialDict({"default": None}),
                }),
            })

    async def test_write_descriptions_carry_the_privacy_rules(self, context: Context) -> None:
        schemas = list(await BhivedToolkit(api_key=API_KEY).schemas(context))

        for schema in schemas[1:]:
            assert "public shared lessons" in schema.function.description
            assert "Never include secrets" in schema.function.description

    async def test_custom_name_and_description(self, context: Context) -> None:
        tool = BhivedToolkit(api_key=API_KEY).write_mistake(name="warn", description="Warn other agents.")

        [schema] = list(await tool.schemas(context))

        assert schema.function.name == "warn"
        assert schema.function.description == "Warn other agents."


@pytest.mark.asyncio
class TestQuery:
    @respx.mock
    async def test_returns_lessons_warnings_and_failed_approaches(self) -> None:
        route = respx.post(f"{BHIVED_API_URL}/v2/query").mock(return_value=httpx.Response(200, json=QUERY_PAYLOAD))
        config = TrackingConfig(
            _tool_call_config(
                {"query": "respx httpx 1.0 TypeError", "context": "python 3.12"}, tool_name="bhived_query"
            )
        )
        agent = Agent("a", config=config, tools=[BhivedToolkit(api_key=API_KEY)])

        await agent.ask("Why do my respx mocks fail?")

        assert _first_result(config) == DataInput(
            BhivedQueryResponse(
                query="respx httpx 1.0 TypeError",
                query_id="q-123",
                lessons=[
                    BhivedLesson(
                        id="t-1",
                        title="Our CI pins uv 0.11",
                        text="Pin uv 0.11 in CI; 0.12 changed the lockfile format.",
                        type="instruction",
                        tier="team",
                        score=0.9,
                    ),
                    BhivedLesson(
                        id="m-1",
                        title="Pin httpx below 1.0 with respx 0.21",
                        text="respx 0.21 breaks with httpx 1.0 betas. Pin httpx<1 until respx ships support.",
                        type="instruction",
                        tier="public",
                        score=0.83,
                        corroborations=2,
                        contradictions=1,
                    ),
                ],
                warnings=[
                    BhivedWarning(
                        id="w-1",
                        title="Do NOT patch httpx.Client.send",
                        text="Patching send skips transport hooks and hides timeout bugs.",
                        tier="public",
                        confidence=0.72,
                    )
                ],
                failed_approaches=[
                    BhivedFailedApproach(title="Monkeypatching sockets", text="Flaky on Windows.", tier="public")
                ],
            )
        )
        request = route.calls.last.request
        assert request.headers["x-api-key"] == API_KEY
        assert request.headers["user-agent"] == IsStr(regex=r"bhived-python/\S+ ag2")
        assert json.loads(request.content) == {
            "query": "respx httpx 1.0 TypeError",
            "context": "python 3.12",
            "top_k": 5,
            "include_episodes": True,
            "include_warnings": True,
            "include_disputed": True,
        }

    @respx.mock
    async def test_api_key_top_k_and_scope_resolve_from_variables(self) -> None:
        route = respx.post(f"{BHIVED_API_URL}/v2/query").mock(return_value=httpx.Response(200, json=QUERY_PAYLOAD))
        toolkit = BhivedToolkit(api_key=Variable("bhived_api_key"))
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="bhived_query"),
            tools=[toolkit.query(top_k=Variable(), scope=Variable("bhived_scope"))],
            variables={
                "bhived_api_key": TEAM_KEY,
                "top_k": 3,
                "bhived_scope": "team_only",
            },
        )

        await agent.ask("search")

        request = route.calls.last.request
        assert request.headers["x-api-key"] == TEAM_KEY
        assert json.loads(request.content) == IsPartialDict({"top_k": 3, "scope": "team_only"})

    @respx.mock
    async def test_api_key_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BHIVED_API_KEY", ENV_KEY)
        route = respx.post(f"{BHIVED_API_URL}/v2/query").mock(return_value=httpx.Response(200, json=QUERY_PAYLOAD))
        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="bhived_query"), tools=[BhivedToolkit()])

        await agent.ask("search")

        assert route.calls.last.request.headers["x-api-key"] == ENV_KEY

    @respx.mock
    async def test_api_url(self) -> None:
        route = respx.post("http://127.0.0.1:8000/v2/query").mock(return_value=httpx.Response(200, json=QUERY_PAYLOAD))
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="bhived_query"),
            tools=[BhivedToolkit(api_key=API_KEY, api_url="http://127.0.0.1:8000")],
        )

        await agent.ask("search")

        assert route.called

    @respx.mock
    async def test_rejected_key_raises_the_sdk_error(self) -> None:
        respx.post(f"{BHIVED_API_URL}/v2/query").mock(
            return_value=httpx.Response(403, json={"detail": "invalid_api_key"})
        )
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="bhived_query"),
            tools=[BhivedToolkit(api_key=REVOKED_KEY)],
        )

        with pytest.raises(AuthenticationError, match="403"):
            await agent.ask("search")

    async def test_missing_key_raises_before_any_request(self) -> None:
        agent = Agent("a", config=_tool_call_config({"query": "q"}, tool_name="bhived_query"), tools=[BhivedToolkit()])

        with (
            respx.mock(assert_all_called=False) as router,
            pytest.raises(AuthenticationError, match="not authenticated"),
        ):
            await agent.ask("search")

        assert not router.calls

    async def test_missing_variable_raises(self) -> None:
        agent = Agent(
            "a",
            config=_tool_call_config({"query": "q"}, tool_name="bhived_query"),
            tools=[BhivedToolkit(api_key=Variable("bhived_api_key"))],
        )

        with pytest.raises(KeyError):
            await agent.ask("search")


@pytest.mark.asyncio
class TestWrite:
    @pytest.mark.parametrize("lesson_type", ["instruction", "mistake", "update"])
    @respx.mock
    async def test_write_links_the_query_and_reports_the_lesson(self, lesson_type: str) -> None:
        route = respx.post(f"{BHIVED_API_URL}/v1/memories").mock(return_value=httpx.Response(200, json=WRITE_PAYLOAD))
        config = TrackingConfig(
            _tool_call_config(
                {"text": "Pin httpx<1 with respx 0.21.", "title": "respx 0.21 needs httpx<1", "query_id": "q-123"},
                tool_name=f"bhived_write_{lesson_type}",
            )
        )
        agent = Agent("a", config=config, tools=[BhivedToolkit(api_key=API_KEY)])

        await agent.ask("Share what fixed it.")

        assert _first_result(config) == DataInput(
            BhivedWriteResult(
                saved=True,
                lesson_type=lesson_type,
                bhived_id="new-1",
                action="created",
                query_id_linked="q-123",
            )
        )
        request = route.calls.last.request
        assert request.headers["idempotency-key"] == IsStr(min_length=1)
        assert json.loads(request.content) == {
            "text": "Pin httpx<1 with respx 0.21.",
            "title": "respx 0.21 needs httpx<1",
            "type": lesson_type,
            "action": "new",
            "query_id": "q-123",
        }

    @respx.mock
    async def test_write_without_query_id_sends_no_link(self) -> None:
        route = respx.post(f"{BHIVED_API_URL}/v1/memories").mock(
            return_value=httpx.Response(200, json={**WRITE_PAYLOAD, "query_id_linked": None})
        )
        agent = Agent(
            "a",
            config=_tool_call_config({"text": "t", "title": "t"}, tool_name="bhived_write_update"),
            tools=[BhivedToolkit(api_key=API_KEY).write_update()],
        )

        await agent.ask("Share it.")

        assert "query_id" not in json.loads(route.calls.last.request.content)

    @respx.mock
    async def test_near_duplicate_is_reported_not_raised(self) -> None:
        respx.post(f"{BHIVED_API_URL}/v1/memories").mock(
            return_value=httpx.Response(
                409,
                json={"detail": {"error": "duplicate_memory", "duplicate_of": "m-old", "jaccard_similarity": 0.91}},
            )
        )
        config = TrackingConfig(_tool_call_config({"text": "t", "title": "t"}, tool_name="bhived_write_mistake"))
        agent = Agent("a", config=config, tools=[BhivedToolkit(api_key=API_KEY)])

        await agent.ask("Warn others.")

        assert _first_result(config) == DataInput(
            BhivedWriteResult(saved=False, lesson_type="mistake", duplicate_of="m-old", similarity=0.91)
        )

    @respx.mock
    async def test_rejected_write_raises_the_sdk_error(self) -> None:
        respx.post(f"{BHIVED_API_URL}/v1/memories").mock(
            return_value=httpx.Response(422, json={"detail": "text is empty after cleanup"})
        )
        agent = Agent(
            "a",
            config=_tool_call_config({"text": "t", "title": "t"}, tool_name="bhived_write_instruction"),
            tools=[BhivedToolkit(api_key=API_KEY)],
        )

        with pytest.raises(APIError, match="422"):
            await agent.ask("Share it.")

    async def test_title_over_100_characters_is_rejected_before_any_request(self) -> None:
        agent = Agent(
            "a",
            config=_tool_call_config({"text": "t", "title": "x" * 101}, tool_name="bhived_write_instruction"),
            tools=[BhivedToolkit(api_key=API_KEY)],
        )

        with respx.mock(assert_all_called=False) as router, pytest.raises(ValueError, match="at most 100 characters"):
            await agent.ask("Share it.")

        assert not router.calls
