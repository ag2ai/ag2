# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest
import respx
from dirty_equals import IsPartialDict, IsStr

from ag2 import Agent, Context, DataInput, Variable
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolErrorEvent, ToolResultsEvent
from ag2.extensions.tools.search import GlasserToolkit
from ag2.extensions.tools.search.glasser import GlasserResult
from ag2.stream import MemoryStream
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.final.function_tool import FunctionTool


def _config(capability: str, arguments: dict[str, Any], *, call_id: str = "call-1") -> TestConfig:
    return TestConfig(
        ModelResponse(
            tool_calls=ToolCallsEvent([
                ToolCallEvent(
                    id=call_id,
                    name={
                        "web_research": "glasser_web_search",
                        "company_intelligence": "glasser_company_profile",
                        "people_search": "glasser_find_people",
                        "seo_research": "glasser_keyword_overview",
                        "social_research": "glasser_social_search",
                        "market_data": "glasser_stock_quote",
                    }[capability],
                    arguments=json.dumps(arguments),
                ),
            ])
        ),
        "done",
    )


@pytest.mark.asyncio
class TestSchema:
    async def test_six_default_tools(self, context: Context) -> None:
        schemas = list(await GlasserToolkit("test").schemas(context))
        assert [s.function.name for s in schemas] == [
            "glasser_web_search",
            "glasser_company_profile",
            "glasser_find_people",
            "glasser_keyword_overview",
            "glasser_social_search",
            "glasser_stock_quote",
        ]
        for schema in schemas:
            properties = schema.function.parameters["properties"]
            assert "provider" not in properties
            assert "api_key" not in properties
            assert "ctx" not in properties
        assert schemas[1].function.parameters == IsPartialDict({"required": ["domain"]})
        assert schemas[4].function.parameters == IsPartialDict({"required": ["platform", "query"]})
        assert schemas[5].function.parameters == IsPartialDict({"required": ["symbol"]})
        assert all("action" not in schema.function.parameters["properties"] for schema in schemas)

    async def test_factory_can_select_and_rename_one_tool(self, context: Context) -> None:
        toolkit = GlasserToolkit("test")
        selected = toolkit.web_search(name="research", description="Find evidence.")
        assert isinstance(selected, FunctionTool)
        [schema] = await selected.schemas(context)
        assert schema.function.name == "research"
        assert schema.function.description == "Find evidence."

    async def test_platform_enum_and_small_schemas(self, context: Context) -> None:
        [schema] = await GlasserToolkit("test").social_search().schemas(context)
        properties = schema.function.parameters["properties"]
        assert properties["platform"] == IsPartialDict({
            "enum": ["reddit", "x", "youtube", "tiktok", "instagram", "linkedin"],
        })
        schemas = list(await GlasserToolkit("test").schemas(context))
        schema_size = sum(
            len(json.dumps(schema.function.parameters)) + len(schema.function.description) for schema in schemas
        )
        assert schema_size < 3000


@pytest.mark.parametrize(
    "settings",
    [
        {"api_key": ""},
        {"api_key": "   "},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"wait_timeout": -1},
        {"wait_timeout": float("inf")},
        {"poll_interval": 0},
        {"poll_interval": float("nan")},
        {"base_url": "file:///tmp/api"},
        {"base_url": "https://secret@example.com"},
        {"base_url": "https://example.com?token=secret"},
        {"base_url": "https://example.com/#fragment"},
    ],
)
def test_invalid_connection_settings(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        GlasserToolkit(**({"api_key": "test"} | settings))


def test_result_rejects_negative_content_limit() -> None:
    with pytest.raises(ValueError, match="max_characters must be non-negative"):
        GlasserResult.from_api({"status": "COMPLETED"}, -1)


@pytest.mark.asyncio
class TestExecution:
    @pytest.mark.parametrize(
        ("capability", "arguments"),
        [
            ("web_research", {"query": "AG2"}),
            ("company_intelligence", {"domain": "example.com"}),
            ("people_search", {"job_titles": ["CTO"], "locations": ["London"]}),
            ("seo_research", {"keywords": ["agent framework"]}),
            ("social_research", {"platform": "x", "mode": "search", "query": "AG2"}),
            ("market_data", {"symbol": "MSFT"}),
        ],
    )
    @respx.mock
    async def test_all_six_tools_preserve_payloads_and_results(
        self, capability: str, arguments: dict[str, Any]
    ) -> None:
        result = {
            "id": "run-1",
            "status": "COMPLETED",
            "result_status": "success",
            "output": {"items": [{"name": "Example", "missing": None}]},
            "charge_usd": "0.001234",
            "provider": "example",
            "runs": [{"id": "run-1", "charge_usd": "0.001234"}],
            "run_url": "https://glasser.ai/runs/run-1",
        }
        route = respx.post(f"https://api.glasser.ai/v1/solutions/gtm/{capability}").mock(
            return_value=httpx.Response(200, json=result),
        )
        config = TrackingConfig(_config(capability, arguments))
        agent = Agent("a", config=config, tools=[GlasserToolkit("test-key")])

        await agent.ask("research")

        request = route.calls.last.request
        actions = {
            "web_research": "search",
            "company_intelligence": "enrich",
            "people_search": "search",
            "seo_research": "keyword_overview",
            "social_research": None,
            "market_data": "stock_quote",
        }
        expected = arguments | (
            {"mode": "search"} if capability == "social_research" else {"action": actions[capability]}
        )
        assert json.loads(request.content) == expected
        assert dict(request.headers) == IsPartialDict({
            "authorization": "Bearer test-key",
            "user-agent": "ag2-glasser-extension",
            "idempotency-key": IsStr(min_length=1),
        })
        event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        tool_result = event.results[0].result
        assert tool_result.parts == [
            DataInput(
                GlasserResult(
                    id="run-1",
                    status="COMPLETED",
                    result_status="success",
                    provider="example",
                    charge_usd="0.001234",
                    output={"items": [{"name": "Example", "missing": None}]},
                    runs=[{"id": "run-1", "charge_usd": "0.001234"}],
                    run_url="https://glasser.ai/runs/run-1",
                )
            )
        ]

    @respx.mock
    async def test_defaults_and_variables(self) -> None:
        route = respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(200, json={"status": "COMPLETED"}),
        )
        toolkit = GlasserToolkit("test", limit=2, country="us", metadata={"user_id": "opaque-42"})
        agent = Agent(
            "a",
            config=_config("web_research", {"query": "AG2"}),
            tools=[toolkit.web_search(provider=Variable("provider"), limit=Variable("cap"), task_id=Variable("task"))],
            variables={"provider": "serper", "cap": 5, "task": "research-1"},
        )

        await agent.ask("research")

        assert json.loads(route.calls.last.request.content) == {
            "action": "search",
            "query": "AG2",
            "provider": "serper",
            "limit": 5,
            "task_id": "research-1",
        }

    @respx.mock
    async def test_every_factory_setting_resolves_a_variable(self) -> None:
        route = respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(200, json={"status": "COMPLETED", "output": "abcdef"}),
        )
        toolkit = GlasserToolkit("test")
        config = TrackingConfig(_config("web_research", {"query": "AG2"}))
        agent = Agent(
            "a",
            config=config,
            tools=[
                toolkit.web_search(
                    provider=Variable("provider"),
                    limit=Variable("limit"),
                    country=Variable("country"),
                    task_id=Variable("task_id"),
                    metadata=Variable("metadata"),
                    max_characters=Variable("max_characters"),
                )
            ],
            variables={
                "provider": "serper",
                "limit": 5,
                "country": "us",
                "task_id": "task-1",
                "metadata": {"user_id": "u1"},
                "max_characters": 3,
            },
        )
        await agent.ask("search")
        assert json.loads(route.calls.last.request.content) == {
            "action": "search",
            "query": "AG2",
            "provider": "serper",
            "limit": 5,
            "country": "us",
            "task_id": "task-1",
            "metadata": {"user_id": "u1"},
        }
        event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert event.results[0].result.parts == [
            DataInput(GlasserResult(status="COMPLETED", output="abc", truncated=True))
        ]

    @respx.mock
    async def test_constructor_settings_reach_default_tool(self) -> None:
        route = respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(200, json={"status": "COMPLETED"}),
        )
        await Agent(
            "a",
            config=_config("web_research", {"query": "AG2"}),
            tools=[
                GlasserToolkit(
                    "test", provider="serper", limit=2, country="gb", task_id="task-1", metadata={"user_id": "u1"}
                )
            ],
        ).ask("search")
        assert json.loads(route.calls.last.request.content) == {
            "action": "search",
            "query": "AG2",
            "provider": "serper",
            "limit": 2,
            "country": "gb",
            "task_id": "task-1",
            "metadata": {"user_id": "u1"},
        }

    @pytest.mark.parametrize(
        ("capability", "arguments", "error"),
        [
            ("web_research", {"query": "  "}, "query must not be empty"),
            ("company_intelligence", {"domain": ""}, "domain must not be empty"),
            ("people_search", {}, "supply job_titles, locations, company_domain or keywords"),
            ("people_search", {"job_titles": ["  "]}, "supply job_titles, locations, company_domain or keywords"),
            ("seo_research", {"keywords": []}, "keywords must contain a non-empty value"),
            ("social_research", {"platform": "x", "query": ""}, "query must not be empty"),
            ("market_data", {"symbol": "  "}, "symbol must not be empty"),
        ],
    )
    @respx.mock
    async def test_invalid_input_returns_error_without_paid_call(
        self, capability: str, arguments: dict[str, Any], error: str
    ) -> None:
        route = respx.post(f"https://api.glasser.ai/v1/solutions/gtm/{capability}").mock(
            return_value=httpx.Response(200, json={"status": "COMPLETED"}),
        )
        config = TrackingConfig(_config(capability, arguments))
        await Agent("a", config=config, tools=[GlasserToolkit("test")]).ask("search")
        assert not route.called
        event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert event.results[0].result.parts == [DataInput({"error": error})]

    @respx.mock
    async def test_truncation_never_cuts_status_ids_urls_or_charge(self) -> None:
        result = {
            "id": "run-123",
            "status": "COMPLETED",
            "result_status": "success",
            "provider": "serper",
            "charge_usd": "0.001100",
            "output": "abcdef",
            "error": "failed",
            "result_url": "https://api.glasser.ai/v1/solutions/gtm/web_research/result?token=opaque",
            "runs": [{"id": "run-123"}],
        }
        respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(200, json=result),
        )
        config = TrackingConfig(_config("web_research", {"query": "AG2"}))
        await Agent("a", config=config, tools=[GlasserToolkit("test", max_characters=3)]).ask("search")
        event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        assert event.results[0].result.parts == [
            DataInput(
                GlasserResult(
                    id="run-123",
                    status="COMPLETED",
                    result_status="success",
                    provider="serper",
                    charge_usd="0.001100",
                    output="abc",
                    error="",
                    result_url=result["result_url"],
                    runs=[{"id": "run-123"}],
                    truncated=True,
                )
            )
        ]

    @respx.mock
    async def test_negative_content_limit_fails_before_paid_call(self) -> None:
        route = respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(200, json={"status": "COMPLETED"}),
        )
        agent = Agent(
            "a",
            config=_config("web_research", {"query": "AG2"}),
            tools=[GlasserToolkit("test", max_characters=-1)],
        )
        with pytest.raises(ValueError, match="max_characters must be non-negative"):
            await agent.ask("search")
        assert not route.called

    async def test_missing_variable_fails_before_http(self) -> None:
        toolkit = GlasserToolkit("test")
        agent = Agent(
            "a",
            config=_config("web_research", {"query": "AG2"}),
            tools=[toolkit.web_search(limit=Variable("missing"))],
        )
        with pytest.raises(KeyError, match="missing"):
            await agent.ask("research")

    @respx.mock
    async def test_social_sends_only_supported_defaults(self) -> None:
        route = respx.post("https://api.glasser.ai/v1/solutions/gtm/social_research").mock(
            return_value=httpx.Response(200, json={"status": "COMPLETED"}),
        )
        agent = Agent(
            "a",
            config=_config("social_research", {"platform": "reddit", "query": "AG2"}),
            tools=[GlasserToolkit("test", limit=5, country="us", task_id="task-1", metadata={"user_id": "u1"})],
        )
        await agent.ask("research")
        assert json.loads(route.calls.last.request.content) == {
            "mode": "search",
            "platform": "reddit",
            "query": "AG2",
            "metadata": {"user_id": "u1"},
        }

    @respx.mock
    async def test_factory_does_not_inherit_constructor_settings(self) -> None:
        route = respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(200, json={"status": "COMPLETED"}),
        )
        toolkit = GlasserToolkit("test", provider="serper", limit=2)
        await Agent("a", config=_config("web_research", {"query": "AG2"}), tools=[toolkit.web_search()]).ask("search")
        assert json.loads(route.calls.last.request.content) == {"action": "search", "query": "AG2"}

    @respx.mock
    async def test_custom_gateway_path(self) -> None:
        route = respx.post("https://proxy.example.com/glasser/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(200, json={"status": "COMPLETED"}),
        )
        agent = Agent(
            "a",
            config=_config("web_research", {"query": "AG2"}),
            tools=[GlasserToolkit("test", base_url="https://proxy.example.com/glasser//")],
        )
        await agent.ask("research")
        assert route.called

    @pytest.mark.parametrize("status", [400, 401, 402, 403, 409, 429, 503])
    @respx.mock
    async def test_http_errors_are_not_retried(self, status: int) -> None:
        route = respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(status, json={"error": "refused"}),
        )
        agent = Agent("a", config=_config("web_research", {"query": "AG2"}), tools=[GlasserToolkit("test")])
        with pytest.raises(httpx.HTTPStatusError):
            await agent.ask("research")
        assert route.call_count == 1

    @pytest.mark.parametrize("payload", [None, [], "invalid"])
    @respx.mock
    async def test_non_object_response_is_an_error(self, payload: Any) -> None:
        respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(200, content=json.dumps(payload), headers={"content-type": "application/json"}),
        )
        agent = Agent("a", config=_config("web_research", {"query": "AG2"}), tools=[GlasserToolkit("test")])
        with pytest.raises(ValueError, match="non-object"):
            await agent.ask("research")

    @pytest.mark.parametrize("result_status", ["error", "no_result", "partial"])
    @respx.mock
    async def test_business_outcome_is_preserved(self, result_status: str) -> None:
        result = {"status": "COMPLETED", "result_status": result_status, "output": None, "charge_usd": "0.01"}
        respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(200, json=result),
        )
        config = TrackingConfig(_config("web_research", {"query": "AG2"}))
        await Agent("a", config=config, tools=[GlasserToolkit("test")]).ask("research")
        event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        tool_result = event.results[0].result
        assert tool_result.parts == [
            DataInput(GlasserResult(status="COMPLETED", result_status=result_status, charge_usd="0.01"))
        ]


@pytest.mark.asyncio
class TestIdempotency:
    @respx.mock
    async def test_replaying_a_tool_call_reuses_key_but_other_executions_do_not(self, context: Context) -> None:
        route = respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(200, json={"status": "COMPLETED"}),
        )
        selected = GlasserToolkit("test").web_search()
        call = ToolCallEvent(id="same-call", name="glasser_web_search", arguments='{"query":"AG2"}')
        await selected(call, context)
        await selected(call, context)
        await selected(ToolCallEvent(id="another-call", name=call.name, arguments=call.arguments), context)
        await selected(call, Context(stream=MemoryStream()))
        keys = [c.request.headers["Idempotency-Key"] for c in route.calls]
        assert keys[0] == keys[1]
        assert len({keys[0], keys[2], keys[3]}) == 3

    @respx.mock
    async def test_transport_timeout_can_be_replayed_with_the_same_key(self, context: Context) -> None:
        route = respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            side_effect=[
                httpx.ReadTimeout("response lost"),
                httpx.Response(200, json={"status": "COMPLETED"}),
            ]
        )
        selected = GlasserToolkit("test").web_search()
        call = ToolCallEvent(id="retry", name="glasser_web_search", arguments='{"query":"AG2"}')
        error = await selected(call, context)
        assert isinstance(error, ToolErrorEvent)
        assert isinstance(error.error, httpx.ReadTimeout)
        await selected(call, context)
        assert route.call_count == 2
        assert route.calls[0].request.headers["Idempotency-Key"] == route.calls[1].request.headers["Idempotency-Key"]


@pytest.mark.asyncio
class TestPolling:
    @pytest.mark.parametrize(
        ("capability", "arguments", "pending", "message"),
        [
            (
                "people_search",
                {"keywords": "engineer"},
                {"id": "bad/id", "status": "RUNNING"},
                "invalid pending Run id",
            ),
            ("web_research", {"query": "AG2"}, {"status": "RUNNING"}, "no result_url"),
        ],
    )
    @respx.mock
    async def test_pending_result_requires_safe_retrieval_reference(
        self, capability: str, arguments: dict[str, Any], pending: dict[str, Any], message: str
    ) -> None:
        route = respx.post(f"https://api.glasser.ai/v1/solutions/gtm/{capability}").mock(
            return_value=httpx.Response(202, json=pending),
        )
        agent = Agent("a", config=_config(capability, arguments), tools=[GlasserToolkit("test")])
        with pytest.raises(ValueError, match=message):
            await agent.ask("research")
        assert route.call_count == 1

    @pytest.mark.parametrize(
        ("capability", "arguments"),
        [
            ("web_research", {"query": "AG2"}),
            ("company_intelligence", {"domain": "example.com"}),
            ("seo_research", {"keywords": ["agent framework"]}),
            ("social_research", {"platform": "x", "query": "AG2"}),
            ("market_data", {"symbol": "MSFT"}),
        ],
    )
    @respx.mock
    async def test_result_url_polled_until_business_result_is_terminal(
        self, capability: str, arguments: dict[str, Any]
    ) -> None:
        path = f"https://api.glasser.ai/v1/solutions/gtm/{capability}"
        url = f"{path}/result?token=opaque"
        pending = {"id": "run-1", "status": "RUNNING", "result_status": "pending", "result_url": url}
        final = {
            "id": "run-1",
            "status": "COMPLETED",
            "result_status": "success",
            "output": {"items": []},
            "charge_usd": "0.02",
        }
        post = respx.post(path).mock(return_value=httpx.Response(202, json=pending))
        get = respx.get(url).mock(
            side_effect=[
                httpx.Response(200, json=pending | {"status": "COMPLETED"}),
                httpx.Response(200, json=final),
            ]
        )
        config = TrackingConfig(_config(capability, arguments))
        await Agent("a", config=config, tools=[GlasserToolkit("test", poll_interval=0.001)]).ask("research")
        assert post.call_count == 1
        assert get.call_count == 2
        assert get.calls.last.request.headers["Authorization"] == "Bearer test"
        event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        tool_result = event.results[0].result
        assert tool_result.parts == [
            DataInput(
                GlasserResult(
                    id="run-1",
                    status="COMPLETED",
                    result_status="success",
                    charge_usd="0.02",
                    output={"items": []},
                )
            )
        ]

    @respx.mock
    async def test_people_search_polls_native_run(self) -> None:
        run_id = "de1875ac-af29-4d6c-b9c4-568ae4d9604f"
        pending = {"id": run_id, "status": "QUEUED", "charge_usd": None}
        final = {
            "id": run_id,
            "status": "COMPLETED",
            "provider": "pdl",
            "output": {"data": []},
            "provider_response": {"http_status": 200},
            "charge_usd": "0.02",
        }
        post = respx.post("https://api.glasser.ai/v1/solutions/gtm/people_search").mock(
            return_value=httpx.Response(202, json=pending),
        )
        get = respx.get(f"https://api.glasser.ai/v1/runs/{run_id}").mock(return_value=httpx.Response(200, json=final))
        config = TrackingConfig(_config("people_search", {"keywords": "engineer"}))
        await Agent("a", config=config, tools=[GlasserToolkit("test", poll_interval=0.001)]).ask("research")
        assert post.call_count == get.call_count == 1
        event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        tool_result = event.results[0].result
        assert tool_result.parts == [
            DataInput(
                GlasserResult(
                    id=run_id,
                    status="COMPLETED",
                    provider="pdl",
                    charge_usd="0.02",
                    output={"data": []},
                    provider_response={"http_status": 200},
                )
            )
        ]

    @pytest.mark.parametrize(
        "failure",
        [
            httpx.ReadTimeout("poll response lost"),
            httpx.Response(503, json={"error": "try later"}),
            httpx.Response(200, json=["unexpected"]),
        ],
    )
    @respx.mock
    async def test_failed_poll_preserves_admitted_result(self, failure: Any) -> None:
        url = "https://api.glasser.ai/v1/solutions/gtm/web_research/result?token=opaque"
        pending = {
            "id": "run-1",
            "status": "RUNNING",
            "result_status": "pending",
            "result_url": url,
            "charge_usd": None,
        }
        post = respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(202, json=pending),
        )
        get = respx.get(url).mock(
            side_effect=failure if isinstance(failure, Exception) else None,
            return_value=failure if isinstance(failure, httpx.Response) else None,
        )
        config = TrackingConfig(_config("web_research", {"query": "AG2"}))
        await Agent("a", config=config, tools=[GlasserToolkit("test", poll_interval=0.001)]).ask("research")
        assert post.call_count == get.call_count == 1
        event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        tool_result = event.results[0].result
        assert tool_result.parts == [
            DataInput(
                GlasserResult(
                    id="run-1",
                    status="RUNNING",
                    result_status="pending",
                    result_url=url,
                )
            )
        ]

    @pytest.mark.parametrize("wait_timeout", [0.0, 0.001])
    @respx.mock
    async def test_poll_budget_returns_pending_without_resubmitting(self, wait_timeout: float) -> None:
        url = "https://api.glasser.ai/v1/solutions/gtm/web_research/result?token=opaque"
        pending = {"status": "RUNNING", "result_status": "pending", "result_url": url, "charge_usd": None}
        post = respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(202, json=pending),
        )
        config = TrackingConfig(_config("web_research", {"query": "AG2"}))
        await Agent("a", config=config, tools=[GlasserToolkit("test", wait_timeout=wait_timeout, poll_interval=1)]).ask(
            "research"
        )
        assert post.call_count == 1
        event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        tool_result = event.results[0].result
        assert tool_result.parts == [
            DataInput(GlasserResult(status="RUNNING", result_status="pending", result_url=url))
        ]

    @pytest.mark.parametrize(
        "result_url",
        [
            "https://evil.example/v1/solutions/gtm/web_research/result?token=x",
            "http://api.glasser.ai/v1/solutions/gtm/web_research/result?token=x",
            "https://api.glasser.ai:8443/v1/solutions/gtm/web_research/result?token=x",
            "https://api.glasser.ai/v1/solutions/gtm/seo_research/result?token=x",
            "https://secret@api.glasser.ai/v1/solutions/gtm/web_research/result?token=x",
            "https://api.glasser.ai/v1/solutions/gtm/web_research/result?token=x#fragment",
        ],
    )
    @respx.mock
    async def test_rejects_untrusted_result_urls_before_sending_credentials(self, result_url: str) -> None:
        respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(202, json={"status": "RUNNING", "result_url": result_url}),
        )
        agent = Agent("a", config=_config("web_research", {"query": "AG2"}), tools=[GlasserToolkit("test")])
        with pytest.raises(ValueError, match="configured API origin"):
            await agent.ask("research")

    @respx.mock
    async def test_redirects_are_not_followed(self) -> None:
        respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(307, headers={"location": "https://evil.example"}),
        )
        agent = Agent("a", config=_config("web_research", {"query": "AG2"}), tools=[GlasserToolkit("test")])
        with pytest.raises(httpx.HTTPStatusError):
            await agent.ask("research")
