# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest
import respx
from dirty_equals import IsPartialDict

from ag2 import Agent, Context, DataInput, Variable
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolErrorEvent, ToolResultsEvent
from ag2.extensions.tools.search import GlasserToolkit
from ag2.stream import MemoryStream
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.final.function_tool import FunctionTool


def _config(capability: str, arguments: dict[str, Any], *, call_id: str = "call-1") -> TestConfig:
    return TestConfig(
        ModelResponse(
            tool_calls=ToolCallsEvent([
                ToolCallEvent(id=call_id, name=f"glasser_{capability}", arguments=json.dumps(arguments)),
            ])
        ),
        "done",
    )


@pytest.mark.asyncio
class TestSchema:
    async def test_six_default_tools(self, context: Context) -> None:
        schemas = list(await GlasserToolkit("test").schemas(context))
        assert [s.function.name for s in schemas] == [
            "glasser_web_research",
            "glasser_company_intelligence",
            "glasser_people_search",
            "glasser_seo_research",
            "glasser_social_research",
            "glasser_market_data",
        ]
        for schema in schemas:
            properties = schema.function.parameters["properties"]
            assert "provider" not in properties
            assert "api_key" not in properties
            assert "ctx" not in properties
        assert schemas[1].function.parameters == IsPartialDict({"required": ["domain"]})
        assert schemas[4].function.parameters == IsPartialDict({"required": ["platform"]})
        assert schemas[5].function.parameters == IsPartialDict({"required": ["action"]})

    async def test_factory_can_select_and_rename_one_tool(self, context: Context) -> None:
        toolkit = GlasserToolkit("test")
        selected = toolkit.web_research(name="research", description="Find evidence.")
        assert isinstance(selected, FunctionTool)
        [schema] = await selected.schemas(context)
        assert schema.function.name == "research"
        assert schema.function.description == "Find evidence."

    async def test_action_and_platform_enums(self, context: Context) -> None:
        [schema] = await GlasserToolkit("test").social_research().schemas(context)
        properties = schema.function.parameters["properties"]
        assert properties["platform"] == IsPartialDict({
            "enum": ["reddit", "x", "youtube", "tiktok", "instagram", "linkedin"],
        })
        [schema] = await GlasserToolkit("test").market_data().schemas(context)
        properties = schema.function.parameters["properties"]
        assert properties["action"] == IsPartialDict({
            "enum": [
                "property_value",
                "property_rent",
                "property_search",
                "listings_sale",
                "listings_rental",
                "market_stats",
                "stock_quote",
            ],
        })


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


@pytest.mark.asyncio
class TestExecution:
    @pytest.mark.parametrize(
        ("capability", "arguments"),
        [
            ("web_research", {"query": "AG2", "action": "search", "time_range": "week"}),
            ("company_intelligence", {"domain": "example.com", "action": "tech_stack"}),
            ("people_search", {"job_titles": ["CTO"], "seniorities": ["c_suite"], "locations": ["London"]}),
            ("seo_research", {"action": "keyword_gap", "domain": "example.com", "competitors": ["example.org"]}),
            ("social_research", {"platform": "x", "mode": "search", "query": "AG2"}),
            ("market_data", {"action": "stock_quote", "symbol": "MSFT"}),
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
        assert json.loads(request.content) == arguments
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["User-Agent"] == "ag2-glasser-extension"
        assert request.headers["Idempotency-Key"]
        event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
        tool_result = event.results[0].result
        assert tool_result.parts == [DataInput(result)]

    @respx.mock
    async def test_defaults_and_variables(self) -> None:
        route = respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(200, json={"status": "COMPLETED"}),
        )
        toolkit = GlasserToolkit("test", limit=2, country="us", metadata={"user_id": "opaque-42"})
        agent = Agent(
            "a",
            config=_config("web_research", {"query": "AG2"}),
            tools=[
                toolkit.web_research(provider=Variable("provider"), limit=Variable("cap"), task_id=Variable("task"))
            ],
            variables={"provider": "serper", "cap": 5, "task": "research-1"},
        )

        await agent.ask("research")

        assert json.loads(route.calls.last.request.content) == {
            "query": "AG2",
            "provider": "serper",
            "limit": 5,
            "country": "us",
            "task_id": "research-1",
            "metadata": {"user_id": "opaque-42"},
        }

    async def test_missing_variable_fails_before_http(self) -> None:
        toolkit = GlasserToolkit("test")
        agent = Agent(
            "a",
            config=_config("web_research", {"query": "AG2"}),
            tools=[toolkit.web_research(limit=Variable("missing"))],
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
            "platform": "reddit",
            "query": "AG2",
            "metadata": {"user_id": "u1"},
        }

    @respx.mock
    async def test_scrape_budget_is_an_exact_decimal_string(self) -> None:
        route = respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(200, json={"status": "COMPLETED"}),
        )
        arguments = {
            "action": "scrape",
            "url": "https://example.com",
            "formats": ["markdown"],
            "max_charge_usd": "0.005001",
            "timeout_ms": 30000,
            "cache_max_age_seconds": 0,
        }
        agent = Agent("a", config=_config("web_research", arguments), tools=[GlasserToolkit("test")])
        await agent.ask("fetch")
        assert json.loads(route.calls.last.request.content) == arguments

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
        assert tool_result.parts == [DataInput(result)]


@pytest.mark.asyncio
class TestIdempotency:
    @respx.mock
    async def test_replaying_a_tool_call_reuses_key_but_other_executions_do_not(self, context: Context) -> None:
        route = respx.post("https://api.glasser.ai/v1/solutions/gtm/web_research").mock(
            return_value=httpx.Response(200, json={"status": "COMPLETED"}),
        )
        selected = GlasserToolkit("test").web_research()
        call = ToolCallEvent(id="same-call", name="glasser_web_research", arguments='{"query":"AG2"}')
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
        selected = GlasserToolkit("test").web_research()
        call = ToolCallEvent(id="retry", name="glasser_web_research", arguments='{"query":"AG2"}')
        error = await selected(call, context)
        assert isinstance(error, ToolErrorEvent)
        assert isinstance(error.error, httpx.ReadTimeout)
        await selected(call, context)
        assert route.call_count == 2
        assert route.calls[0].request.headers["Idempotency-Key"] == route.calls[1].request.headers["Idempotency-Key"]


@pytest.mark.asyncio
class TestPolling:
    @pytest.mark.parametrize(
        ("capability", "arguments"),
        [
            ("web_research", {"query": "AG2"}),
            ("company_intelligence", {"domain": "example.com"}),
            ("seo_research", {"action": "site_audit", "domain": "example.com"}),
            ("social_research", {"platform": "x", "query": "AG2"}),
            ("market_data", {"action": "stock_quote", "symbol": "MSFT"}),
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
        assert tool_result.parts == [DataInput(final)]

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
        assert tool_result.parts == [DataInput(final)]

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
        assert tool_result.parts == [DataInput(pending)]

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
        assert tool_result.parts == [DataInput(pending)]

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
