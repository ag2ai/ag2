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
from ag2.extensions.tools.search import DealportResult, DealportToolkit
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.final.function_tool import FunctionTool, FunctionToolSchema

BASE_URL = "https://app.dealport.com"
SAMPLE_UUID = "00b0ab0b-b28b-4f7a-8336-34cba7366bed"
SAMPLE_BBL = "1008350041"


def _call_config(tool_name: str, arguments: dict[str, Any], *, call_id: str = "call-1") -> TestConfig:
    return TestConfig(
        ModelResponse(
            tool_calls=ToolCallsEvent([
                ToolCallEvent(id=call_id, name=tool_name, arguments=json.dumps(arguments)),
            ])
        ),
        "done",
    )


def _last_tool_data(config: TrackingConfig) -> Any:
    event = config.mock.call_args_list[1].args[0]
    assert isinstance(event, ToolResultsEvent)
    part = event.results[0].result.parts[0]
    assert isinstance(part, DataInput)
    return part.data


@pytest.mark.asyncio
class TestSchema:
    async def test_five_default_tools_and_hidden_app_controls(self, context: Context) -> None:
        toolkit = DealportToolkit()
        raw_schemas = list(await toolkit.schemas(context))
        schemas = [s for s in raw_schemas if isinstance(s, FunctionToolSchema)]
        assert len(schemas) == 5
        assert [s.function.name for s in schemas] == [
            "dealport_preview_companies",
            "dealport_search_companies",
            "dealport_search_buildings",
            "dealport_get_record",
            "dealport_find_owner",
        ]
        for schema in schemas:
            props = schema.function.parameters["properties"]
            assert "api_key" not in props
            assert "limit" not in props
            assert "max_characters" not in props
            assert "ctx" not in props

        assert schemas[3].function.parameters == IsPartialDict({"required": ["record_id"]})
        assert schemas[4].function.parameters == IsPartialDict({"required": ["company_id"]})

    async def test_factory_selection_and_rename(self, context: Context) -> None:
        toolkit = DealportToolkit()
        selected = toolkit.preview_companies(name="size_market", description="Size US SMB market.")
        assert isinstance(selected, FunctionTool)
        [schema] = list(await selected.schemas(context))
        assert isinstance(schema, FunctionToolSchema)
        assert schema.function.name == "size_market"
        assert schema.function.description == "Size US SMB market."

    def test_negative_max_characters_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_characters"):
            DealportToolkit(max_characters=-1)
        with pytest.raises(ValueError, match="max_characters"):
            DealportResult.from_api("/api/v1/companies", {}, -1)


@pytest.mark.asyncio
class TestExecution:
    @respx.mock
    async def test_preview_companies_keyed_uses_rest(self) -> None:
        route = respx.post(f"{BASE_URL}/api/v1/companies/preview").mock(
            return_value=httpx.Response(
                200,
                json={
                    "matching_count": 1420,
                    "cost_usd": 0.0,
                    "sample": [{"id": SAMPLE_UUID, "name": "ACME MECHANICAL LLC", "state": "OH"}],
                },
            )
        )
        toolkit = DealportToolkit(api_key="dp_live_test")
        config = TrackingConfig(_call_config("dealport_preview_companies", {"state": "oh", "naics": ["238220"]}))
        agent = Agent("researcher", config=config, tools=[toolkit])
        await agent.ask("size Ohio HVAC")

        assert route.called
        sent = json.loads(route.calls[0].request.content)
        assert sent == {"state": "OH", "naics": ["238220"]}
        assert route.calls[0].request.headers["authorization"] == "Bearer dp_live_test"
        result = _last_tool_data(config)
        assert isinstance(result, DealportResult)
        assert result.matching_count == 1420
        assert result.cost_usd == 0.0
        assert result.truncated is False

    @respx.mock
    async def test_preview_companies_keyless_uses_mcp_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEALPORT_API_KEY", raising=False)
        monkeypatch.delenv("DP_API_KEY", raising=False)
        inner = json.dumps({"matching_count": 10000, "sample": [{"id": SAMPLE_UUID, "name": "ACME TX"}]})
        envelope = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": inner}]}}
        sse_payload = f"event: message\ndata: {json.dumps(envelope)}\n\n"
        route = respx.post(f"{BASE_URL}/api/mcp").mock(
            return_value=httpx.Response(200, text=sse_payload, headers={"content-type": "text/event-stream"})
        )
        toolkit = DealportToolkit()
        config = TrackingConfig(_call_config("dealport_preview_companies", {"state": "TX"}))
        agent = Agent("researcher", config=config, tools=[toolkit.preview_companies()])
        await agent.ask("size Texas")

        assert route.called
        assert "authorization" not in route.calls[0].request.headers
        result = _last_tool_data(config)
        assert isinstance(result, DealportResult)
        assert result.matching_count == 10000
        assert result.cost_usd == 0.0

    @respx.mock
    async def test_search_companies_resolves_limit_variable_and_truncates(self) -> None:
        route = respx.get(f"{BASE_URL}/api/v1/companies").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"id": SAMPLE_UUID, "name": "LONG COMPANY " * 20}],
                    "next_cursor": "cursor-2",
                    "cost_usd": 0.01,
                },
            )
        )
        toolkit = DealportToolkit(api_key="dp_live_test")
        tool_fn = toolkit.search_companies(limit=Variable("page_size"), max_characters=40)
        config = TrackingConfig(_call_config("dealport_search_companies", {"state": "OH", "q": "Acme"}))
        agent = Agent("researcher", config=config, tools=[tool_fn], variables={"page_size": 5})
        await agent.ask("search Ohio")

        assert route.called
        assert route.calls[0].request.url.params["limit"] == "5"
        assert route.calls[0].request.url.params["state"] == "OH"
        result = _last_tool_data(config)
        assert isinstance(result, DealportResult)
        assert result.cost_usd == 0.01
        assert result.next_cursor == "cursor-2"
        assert result.truncated is True
        assert len(result.data) == 40

    @respx.mock
    async def test_search_buildings_filters(self) -> None:
        route = respx.get(f"{BASE_URL}/api/v1/buildings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"bbl": SAMPLE_BBL, "address": "350 5 AVE"}], "cost_usd": 0.01},
            )
        )
        toolkit = DealportToolkit(api_key="dp_live_test", limit=3)
        config = TrackingConfig(_call_config("dealport_search_buildings", {"borough": "MN", "min_units": 20}))
        agent = Agent("researcher", config=config, tools=[toolkit])
        await agent.ask("search Manhattan buildings")

        assert route.called
        assert route.calls[0].request.url.params["borough"] == "MN"
        assert route.calls[0].request.url.params["min_units"] == "20"
        assert route.calls[0].request.url.params["limit"] == "3"
        result = _last_tool_data(config)
        assert isinstance(result, DealportResult)
        assert result.cost_usd == 0.01

    @respx.mock
    async def test_get_record_uuid_and_bbl(self) -> None:
        uuid_route = respx.get(f"{BASE_URL}/api/v1/records/{SAMPLE_UUID}").mock(
            return_value=httpx.Response(200, json={"data": {"id": SAMPLE_UUID, "cost_usd": 0.0}})
        )
        bbl_route = respx.get(f"{BASE_URL}/api/v1/records/{SAMPLE_BBL}").mock(
            return_value=httpx.Response(200, json={"data": {"bbl": SAMPLE_BBL}, "cost_usd": 0.01})
        )
        toolkit = DealportToolkit(api_key="dp_live_test")

        cfg1 = TrackingConfig(_call_config("dealport_get_record", {"record_id": SAMPLE_UUID}))
        await Agent("r", config=cfg1, tools=[toolkit]).ask("open company")
        assert uuid_route.called
        res1 = _last_tool_data(cfg1)
        assert isinstance(res1, DealportResult)
        assert res1.cost_usd == 0.0

        cfg2 = TrackingConfig(_call_config("dealport_get_record", {"record_id": SAMPLE_BBL}))
        await Agent("r", config=cfg2, tools=[toolkit]).ask("open building")
        assert bbl_route.called
        res2 = _last_tool_data(cfg2)
        assert isinstance(res2, DealportResult)
        assert res2.cost_usd == 0.01

    @respx.mock
    async def test_find_owner_sends_company_uuid_and_person(self) -> None:
        route = respx.post(f"{BASE_URL}/api/v1/owners").mock(
            return_value=httpx.Response(
                200,
                json={"data": {"owner_name": "Jane Doe", "hit": True}, "cost_usd": 0.09},
            )
        )
        toolkit = DealportToolkit(api_key="dp_live_test")
        config = TrackingConfig(_call_config("dealport_find_owner", {"company_id": SAMPLE_UUID, "person": "Jane Doe"}))
        await Agent("r", config=config, tools=[toolkit]).ask("find owner")

        assert route.called
        assert json.loads(route.calls[0].request.content) == {"company_id": SAMPLE_UUID, "person": "Jane Doe"}
        result = _last_tool_data(config)
        assert isinstance(result, DealportResult)
        assert result.cost_usd == 0.09

    @respx.mock
    async def test_invalid_inputs_return_error_without_network_call(self) -> None:
        router = respx.mock(assert_all_called=False)
        with router:
            catch_all = router.route().mock(return_value=httpx.Response(500))
            toolkit = DealportToolkit(api_key="dp_live_test")

            for tool_name, bad_args in [
                ("dealport_preview_companies", {}),
                ("dealport_search_companies", {}),
                ("dealport_search_buildings", {}),
                ("dealport_get_record", {"record_id": "../admin"}),
                ("dealport_get_record", {"record_id": "not-a-uuid"}),
                ("dealport_find_owner", {"company_id": "12345"}),
            ]:
                cfg = TrackingConfig(_call_config(tool_name, bad_args))
                await Agent("r", config=cfg, tools=[toolkit]).ask("bad input")
                res = _last_tool_data(cfg)
                assert isinstance(res, dict) and "error" in res
            assert not catch_all.called

    @respx.mock
    async def test_http_error_does_not_echo_secret(self, context: Context) -> None:
        secret = "dp_live_super_secret_token_999"  # pragma: allowlist secret
        respx.get(f"{BASE_URL}/api/v1/companies").mock(
            return_value=httpx.Response(
                429,
                headers={"Retry-After": "12"},
                json={"error": {"message": "Rate limit exceeded"}},
            )
        )
        selected = DealportToolkit(api_key=secret).search_companies()
        call = ToolCallEvent(id="c-429", name="dealport_search_companies", arguments='{"state":"OH"}')
        error = await selected(call, context)

        assert isinstance(error, ToolErrorEvent)
        err_text = str(error.error)
        assert "429" in err_text
        assert "Retry-After: 12" in err_text
        assert secret not in err_text
