# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
import pytest
import respx
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from ag2.tools.final.function_tool import FunctionToolSchema

from ag2 import Agent, Context, DataInput
from ag2.extensions.tools.finance.fxmacrodata import (
    FXMacroDataError,
    FXMacroDataResponse,
    FXMacroDataToolkit,
    _get,
)
from ag2.testing import TestConfig, TrackingConfig

BASE_URL = "https://api.fxmacrodata.com"

ANNOUNCEMENTS = {
    "currency": "USD",
    "indicator": "inflation",
    "name": "Inflation (CPI)",
    "source": "BLS",
    "data": [
        {
            "date": "2026-08-31",
            "val": 3.4,
            "announcement_datetime": 1789129800,
            "announcement_datetime_local": "2026-09-11T08:30:00-04:00",
            "source_url": "https://www.bls.gov/news.release/archives/cpi_09112026.htm",
        }
    ],
}


def _tool_call_config(arguments: dict[str, object], *, tool_name: str) -> TestConfig:
    return TestConfig(
        ModelResponse(
            tool_calls=ToolCallsEvent([
                ToolCallEvent(arguments=json.dumps(arguments), name=tool_name),
            ]),
        ),
        "done",
    )


async def _call(toolkit: FXMacroDataToolkit, tool_name: str, arguments: dict[str, Any]) -> Any:
    config = TrackingConfig(_tool_call_config(arguments, tool_name=tool_name))
    agent = Agent("a", config=config, tools=[toolkit])
    await agent.ask("go")
    event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
    return event.results[0].result


def _client_kwargs(api_key: str | None = "test-key") -> dict[str, Any]:
    return FXMacroDataToolkit(api_key=api_key)._client_kwargs()


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schemas(self, context: Context) -> None:
        toolkit = FXMacroDataToolkit()

        schemas = list(await toolkit.schemas(context))

        assert all(isinstance(schema, FunctionToolSchema) for schema in schemas)
        assert [schema.function.name for schema in schemas] == [
            "fxmacrodata_catalogue",
            "fxmacrodata_announcements",
            "fxmacrodata_calendar",
            "fxmacrodata_forex",
        ]
        by_name = {schema.function.name: schema.function.parameters for schema in schemas}
        assert by_name["fxmacrodata_announcements"]["required"] == ["currency", "indicator"]
        assert by_name["fxmacrodata_forex"]["required"] == ["base", "quote"]
        assert "ctx" not in by_name["fxmacrodata_announcements"]["properties"]

    async def test_custom_name_and_description(self, context: Context) -> None:
        custom = FXMacroDataToolkit().calendar(name="release_dates", description="Upcoming releases.")

        [schema] = list(await custom.schemas(context))

        assert schema.function.name == "release_dates"
        assert schema.function.description == "Upcoming releases."


class TestApiKey:
    def test_keyless_sends_no_key_header(self) -> None:
        assert "X-API-Key" not in _client_kwargs(None)["headers"]
        assert "X-API-Key" not in _client_kwargs("   ")["headers"]

    def test_key_is_stripped(self) -> None:
        assert _client_kwargs("  test-key\n")["headers"]["X-API-Key"] == "test-key"

    @pytest.mark.parametrize("bad", ["secret key", "secret\tkey", "secretékey", "secret\x00key"])
    def test_malformed_key_is_rejected_without_echoing_it(self, bad: str) -> None:
        with pytest.raises(ValueError) as exc:
            FXMacroDataToolkit(api_key=bad)
        assert "secret" not in str(exc.value)

    def test_redirects_are_not_followed(self) -> None:
        assert _client_kwargs()["follow_redirects"] is False


@pytest.mark.asyncio
class TestAnnouncements:
    @respx.mock
    async def test_returns_payload_unchanged(self) -> None:
        route = respx.get(f"{BASE_URL}/v1/announcements/usd/inflation").mock(
            return_value=httpx.Response(200, json=ANNOUNCEMENTS)
        )

        result = await _call(
            FXMacroDataToolkit(api_key="test-key"),
            "fxmacrodata_announcements",
            {"currency": "USD", "indicator": "Inflation", "start_date": "2026-01-01", "rows": 1},
        )

        assert result.parts[0] == DataInput(
            FXMacroDataResponse(endpoint="/v1/announcements/usd/inflation", data=ANNOUNCEMENTS)
        )
        request = route.calls.last.request
        assert dict(request.url.params) == {"start_date": "2026-01-01", "limit": "1"}
        assert request.headers["X-API-Key"] == "test-key"
        assert request.headers["User-Agent"] == "ag2-fxmacrodata-extension"

    @respx.mock
    async def test_limit_default_applies_when_model_omits_rows(self) -> None:
        route = respx.get(f"{BASE_URL}/v1/announcements/usd/policy_rate").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        toolkit = FXMacroDataToolkit()
        config = TrackingConfig(
            _tool_call_config({"currency": "usd", "indicator": "policy_rate"}, tool_name="fxmacrodata_announcements")
        )
        agent = Agent("a", config=config, tools=[toolkit.announcements(limit=3)])

        await agent.ask("go")

        assert dict(route.calls.last.request.url.params) == {"limit": "3"}
        assert "X-API-Key" not in route.calls.last.request.headers


@pytest.mark.asyncio
class TestOtherEndpoints:
    @respx.mock
    async def test_catalogue(self) -> None:
        payload = {"gdp": {"name": "GDP", "unit": "USD bn", "source": "BEA"}}
        respx.get(f"{BASE_URL}/v1/data_catalogue/jpy").mock(return_value=httpx.Response(200, json=payload))

        result = await _call(FXMacroDataToolkit(), "fxmacrodata_catalogue", {"currency": "JPY"})

        assert result.parts[0] == DataInput(FXMacroDataResponse(endpoint="/v1/data_catalogue/jpy", data=payload))

    @respx.mock
    async def test_calendar_filters_by_indicator(self) -> None:
        payload = {"currency": "USD", "data": [{"release": "inflation", "name": "Inflation (CPI)"}]}
        route = respx.get(f"{BASE_URL}/v1/calendar/usd").mock(return_value=httpx.Response(200, json=payload))

        await _call(FXMacroDataToolkit(), "fxmacrodata_calendar", {"currency": "usd", "indicator": "inflation"})

        assert dict(route.calls.last.request.url.params) == {"indicator": "inflation"}

    @respx.mock
    async def test_forex(self) -> None:
        payload = {"base": "EUR", "quote": "USD", "data": [{"date": "2026-10-02", "val": 1.1712}]}
        route = respx.get(f"{BASE_URL}/v1/forex/eur/usd").mock(return_value=httpx.Response(200, json=payload))

        result = await _call(FXMacroDataToolkit(api_key="k"), "fxmacrodata_forex", {"base": "eur", "quote": "usd"})

        assert result.parts[0] == DataInput(FXMacroDataResponse(endpoint="/v1/forex/eur/usd", data=payload))
        assert route.calls.last.request.url.params.get("limit") is None


@pytest.mark.asyncio
class TestValidation:
    async def test_bad_currency_never_reaches_the_api(self) -> None:
        with respx.mock(assert_all_called=False) as mock:
            route = mock.get(url__startswith=BASE_URL)
            with pytest.raises(ValueError, match="three-letter currency code"):
                await _call(FXMacroDataToolkit(), "fxmacrodata_calendar", {"currency": "us dollar"})
            assert not route.called

    async def test_indicator_slug_cannot_change_the_path(self) -> None:
        with respx.mock(assert_all_called=False) as mock:
            route = mock.get(url__startswith=BASE_URL)
            with pytest.raises(ValueError, match="slug"):
                await _call(
                    FXMacroDataToolkit(), "fxmacrodata_announcements", {"currency": "usd", "indicator": "../forex/eur"}
                )
            assert not route.called


@pytest.mark.asyncio
class TestErrors:
    @respx.mock
    async def test_redirect_is_refused_and_not_followed(self) -> None:
        respx.get(f"{BASE_URL}/v1/calendar/usd").mock(
            return_value=httpx.Response(302, headers={"Location": "http://elsewhere.example/steal"})
        )
        other = respx.get("http://elsewhere.example/steal").mock(return_value=httpx.Response(200, json={"data": []}))

        with pytest.raises(FXMacroDataError, match="redirect"):
            await _get(_client_kwargs(), "/v1/calendar/usd", {})
        assert not other.called

    @respx.mock
    async def test_http_error_carries_api_detail(self) -> None:
        respx.get(f"{BASE_URL}/v1/forex/eur/usd").mock(
            return_value=httpx.Response(401, json={"detail": "This endpoint requires an API key."})
        )

        with pytest.raises(FXMacroDataError, match="HTTP 401.*requires an API key"):
            await _get(_client_kwargs(None), "/v1/forex/eur/usd", {})

    @respx.mock
    async def test_error_body_with_http_200_is_an_error(self) -> None:
        respx.get(f"{BASE_URL}/v1/calendar/usd").mock(return_value=httpx.Response(200, json={"detail": "Quota"}))

        with pytest.raises(FXMacroDataError, match="Quota"):
            await _get(_client_kwargs(), "/v1/calendar/usd", {})

    @respx.mock
    async def test_non_json_body_is_an_error(self) -> None:
        respx.get(f"{BASE_URL}/v1/calendar/usd").mock(return_value=httpx.Response(200, text="<html>"))

        with pytest.raises(FXMacroDataError, match="non-JSON"):
            await _get(_client_kwargs(), "/v1/calendar/usd", {})

    @pytest.mark.parametrize(
        "path,payload",
        [
            ("/v1/calendar/usd", []),
            ("/v1/calendar/usd", {"currency": "USD"}),
            ("/v1/calendar/usd", {"data": "x"}),
            ("/v1/forex/eur/usd", {"data": [1.17]}),
            ("/v1/data_catalogue/usd", ["gdp"]),
        ],
    )
    @respx.mock
    async def test_malformed_payloads_are_errors(self, path: str, payload: Any) -> None:
        respx.get(f"{BASE_URL}{path}").mock(return_value=httpx.Response(200, json=payload))

        with pytest.raises(FXMacroDataError, match="Unexpected response"):
            await _get(_client_kwargs(), path, {})

    @respx.mock
    async def test_key_never_appears_in_error_text(self) -> None:
        respx.get(f"{BASE_URL}/v1/announcements/eur/inflation").mock(
            return_value=httpx.Response(403, json={"detail": "No EUR access on this plan."})
        )

        with pytest.raises(FXMacroDataError) as exc:
            await _get(_client_kwargs("secret-key-123"), "/v1/announcements/eur/inflation", {})
        assert "secret-key-123" not in str(exc.value)
