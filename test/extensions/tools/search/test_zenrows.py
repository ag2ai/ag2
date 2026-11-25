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
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultsEvent
from ag2.extensions.tools.search.zenrows import (
    ZenRowsData,
    ZenRowsPage,
    ZenRowsRefusal,
    ZenRowsToolkit,
)
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.final.function_tool import FunctionToolSchema

ZENROWS_URL = "https://api.zenrows.com/v1/"
PAGE_URL = "https://example.com/"
SENTINEL_KEY = "zr-sentinel-key-do-not-leak"  # pragma: allowlist secret
REQUEST_ID = "6af5ab9d3ab68237a8c038c8376a8538"

# Recorded against the live API with `url=https://example.com/`.
MARKDOWN_BODY = (
    "This domain is for use in documentation examples without needing permission. "
    "This is not a service; avoid relying on it for testing and monitoring purposes."
)
HTML_BODY = "<!doctype html><html lang=en><head><title>Example Domain</title></head><body><p>Example</p></body></html>"
CSS_BODY: dict[str, Any] = {"links": "", "title": "Example Domain"}
OUTPUTS_BODY: dict[str, Any] = {"headings": [], "links": []}
EXTRACT_BODY: dict[str, Any] = {
    "html": "<html><body>the whole page source, 61 KB of it in one recorded capture</body></html>",
    "parsed": {"article": {"title": "Example Domain"}, "breadcrumb": ["Home"]},
}

SUCCESS_HEADERS = {"X-Request-Id": REQUEST_ID, "Zr-Final-Url": PAGE_URL}

AUTH002: dict[str, Any] = {
    "code": "AUTH002",
    "detail": "Invalid API key.",
    "instance": "/v1",
    "status": 401,
    "title": "Invalid apikey provided (AUTH002)",
    "type": "https://docs.zenrows.com/api-error-codes#AUTH002",
}
REQS004: dict[str, Any] = {
    "code": "REQS004",
    "detail": "The provided URL is not valid. Make sure the URL is properly formatted and try again.",
    "instance": "/v1",
    "status": 400,
    "title": "Invalid params provided (REQS004)",
    "type": "https://docs.zenrows.com/api-error-codes#REQS004",
}
REQS007_PAGE_NOT_RECOGNISED: dict[str, Any] = {
    "code": "REQS007",
    "detail": (
        "Extract serves scrapingcourse.com, but did not recognise this page as one it can extract. "
        "The request was not charged."
    ),
    "instance": "/v1",
    "reason": "page_not_recognised",
    "status": 403,
    "title": "Extract did not recognise this page on scrapingcourse.com (REQS007)",
    "type": "https://docs.zenrows.com/api-error-codes#REQS007",
}
RESP003: dict[str, Any] = {
    "code": "RESP003",
    "detail": "The request failed because we couldn't parse the contents. Please retry by removing 'extract'.",
    "instance": "/v1",
    "status": 502,
    "title": "Could not parse response as the retrieved content is not html (RESP003)",
    "type": "https://docs.zenrows.com/api-error-codes#RESP003",
}


def _tool_call_config(
    arguments: dict[str, Any],
    *,
    tool_name: str,
    final_reply: str = "done",
) -> TestConfig:
    return TestConfig(
        ModelResponse(tool_calls=ToolCallsEvent([ToolCallEvent(arguments=json.dumps(arguments), name=tool_name)])),
        final_reply,
    )


def _result(config: TrackingConfig) -> Any:
    tool_results_event: ToolResultsEvent = config.mock.call_args_list[1].args[0]
    return tool_results_event.results[0].result.parts[0]


def _scrape(**arguments: Any) -> TestConfig:
    return _tool_call_config({"url": PAGE_URL, **arguments}, tool_name="zenrows_scrape")


def _extract(**arguments: Any) -> TestConfig:
    return _tool_call_config({"url": PAGE_URL, **arguments}, tool_name="zenrows_extract")


def _text_route(body: str = MARKDOWN_BODY) -> respx.Route:
    return respx.get(ZENROWS_URL).mock(return_value=httpx.Response(200, text=body, headers=SUCCESS_HEADERS))


def _json_route(body: Any) -> respx.Route:
    return respx.get(ZENROWS_URL).mock(return_value=httpx.Response(200, json=body, headers=SUCCESS_HEADERS))


def _problem_route(status: int, body: Any, headers: dict[str, str] | None = None) -> respx.Route:
    return respx.get(ZENROWS_URL).mock(return_value=httpx.Response(status, json=body, headers=headers or {}))


@pytest.mark.asyncio
class TestSchema:
    async def test_default_schemas(self, context: Context) -> None:
        toolkit = ZenRowsToolkit(api_key="test")

        schemas = list(await toolkit.schemas(context))

        assert [s.function.name for s in schemas] == ["zenrows_scrape", "zenrows_extract"]
        assert all(isinstance(s, FunctionToolSchema) for s in schemas)

    async def test_scrape_schema_takes_url_and_optional_css_extractor(self, context: Context) -> None:
        toolkit = ZenRowsToolkit(api_key="test")

        [scrape_schema, _] = list(await toolkit.schemas(context))

        assert scrape_schema.function.parameters == IsPartialDict({
            "required": ["url"],
            "properties": IsPartialDict({
                "url": IsPartialDict({"type": "string"}),
                "css_extractor": IsPartialDict({"default": None}),
            }),
        })
        assert "ctx" not in scrape_schema.function.parameters["properties"]

    async def test_extract_schema_takes_url_only(self, context: Context) -> None:
        toolkit = ZenRowsToolkit(api_key="test")

        [_, extract_schema] = list(await toolkit.schemas(context))

        assert extract_schema.function.parameters == IsPartialDict({"required": ["url"]})
        assert list(extract_schema.function.parameters["properties"]) == ["url"]

    async def test_custom_scrape_name_and_description(self, context: Context) -> None:
        toolkit = ZenRowsToolkit(api_key="test")

        [schema] = list(await toolkit.scrape(name="read_page", description="Read a page.").schemas(context))

        assert schema.function.name == "read_page"
        assert schema.function.description == "Read a page."

    async def test_custom_extract_name_and_description(self, context: Context) -> None:
        toolkit = ZenRowsToolkit(api_key="test")

        [schema] = list(await toolkit.extract(name="read_data", description="Read page data.").schemas(context))

        assert schema.function.name == "read_data"
        assert schema.function.description == "Read page data."

    async def test_toolkit_builds_without_api_key(self, monkeypatch: pytest.MonkeyPatch, context: Context) -> None:
        monkeypatch.delenv("ZENROWS_API_KEY", raising=False)

        toolkit = ZenRowsToolkit()

        assert len(list(await toolkit.schemas(context))) == 2


@pytest.mark.asyncio
class TestScrapeExecution:
    @respx.mock
    async def test_returns_markdown_by_default(self) -> None:
        route = _text_route()
        toolkit = ZenRowsToolkit(api_key="test")
        config = TrackingConfig(_scrape())

        await Agent("a", config=config, tools=[toolkit]).ask("read")

        assert dict(route.calls.last.request.url.params) == {
            "url": PAGE_URL,
            "response_type": "markdown",
            "mode": "auto",
        }
        assert _result(config) == DataInput(
            ZenRowsPage(
                url=PAGE_URL,
                format="markdown",
                content=MARKDOWN_BODY,
                truncated=False,
                request_id=REQUEST_ID,
            )
        )

    @respx.mock
    async def test_sends_key_as_header_and_stamps_ag2(self) -> None:
        route = _text_route()
        toolkit = ZenRowsToolkit(api_key=SENTINEL_KEY)

        await Agent("a", config=_scrape(), tools=[toolkit]).ask("read")

        request = route.calls.last.request
        assert request.headers["X-API-Key"] == SENTINEL_KEY
        assert request.headers["User-Agent"] == "ag2-zenrows-extension"
        assert "apikey" not in dict(request.url.params)
        assert SENTINEL_KEY not in str(request.url)

    @respx.mock
    async def test_reads_key_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZENROWS_API_KEY", "env-key")
        route = _text_route()

        await Agent("a", config=_scrape(), tools=[ZenRowsToolkit()]).ask("read")

        assert route.calls.last.request.headers["X-API-Key"] == "env-key"

    @respx.mock
    async def test_plaintext_response_type(self) -> None:
        route = _text_route()
        toolkit = ZenRowsToolkit(api_key="test", response_type="plaintext")
        config = TrackingConfig(_scrape())

        await Agent("a", config=config, tools=[toolkit]).ask("read")

        assert dict(route.calls.last.request.url.params)["response_type"] == "plaintext"
        assert _result(config).data.format == "plaintext"

    @respx.mock
    async def test_html_is_the_api_default_and_sends_no_response_type(self) -> None:
        route = _text_route(HTML_BODY)
        toolkit = ZenRowsToolkit(api_key="test", response_type="html")
        config = TrackingConfig(_scrape())

        await Agent("a", config=config, tools=[toolkit]).ask("read")

        assert dict(route.calls.last.request.url.params) == {"url": PAGE_URL, "mode": "auto"}
        assert _result(config) == DataInput(
            ZenRowsPage(url=PAGE_URL, format="html", content=HTML_BODY, request_id=REQUEST_ID)
        )

    @respx.mock
    async def test_css_extractor_returns_named_fields(self) -> None:
        route = _json_route(CSS_BODY)
        toolkit = ZenRowsToolkit(api_key="test")
        config = TrackingConfig(_scrape(css_extractor={"title": "h1", "links": "a @href"}))

        await Agent("a", config=config, tools=[toolkit]).ask("read")

        params = dict(route.calls.last.request.url.params)
        assert json.loads(params["css_extractor"]) == {"title": "h1", "links": "a @href"}
        assert "response_type" not in params
        assert _result(config) == DataInput(ZenRowsData(url=PAGE_URL, data=CSS_BODY, request_id=REQUEST_ID))

    @respx.mock
    async def test_outputs_returns_named_fields(self) -> None:
        route = _json_route(OUTPUTS_BODY)
        toolkit = ZenRowsToolkit(api_key="test", outputs=["headings", "links"])
        config = TrackingConfig(_scrape())

        await Agent("a", config=config, tools=[toolkit]).ask("read")

        params = dict(route.calls.last.request.url.params)
        assert params["outputs"] == "headings,links"
        assert "response_type" not in params
        assert _result(config) == DataInput(ZenRowsData(url=PAGE_URL, data=OUTPUTS_BODY, request_id=REQUEST_ID))

    @respx.mock
    async def test_outputs_given_as_one_string(self) -> None:
        route = _json_route(OUTPUTS_BODY)
        toolkit = ZenRowsToolkit(api_key="test", outputs="headings")

        await Agent("a", config=_scrape(), tools=[toolkit]).ask("read")

        assert dict(route.calls.last.request.url.params)["outputs"] == "headings"

    @respx.mock
    async def test_css_extractor_and_outputs_are_both_sent(self) -> None:
        route = _json_route(CSS_BODY)
        toolkit = ZenRowsToolkit(api_key="test", outputs=["links"])

        await Agent("a", config=_scrape(css_extractor={"title": "h1"}), tools=[toolkit]).ask("read")

        params = dict(route.calls.last.request.url.params)
        assert params["outputs"] == "links"
        assert json.loads(params["css_extractor"]) == {"title": "h1"}

    @respx.mock
    async def test_falls_back_to_the_requested_url_without_a_final_url_header(self) -> None:
        respx.get(ZENROWS_URL).mock(return_value=httpx.Response(200, text=MARKDOWN_BODY))
        toolkit = ZenRowsToolkit(api_key="test")
        config = TrackingConfig(_scrape())

        await Agent("a", config=config, tools=[toolkit]).ask("read")

        assert _result(config) == DataInput(
            ZenRowsPage(url=PAGE_URL, format="markdown", content=MARKDOWN_BODY, request_id=None)
        )

    @respx.mock
    async def test_non_object_json_raises(self) -> None:
        _json_route([1, 2])
        toolkit = ZenRowsToolkit(api_key="test")

        with pytest.raises(ValueError, match="not an object"):
            await Agent("a", config=_scrape(css_extractor={"title": "h1"}), tools=[toolkit]).ask("read")


@pytest.mark.asyncio
class TestAdaptiveStealthMode:
    @respx.mock
    @pytest.mark.parametrize(
        ("options", "expected"),
        [
            ({}, {"mode": "auto"}),
            ({"proxy_country": "us"}, {"proxy_country": "us", "mode": "auto"}),
            ({"js_render": True}, {"js_render": "true"}),
            ({"js_render": False}, {"js_render": "false"}),
            ({"premium_proxy": True}, {"premium_proxy": "true"}),
            (
                {"js_render": True, "premium_proxy": True, "proxy_country": "us"},
                {"js_render": "true", "premium_proxy": "true", "proxy_country": "us"},
            ),
        ],
    )
    async def test_mode_auto_is_sent_until_a_flag_is_set(
        self, options: dict[str, Any], expected: dict[str, str]
    ) -> None:
        route = _text_route()
        toolkit = ZenRowsToolkit(api_key="test", **options)

        await Agent("a", config=_scrape(), tools=[toolkit]).ask("read")

        assert dict(route.calls.last.request.url.params) == {
            "url": PAGE_URL,
            "response_type": "markdown",
            **expected,
        }

    @respx.mock
    async def test_extract_also_rides_adaptive_stealth_mode(self) -> None:
        route = _json_route(EXTRACT_BODY)
        toolkit = ZenRowsToolkit(api_key="test")

        await Agent("a", config=_extract(), tools=[toolkit]).ask("extract")

        assert dict(route.calls.last.request.url.params) == {
            "url": PAGE_URL,
            "extract": "auto",
            "mode": "auto",
        }


@pytest.mark.asyncio
class TestValidation:
    async def test_response_type_and_outputs_rejected_when_building_the_tool(self) -> None:
        toolkit = ZenRowsToolkit(api_key="test")

        with pytest.raises(ValueError, match="response_type and outputs cannot both be set"):
            toolkit.scrape(response_type="markdown", outputs=["links"])

    @respx.mock
    async def test_response_type_and_outputs_from_variables_rejected_before_the_request(self) -> None:
        route = _text_route()
        toolkit = ZenRowsToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_scrape(),
            tools=[toolkit.scrape(response_type=Variable(), outputs=Variable())],
            variables={"response_type": "markdown", "outputs": ["links"]},
        )

        with pytest.raises(ValueError, match="response_type and outputs cannot both be set"):
            await agent.ask("read")
        assert not route.called

    @pytest.mark.parametrize("options", [{"js_render": True}, {"premium_proxy": False}])
    async def test_proxy_country_without_premium_proxy_rejected_when_building_the_tool(
        self, options: dict[str, Any]
    ) -> None:
        toolkit = ZenRowsToolkit(api_key="test")

        with pytest.raises(ValueError, match="proxy_country requires premium_proxy"):
            toolkit.scrape(proxy_country="us", **options)

    async def test_extract_validates_the_same_rule(self) -> None:
        toolkit = ZenRowsToolkit(api_key="test")

        with pytest.raises(ValueError, match="proxy_country requires premium_proxy"):
            toolkit.extract(proxy_country="us", js_render=True)

    @respx.mock
    async def test_proxy_country_without_premium_proxy_from_variables_rejected_before_the_request(self) -> None:
        route = _text_route()
        toolkit = ZenRowsToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_scrape(),
            tools=[toolkit.scrape(proxy_country="us", premium_proxy=Variable())],
            variables={"premium_proxy": False},
        )

        with pytest.raises(ValueError, match="proxy_country requires premium_proxy"):
            await agent.ask("read")
        assert not route.called

    async def test_deferred_premium_proxy_is_not_rejected_at_build_time(self) -> None:
        toolkit = ZenRowsToolkit(api_key="test")

        assert toolkit.scrape(proxy_country="us", premium_proxy=Variable()) is not None

    @respx.mock
    async def test_deferred_premium_proxy_resolving_true_is_served(self) -> None:
        route = _text_route()
        toolkit = ZenRowsToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_scrape(),
            tools=[toolkit.scrape(proxy_country="us", premium_proxy=Variable())],
            variables={"premium_proxy": True},
        )
        await agent.ask("read")

        assert dict(route.calls.last.request.url.params) == {
            "url": PAGE_URL,
            "response_type": "markdown",
            "proxy_country": "us",
            "premium_proxy": "true",
        }

    async def test_proxy_country_alone_is_allowed(self) -> None:
        toolkit = ZenRowsToolkit(api_key="test")

        assert toolkit.scrape(proxy_country="us") is not None


@pytest.mark.asyncio
class TestExtractExecution:
    @respx.mock
    async def test_returns_parsed_and_never_the_page_source(self) -> None:
        _json_route(EXTRACT_BODY)
        toolkit = ZenRowsToolkit(api_key="test")
        config = TrackingConfig(_extract())

        await Agent("a", config=config, tools=[toolkit]).ask("extract")

        part = _result(config)
        assert part == DataInput(ZenRowsData(url=PAGE_URL, data=EXTRACT_BODY["parsed"], request_id=REQUEST_ID))
        assert "html" not in part.data.data
        assert EXTRACT_BODY["html"] not in json.dumps(part.data.data)

    @respx.mock
    async def test_missing_parsed_key_returns_no_data(self) -> None:
        _json_route({"html": "<html></html>"})
        toolkit = ZenRowsToolkit(api_key="test")
        config = TrackingConfig(_extract())

        await Agent("a", config=config, tools=[toolkit]).ask("extract")

        assert _result(config) == DataInput(ZenRowsData(url=PAGE_URL, data={}, request_id=REQUEST_ID))


@pytest.mark.asyncio
class TestRefusals:
    @respx.mock
    @pytest.mark.parametrize("tool_name", ["zenrows_scrape", "zenrows_extract"])
    async def test_non_http_url_is_refused_before_any_request(self, tool_name: str) -> None:
        route = _text_route()
        toolkit = ZenRowsToolkit(api_key="test")
        config = TrackingConfig(_tool_call_config({"url": "ftp://example.com/x"}, tool_name=tool_name))

        await Agent("a", config=config, tools=[toolkit]).ask("read")

        part = _result(config)
        assert part.data.url == "ftp://example.com/x"
        assert part.data.reason == "unsupported_url_scheme"
        assert part.data.next_step == "Call this tool again with an http or https URL."
        assert not route.called

    @respx.mock
    @pytest.mark.parametrize(
        ("problem", "status", "reason"),
        [
            ({"code": "REQS007", "reason": "domain_not_prepared", "title": "x"}, 403, "domain_not_prepared"),
            ({"code": "REQS007", "reason": "page_not_recognised", "title": "x"}, 403, "page_not_recognised"),
            ({"code": "REQS007", "reason": "domain_not_enabled", "title": "x"}, 403, "domain_not_enabled"),
            (
                {"code": "REQS007", "title": "Extract is not prepared for the requested domain example.com"},
                403,
                "domain_not_prepared",
            ),
            (
                {"code": "REQS007", "title": "Extract did not recognise this page on example.com (REQS007)"},
                403,
                "page_not_recognised",
            ),
            (
                {"code": "REQS007", "title": "Extract is not enabled for the requested domain example.com"},
                403,
                "domain_not_enabled",
            ),
            ({"code": "REQS007", "reason": "blocked_document", "title": "Blocked"}, 403, "not_extractable"),
            ({"code": "REQS007"}, 403, "not_extractable"),
            ({"code": "AUTH010", "title": "Extract not enabled"}, 402, "domain_not_enabled"),
            ({"code": "no_data_extracted", "title": "No data"}, 422, "no_data_extracted"),
            (RESP003, 502, "not_extractable"),
        ],
    )
    async def test_each_extract_verdict_comes_back_as_a_refusal(
        self, problem: dict[str, Any], status: int, reason: str
    ) -> None:
        _problem_route(status, problem, {"X-Request-Id": REQUEST_ID})
        toolkit = ZenRowsToolkit(api_key="test")
        config = TrackingConfig(_extract())

        await Agent("a", config=config, tools=[toolkit]).ask("extract")

        part = _result(config)
        assert isinstance(part.data, ZenRowsRefusal)
        assert part.data.reason == reason
        assert part.data.request_id == REQUEST_ID
        assert part.data.next_step
        assert "retry" not in part.data.next_step.lower()

    @respx.mock
    async def test_refusal_carries_the_api_detail(self) -> None:
        _problem_route(403, REQS007_PAGE_NOT_RECOGNISED, {"X-Request-Id": REQUEST_ID})
        toolkit = ZenRowsToolkit(api_key="test")
        config = TrackingConfig(_extract())

        await Agent("a", config=config, tools=[toolkit]).ask("extract")

        assert _result(config).data.detail == REQS007_PAGE_NOT_RECOGNISED["detail"]

    @respx.mock
    async def test_a_non_verdict_error_still_raises_on_extract(self) -> None:
        _problem_route(401, AUTH002)
        toolkit = ZenRowsToolkit(api_key="test")

        with pytest.raises(httpx.HTTPStatusError, match="AUTH002"):
            await Agent("a", config=_extract(), tools=[toolkit]).ask("extract")

    @respx.mock
    async def test_scrape_does_not_translate_extract_verdicts(self) -> None:
        _problem_route(403, REQS007_PAGE_NOT_RECOGNISED)
        toolkit = ZenRowsToolkit(api_key="test")

        with pytest.raises(httpx.HTTPStatusError, match="REQS007"):
            await Agent("a", config=_scrape(), tools=[toolkit]).ask("read")


@pytest.mark.asyncio
class TestErrors:
    @respx.mock
    async def test_error_message_carries_code_detail_request_id_and_retry_after(self) -> None:
        _problem_route(400, REQS004, {"X-Request-Id": REQUEST_ID, "Retry-After": "30"})
        toolkit = ZenRowsToolkit(api_key="test")

        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await Agent("a", config=_scrape(), tools=[toolkit]).ask("read")

        message = str(excinfo.value)
        assert "HTTP 400" in message
        assert "(REQS004)" in message
        assert REQS004["detail"] in message
        assert f"[request {REQUEST_ID}]" in message
        assert "(Retry-After: 30)" in message

    @respx.mock
    async def test_error_without_a_problem_body(self) -> None:
        respx.get(ZENROWS_URL).mock(return_value=httpx.Response(500, text="upstream exploded"))
        toolkit = ZenRowsToolkit(api_key="test")

        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await Agent("a", config=_scrape(), tools=[toolkit]).ask("read")

        assert str(excinfo.value).startswith("Zenrows API returned HTTP 500")
        assert "[request" not in str(excinfo.value)

    @respx.mock
    async def test_error_with_a_non_object_problem_body(self) -> None:
        _problem_route(500, ["nope"])
        toolkit = ZenRowsToolkit(api_key="test")

        with pytest.raises(httpx.HTTPStatusError, match="HTTP 500"):
            await Agent("a", config=_scrape(), tools=[toolkit]).ask("read")

    @respx.mock
    async def test_error_without_a_key_names_the_environment_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ZENROWS_API_KEY", raising=False)
        _problem_route(401, {"code": "AUTH001", "detail": "No apikey provided."})

        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await Agent("a", config=_scrape(), tools=[ZenRowsToolkit()]).ask("read")

        assert "ZENROWS_API_KEY" in str(excinfo.value)


@pytest.mark.asyncio
class TestKeyHygiene:
    @respx.mock
    @pytest.mark.parametrize(
        ("status", "body"),
        [
            (400, REQS004),
            (401, AUTH002),
            (429, {"code": "REQS001", "detail": "Too many requests."}),
            (500, {"code": "RESP001", "detail": "Upstream failure."}),
        ],
    )
    async def test_raised_errors_never_carry_the_key(self, status: int, body: dict[str, Any]) -> None:
        route = _problem_route(status, body, {"X-Request-Id": REQUEST_ID, "Retry-After": "5"})
        toolkit = ZenRowsToolkit(api_key=SENTINEL_KEY)

        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await Agent("a", config=_scrape(), tools=[toolkit]).ask("read")

        assert SENTINEL_KEY not in str(excinfo.value)
        assert all(SENTINEL_KEY not in str(call.request.url) for call in route.calls)

    @respx.mock
    async def test_extract_refusal_path_never_carries_the_key(self) -> None:
        route = _problem_route(403, REQS007_PAGE_NOT_RECOGNISED)
        toolkit = ZenRowsToolkit(api_key=SENTINEL_KEY)
        config = TrackingConfig(_extract())

        await Agent("a", config=config, tools=[toolkit]).ask("extract")

        assert SENTINEL_KEY not in repr(_result(config))
        assert all(SENTINEL_KEY not in str(call.request.url) for call in route.calls)

    @respx.mock
    async def test_transport_failure_never_carries_the_key(self) -> None:
        route = respx.get(ZENROWS_URL).mock(side_effect=httpx.ConnectError("connection refused"))
        toolkit = ZenRowsToolkit(api_key=SENTINEL_KEY)

        with pytest.raises(httpx.ConnectError) as excinfo:
            await Agent("a", config=_scrape(), tools=[toolkit]).ask("read")

        assert SENTINEL_KEY not in str(excinfo.value)
        assert all(SENTINEL_KEY not in str(call.request.url) for call in route.calls)

    @respx.mock
    async def test_validation_failure_never_carries_the_key(self) -> None:
        route = _text_route()
        toolkit = ZenRowsToolkit(api_key=SENTINEL_KEY)

        agent = Agent(
            "a",
            config=_scrape(),
            tools=[toolkit.scrape(proxy_country="us", premium_proxy=Variable())],
            variables={"premium_proxy": False},
        )

        with pytest.raises(ValueError) as excinfo:
            await agent.ask("read")

        assert SENTINEL_KEY not in str(excinfo.value)
        assert not route.called

    @respx.mock
    async def test_redirects_are_not_followed(self) -> None:
        route = respx.get(ZENROWS_URL).mock(
            return_value=httpx.Response(302, headers={"Location": "https://evil.example/steal"})
        )
        toolkit = ZenRowsToolkit(api_key=SENTINEL_KEY)

        with pytest.raises(httpx.HTTPStatusError, match="HTTP 302"):
            await Agent("a", config=_scrape(), tools=[toolkit]).ask("read")

        assert [str(call.request.url).startswith(ZENROWS_URL) for call in route.calls] == [True]


@pytest.mark.asyncio
class TestMaxCharacters:
    @respx.mock
    @pytest.mark.parametrize(
        ("max_characters", "expected", "truncated"),
        [
            (None, MARKDOWN_BODY, False),
            (len(MARKDOWN_BODY), MARKDOWN_BODY, False),
            (len(MARKDOWN_BODY) - 1, MARKDOWN_BODY[:-1], True),
            (10, MARKDOWN_BODY[:10], True),
        ],
    )
    async def test_text_is_bounded(self, max_characters: int | None, expected: str, truncated: bool) -> None:
        _text_route()
        toolkit = ZenRowsToolkit(api_key="test", max_characters=max_characters)
        config = TrackingConfig(_scrape())

        await Agent("a", config=config, tools=[toolkit]).ask("read")

        assert _result(config) == DataInput(
            ZenRowsPage(
                url=PAGE_URL,
                format="markdown",
                content=expected,
                truncated=truncated,
                request_id=REQUEST_ID,
            )
        )

    @respx.mock
    @pytest.mark.parametrize(
        ("max_characters", "expected", "truncated"),
        [
            (None, CSS_BODY, False),
            (len(json.dumps(CSS_BODY)), CSS_BODY, False),
            (len(json.dumps(CSS_BODY)) - 1, {"links": ""}, True),
            (1, {}, True),
        ],
    )
    async def test_structured_data_is_cut_by_whole_entries(
        self, max_characters: int | None, expected: dict[str, Any], truncated: bool
    ) -> None:
        _json_route(CSS_BODY)
        toolkit = ZenRowsToolkit(api_key="test", max_characters=max_characters)
        config = TrackingConfig(_scrape(css_extractor={"title": "h1"}))

        await Agent("a", config=config, tools=[toolkit]).ask("read")

        part = _result(config)
        assert part == DataInput(ZenRowsData(url=PAGE_URL, data=expected, truncated=truncated, request_id=REQUEST_ID))
        assert json.loads(json.dumps(part.data.data)) == expected

    @respx.mock
    async def test_extract_data_is_cut_by_whole_entries(self) -> None:
        _json_route(EXTRACT_BODY)
        parsed: dict[str, Any] = EXTRACT_BODY["parsed"]
        toolkit = ZenRowsToolkit(api_key="test", max_characters=len(json.dumps({"article": parsed["article"]})))
        config = TrackingConfig(_extract())

        await Agent("a", config=config, tools=[toolkit]).ask("extract")

        assert _result(config) == DataInput(
            ZenRowsData(
                url=PAGE_URL,
                data={"article": parsed["article"]},
                truncated=True,
                request_id=REQUEST_ID,
            )
        )

    @respx.mock
    async def test_a_refusal_is_never_bounded(self) -> None:
        _problem_route(403, REQS007_PAGE_NOT_RECOGNISED, {"X-Request-Id": REQUEST_ID})
        toolkit = ZenRowsToolkit(api_key="test", max_characters=5)
        config = TrackingConfig(_extract())

        await Agent("a", config=config, tools=[toolkit]).ask("extract")

        part = _result(config)
        assert part.data.detail == REQS007_PAGE_NOT_RECOGNISED["detail"]
        assert part.data.request_id == REQUEST_ID


@pytest.mark.asyncio
class TestVariables:
    @respx.mock
    async def test_options_resolve_from_agent_variables(self) -> None:
        route = _text_route()
        toolkit = ZenRowsToolkit(api_key="test")
        scrape_tool = toolkit.scrape(
            response_type=Variable("page_format"),
            js_render=Variable(),
            max_characters=Variable("cap"),
        )

        config = TrackingConfig(_scrape())
        agent = Agent(
            "a",
            config=config,
            tools=[scrape_tool],
            variables={"page_format": "plaintext", "js_render": True, "cap": 4},
        )
        await agent.ask("read")

        assert dict(route.calls.last.request.url.params) == {
            "url": PAGE_URL,
            "response_type": "plaintext",
            "js_render": "true",
        }
        assert _result(config) == DataInput(
            ZenRowsPage(
                url=PAGE_URL,
                format="plaintext",
                content=MARKDOWN_BODY[:4],
                truncated=True,
                request_id=REQUEST_ID,
            )
        )

    @respx.mock
    async def test_extract_options_resolve_from_agent_variables(self) -> None:
        route = _json_route(EXTRACT_BODY)
        toolkit = ZenRowsToolkit(api_key="test")

        agent = Agent(
            "a",
            config=_extract(),
            tools=[toolkit.extract(premium_proxy=Variable(), proxy_country=Variable())],
            variables={"premium_proxy": True, "proxy_country": "de"},
        )
        await agent.ask("extract")

        assert dict(route.calls.last.request.url.params) == {
            "url": PAGE_URL,
            "extract": "auto",
            "premium_proxy": "true",
            "proxy_country": "de",
        }

    @respx.mock
    async def test_missing_variable_raises(self) -> None:
        route = _text_route()
        toolkit = ZenRowsToolkit(api_key="test")

        agent = Agent("a", config=_scrape(), tools=[toolkit.scrape(response_type=Variable())])

        with pytest.raises(KeyError, match="response_type"):
            await agent.ask("read")
        assert not route.called


@pytest.mark.asyncio
class TestIndividualTools:
    @respx.mock
    async def test_scrape_tool_passed_alone(self) -> None:
        route = _text_route()
        toolkit = ZenRowsToolkit(api_key="test")

        await Agent("a", config=_scrape(), tools=[toolkit.scrape()]).ask("read")

        assert route.called

    @respx.mock
    async def test_extract_tool_passed_alone(self) -> None:
        route = _json_route(EXTRACT_BODY)
        toolkit = ZenRowsToolkit(api_key="test")

        await Agent("a", config=_extract(), tools=[toolkit.extract()]).ask("extract")

        assert route.called
