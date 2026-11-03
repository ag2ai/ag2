# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from collections.abc import Iterator
from unittest.mock import patch

import httpx
import pytest
import respx

from ag2 import Agent, Context, DataInput
from ag2.events import ToolCallEvent, ToolErrorEvent, ToolResultEvent, ToolResultsEvent
from ag2.extensions.tools import AdanosToolkit
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.final.function_tool import FunctionToolSchema


def _payload(source: str, operation: str) -> dict | list:
    identity = "symbol" if source == "reddit_crypto" else "ticker"
    if operation == "trending":
        if source == "polymarket_stocks":
            return [
                {
                    "ticker": "AAPL",
                    "buzz_score": None,
                    "trend": "stable",
                    "trade_count": 0,
                    "market_count": 0,
                    "current_market_count": 0,
                    "bullish_pct": 0,
                    "bearish_pct": 0,
                    "total_liquidity": 0,
                }
            ]
        row = {
            identity: "BTC" if source == "reddit_crypto" else "AAPL",
            "buzz_score": 0,
            "trend": "stable",
            "mentions": 0,
            "sentiment_score": 0,
            "bullish_pct": 0,
            "bearish_pct": 0,
        }
        if source == "news_stocks":
            row["source_count"] = 0
        elif source == "x_stocks":
            row.update(unique_tweets=0, total_upvotes=0)
        else:
            row.update(unique_posts=0, subreddit_count=0, total_upvotes=0)
        return [row]
    if operation == "asset":
        return {
            identity: "BTC" if source == "reddit_crypto" else "AAPL",
            "found": False,
            "buzz_score": None,
            "sentiment_score": None,
            "daily_trend": [],
            "period_days": 7,
        }
    if operation == "compare":
        return {"period_days": 7, "tokens" if source == "reddit_crypto" else "stocks": []}
    if operation == "search":
        return {"query": "apple", "count": 0, "period_days": 7, "results": []}
    data = {
        "buzz_score": None,
        "sentiment_score": None,
        "active_tickers": 0,
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
        "bullish_pct": 0,
        "bearish_pct": 0,
        "drivers": [],
        "trend_history": [],
    }
    if source == "polymarket_stocks":
        data.update(trade_count=0, market_count=0, current_market_count=0, total_liquidity=0)
    elif source == "news_stocks":
        data.update(mentions=0, unique_articles=0, source_count=0)
    elif source == "x_stocks":
        data.update(mentions=0, unique_tweets=0, unique_authors=0, total_upvotes=0)
    else:
        data.update(mentions=0, unique_posts=0, subreddit_count=0, total_upvotes=0)
    return data


async def _invoke(arguments: dict, toolkit: AdanosToolkit | None = None) -> ToolResultEvent:
    config = TrackingConfig(
        TestConfig(
            ToolCallEvent(name="adanos_query", arguments=json.dumps(arguments)),
            "done",
            raise_tool_errors=False,
        )
    )
    agent = Agent("researcher", config=config, tools=[toolkit or AdanosToolkit(api_key="test-secret")])
    reply = await agent.ask("Get source-labelled research evidence.")
    assert reply.body == "done"
    event = config.mock.call_args_list[1].args[0]
    assert isinstance(event, ToolResultsEvent)
    [schema] = config.calls[0].tools
    assert isinstance(schema, FunctionToolSchema)
    assert schema.function.name == "adanos_query"
    assert "api_key" not in str(schema.function.parameters)
    assert "test-secret" not in str(config.calls)
    [result] = event.results
    return result


@pytest.fixture
def clients() -> Iterator[list[httpx.AsyncClient]]:
    instances = []

    class ObservedClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    with patch("httpx.AsyncClient", ObservedClient):
        yield instances
    assert all(client.is_closed for client in instances)


def _assert_error(result: ToolResultEvent) -> None:
    assert isinstance(result, ToolErrorEvent)
    rendered = str(result.result)
    assert "test-secret" not in rendered
    assert "private-provider-body" not in rendered
    assert "requested_period" not in rendered


@pytest.mark.asyncio
async def test_native_schema_has_one_tool_and_no_credentials(context: Context) -> None:
    toolkit = AdanosToolkit(api_key="test-secret")
    [schema] = list(await toolkit.schemas(context))
    assert isinstance(schema, FunctionToolSchema)
    assert schema.function.name == "adanos_query"
    assert set(schema.function.parameters["properties"]) == {
        "source",
        "operation",
        "asset",
        "assets",
        "query",
        "from_date",
        "to_date",
        "limit",
    }
    assert schema.function.parameters["required"] == ["source", "operation"]
    assert "test-secret" not in str(schema)
    assert schema.function.parameters["properties"]["source"]["enum"] == [
        "reddit_stocks",
        "x_stocks",
        "news_stocks",
        "polymarket_stocks",
        "reddit_crypto",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,base",
    [
        ("reddit_stocks", "/reddit/stocks/v1"),
        ("x_stocks", "/x/stocks/v1"),
        ("news_stocks", "/news/stocks/v1"),
        ("polymarket_stocks", "/polymarket/stocks/v1"),
        ("reddit_crypto", "/reddit/crypto/v1"),
    ],
)
@pytest.mark.parametrize("operation", ["trending", "asset", "compare", "search", "market_sentiment"])
@respx.mock
async def test_all_25_native_routes(source: str, base: str, operation: str, clients: list) -> None:
    arguments = {"source": source, "operation": operation}
    params = {}
    path = operation.replace("_", "-")
    if operation == "asset":
        arguments["asset"] = "BTC" if source == "reddit_crypto" else "AAPL"
        path = "token/BTC" if source == "reddit_crypto" else "stock/AAPL"
    elif operation == "compare":
        arguments["assets"] = ["BTC", "ETH"] if source == "reddit_crypto" else ["AAPL", "MSFT"]
        params["symbols" if source == "reddit_crypto" else "tickers"] = (
            "BTC,ETH" if source == "reddit_crypto" else "AAPL,MSFT"
        )
    elif operation == "search":
        arguments.update(query="apple", limit=50)
        params.update(q="apple", limit="50")
    if operation != "search":
        arguments.update(from_date="2026-10-01", to_date="2026-10-07")
        params.update({"from": "2026-10-01", "to": "2026-10-07"})
    if operation == "trending":
        arguments["limit"] = 100
        params["limit"] = "100"
    data = _payload(source, operation)
    route = respx.get(f"https://api.adanos.com{base}/{path}").respond(200, json=data)

    result = await _invoke(arguments)

    assert type(result) is ToolResultEvent
    assert result.result.parts == [
        DataInput({
            "source": source,
            "operation": operation,
            "requested_period": {"from": arguments.get("from_date"), "to": arguments.get("to_date")},
            "data": data,
        })
    ]
    assert route.call_count == 1
    request = route.calls.last.request
    assert dict(request.url.params) == params
    assert request.headers["X-API-Key"] == "test-secret"
    assert request.method == "GET"
    [client] = clients
    assert client.is_closed
    assert not client.follow_redirects
    assert client.timeout == httpx.Timeout(30.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,asset,path",
    [
        ("reddit_stocks", "$BRK.A", "/reddit/stocks/v1/stock/%24BRK.A"),
        ("news_stocks", "abx.DE", "/news/stocks/v1/stock/abx.DE"),
        ("reddit_crypto", "$BTC", "/reddit/crypto/v1/token/%24BTC"),
    ],
)
@respx.mock
async def test_path_encoding_and_identity_are_not_normalized(source: str, asset: str, path: str) -> None:
    route = respx.get(f"https://api.adanos.com{path}").respond(200, json=_payload(source, "asset"))
    result = await _invoke({"source": source, "operation": "asset", "asset": asset})
    assert type(result) is ToolResultEvent
    assert route.calls.last.request.url.raw_path == path.encode()


@pytest.mark.asyncio
@respx.mock
async def test_news_compare_preserves_company_keys() -> None:
    route = respx.get("https://api.adanos.com/news/stocks/v1/compare").respond(
        200,
        json={"period_days": 7, "stocks": []},
    )
    await _invoke({"source": "news_stocks", "operation": "compare", "assets": ["BRK", "abx.DE"]})
    assert dict(route.calls.last.request.url.params) == {"tickers": "BRK,abx.DE"}


@pytest.mark.asyncio
@respx.mock
async def test_empty_trending_and_provider_defaults() -> None:
    route = respx.get("https://api.adanos.com/reddit/stocks/v1/trending").respond(200, json=[])
    result = await _invoke({"source": "reddit_stocks", "operation": "trending"})
    assert result.result.parts == [
        DataInput({
            "source": "reddit_stocks",
            "operation": "trending",
            "requested_period": {"from": None, "to": None},
            "data": [],
        })
    ]
    assert not route.calls.last.request.url.params


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"source": "https://evil.example"},
        {"operation": "mentions"},
        {"operation": "asset"},
        {"operation": "asset", "asset": "../stats"},
        {"operation": "asset", "asset": "%2e%2e"},
        {"operation": "asset", "asset": "AAPL?x=1"},
        {"operation": "asset", "asset": "https://evil.example"},
        {"operation": "asset", "asset": ".."},
        {"operation": "asset", "asset": "AAPL\\x"},
        {"operation": "asset", "asset": "AAPL\n"},
        {"operation": "compare"},
        {"operation": "compare", "assets": []},
        {"operation": "compare", "assets": ["AAPL"] * 11},
        {"operation": "compare", "assets": ["AAPL,MSFT"]},
        {"operation": "compare", "assets": ["../AAPL"]},
        {"operation": "search"},
        {"operation": "search", "query": " $a "},
        {"operation": "search", "query": "A$"},
        {"operation": "search", "query": "$A$"},
        {"operation": "search", "query": "apple", "limit": 51},
        {"source": "news_stocks", "operation": "search", "query": "x" * 101},
        {"source": "polymarket_stocks", "operation": "search", "query": "x" * 51},
        {"operation": "search", "query": "apple", "from_date": "2026-10-01"},
        {"from_date": "2026-02-30"},
        {"from_date": "20261001"},
        {"from_date": "2026-10-01T00:00:00Z"},
        {"from_date": "2026-10-07", "to_date": "2026-10-01"},
        {"limit": 0},
        {"limit": 101},
        {"limit": True},
        {"limit": 1.5},
        {"operation": "market_sentiment", "limit": 2},
        {"asset": "AAPL"},
        {"assets": ["AAPL"]},
        {"query": "apple"},
    ],
)
@respx.mock
async def test_invalid_arguments_never_send_http(overrides: dict, clients: list) -> None:
    result = await _invoke({"source": "reddit_stocks", "operation": "trending", **overrides})
    _assert_error(result)
    assert not respx.calls
    assert not clients


@pytest.mark.asyncio
@pytest.mark.parametrize("key", [None, "", " ", "bad\nkey", "bad\u2603key"])
@respx.mock
async def test_unconfigured_or_unsafe_key_never_sends_http(key: str | None, monkeypatch, clients: list) -> None:
    monkeypatch.delenv("ADANOS_API_KEY", raising=False)
    result = await _invoke({"source": "reddit_stocks", "operation": "trending"}, AdanosToolkit(api_key=key))
    _assert_error(result)
    assert not respx.calls
    assert not clients


@pytest.mark.asyncio
@respx.mock
async def test_environment_key_is_private(monkeypatch) -> None:
    monkeypatch.setenv("ADANOS_API_KEY", "env-secret")
    route = respx.get("https://api.adanos.com/reddit/stocks/v1/trending").respond(200, json=[])
    result = await _invoke({"source": "reddit_stocks", "operation": "trending"}, AdanosToolkit())
    assert route.calls.last.request.headers["X-API-Key"] == "env-secret"
    assert "env-secret" not in str(result)
    assert os.environ["ADANOS_API_KEY"] == "env-secret"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 302, 307, 308, 401, 403, 404, 422, 429, 500, 502, 503])
@respx.mock
async def test_http_failures_are_sanitized_native_errors(status: int, clients: list) -> None:
    route = respx.get("https://api.adanos.com/reddit/stocks/v1/trending").respond(
        status,
        json={"detail": "private-provider-body test-secret"},
        headers={"Location": "https://evil.example/steal", "Retry-After": "1"},
    )
    result = await _invoke({"source": "reddit_stocks", "operation": "trending"})
    _assert_error(result)
    assert str(status) in str(result.error)
    assert route.call_count == 1
    assert len(respx.calls) == 1
    assert clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,data",
    [
        ("trending", {}),
        ("trending", ["private-provider-body"]),
        ("trending", [{"detail": "private-provider-body"}]),
        ("asset", []),
        ("asset", {"detail": "private-provider-body"}),
        ("asset", {"ticker": "AAPL"}),
        ("asset", {"ticker": "AAPL", "found": "wrong"}),
        ("compare", {"period_days": 7, "tokens": []}),
        ("compare", {"period_days": 7, "stocks": "wrong"}),
        ("search", []),
        ("search", {"query": "apple", "results": [], "count": "wrong", "period_days": 7}),
        ("market_sentiment", []),
        ("market_sentiment", {"detail": "private-provider-body"}),
        ("trending", None),
        ("trending", 0),
        ("trending", False),
    ],
)
@respx.mock
async def test_wrong_success_shape_is_not_evidence(operation: str, data: object, clients: list) -> None:
    respx.get().respond(200, content=json.dumps(data))
    arguments = {"source": "reddit_stocks", "operation": operation}
    if operation == "asset":
        arguments["asset"] = "AAPL"
    elif operation == "compare":
        arguments["assets"] = ["AAPL", "MSFT"]
    elif operation == "search":
        arguments["query"] = "apple"
    _assert_error(await _invoke(arguments))
    assert clients[0].is_closed


@pytest.mark.asyncio
@respx.mock
async def test_malformed_json_is_sanitized(clients: list) -> None:
    respx.get().respond(200, content="private-provider-body test-secret")
    _assert_error(await _invoke({"source": "reddit_stocks", "operation": "trending"}))
    assert clients[0].is_closed


@pytest.mark.asyncio
@respx.mock
async def test_x_asset_found_is_optional() -> None:
    respx.get("https://api.adanos.com/x/stocks/v1/stock/AAPL").respond(200, json={"ticker": "AAPL"})
    result = await _invoke({"source": "x_stocks", "operation": "asset", "asset": "AAPL"})
    assert type(result) is ToolResultEvent
    assert result.result.parts == [
        DataInput({
            "source": "x_stocks",
            "operation": "asset",
            "requested_period": {"from": None, "to": None},
            "data": {"ticker": "AAPL"},
        })
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.ConnectError])
@respx.mock
async def test_transport_errors_are_sanitized_without_retry(error: type[httpx.RequestError], clients: list) -> None:
    route = respx.get().mock(side_effect=error("private-provider-body test-secret"))
    _assert_error(await _invoke({"source": "reddit_stocks", "operation": "trending"}))
    assert route.call_count == 1
    assert clients[0].is_closed


@pytest.mark.asyncio
@respx.mock
async def test_cancelled_native_invocation_closes_client(clients: list) -> None:
    started = asyncio.Event()

    async def blocked(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("request must be cancelled")

    respx.get().mock(side_effect=blocked)
    task = asyncio.create_task(_invoke({"source": "reddit_stocks", "operation": "trending"}))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(clients) == 1
    assert clients[0].is_closed


@pytest.mark.asyncio
@respx.mock
async def test_concurrent_agents_have_local_clients(clients: list, monkeypatch) -> None:
    monkeypatch.setenv("ADANOS_API_KEY", "unchanged")
    both_started = asyncio.Event()
    requests = []

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=2)
        return httpx.Response(200, json=[])

    respx.get().mock(side_effect=respond)
    toolkit = AdanosToolkit(api_key="test-secret")
    results = await asyncio.gather(
        _invoke({"source": "reddit_stocks", "operation": "trending"}, toolkit),
        _invoke({"source": "reddit_crypto", "operation": "trending"}, toolkit),
    )
    assert all(type(result) is ToolResultEvent for result in results)
    assert len(clients) == 2
    assert clients[0] is not clients[1]
    assert all(client.is_closed for client in clients)
    assert {str(request.url) for request in requests} == {
        "https://api.adanos.com/reddit/stocks/v1/trending",
        "https://api.adanos.com/reddit/crypto/v1/trending",
    }
    assert os.environ["ADANOS_API_KEY"] == "unchanged"
