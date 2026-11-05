# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Adanos market sentiment Extension for source-labelled research evidence.

Supports five research operations across Reddit Stocks, X Stocks, News Stocks,
Polymarket Stocks, and Reddit Crypto using the Adanos API 1.54.1 contract.

Maintainer: alexander-schneider
"""

import os
import re
from collections.abc import Iterable
from datetime import date
from typing import Annotated, Any, Literal, TypeAlias
from urllib.parse import quote

import httpx
from pydantic import Field

from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.final import Toolkit, tool

Source: TypeAlias = Literal["reddit_stocks", "x_stocks", "news_stocks", "polymarket_stocks", "reddit_crypto"]
Operation: TypeAlias = Literal["trending", "asset", "compare", "search", "market_sentiment"]


def _asset_segment(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9$][A-Za-z0-9$.-]*", value):
        raise ValueError("asset must be a single symbol or News company key, not a URL or path")
    return quote(value, safe="")


def _period(from_date: str | None, to_date: str | None) -> dict[str, str]:
    params = {}
    for key, value in (("from", from_date), ("to", to_date)):
        if value is not None:
            try:
                valid = date.fromisoformat(value).isoformat() == value
            except ValueError:
                valid = False
            if not valid:
                raise ValueError("dates must be valid UTC calendar dates in YYYY-MM-DD format")
            params[key] = value
    if from_date is not None and to_date is not None and from_date > to_date:
        raise ValueError("from_date must not be after to_date")
    return params


def _request(
    source: Source,
    operation: Operation,
    asset: str | None,
    assets: list[str] | None,
    query: str | None,
    from_date: str | None,
    to_date: str | None,
    limit: int | None,
) -> tuple[str, dict[str, str | int]]:
    bases = {
        "reddit_stocks": "/reddit/stocks/v1",
        "x_stocks": "/x/stocks/v1",
        "news_stocks": "/news/stocks/v1",
        "polymarket_stocks": "/polymarket/stocks/v1",
        "reddit_crypto": "/reddit/crypto/v1",
    }
    supplied = {
        name
        for name, value in {
            "asset": asset,
            "assets": assets,
            "query": query,
            "from_date": from_date,
            "to_date": to_date,
            "limit": limit,
        }.items()
        if value is not None
    }
    allowed = {
        "trending": {"from_date", "to_date", "limit"},
        "asset": {"from_date", "to_date", "asset"},
        "compare": {"from_date", "to_date", "assets"},
        "search": {"query", "limit"},
        "market_sentiment": {"from_date", "to_date"},
    }
    if supplied - allowed[operation]:
        raise ValueError("parameters are not supported for the selected operation")
    params: dict[str, str | int] = dict(_period(from_date, to_date))
    path = operation.replace("_", "-")
    if operation == "asset":
        if asset is None:
            raise ValueError("asset is required")
        path = f"{'token' if source == 'reddit_crypto' else 'stock'}/{_asset_segment(asset)}"
    elif operation == "compare":
        if not assets or len(assets) > 10:
            raise ValueError("assets must contain 1 to 10 symbols or News company keys")
        for item in assets:
            _asset_segment(item)
        params["symbols" if source == "reddit_crypto" else "tickers"] = ",".join(assets)
    elif operation == "search":
        max_length = {"news_stocks": 100, "polymarket_stocks": 50}.get(source)
        if query is None or len(query.strip().replace("$", "")) < 2:
            raise ValueError("query needs at least two non-cashtag characters after trimming")
        if max_length is not None and len(query) > max_length:
            raise ValueError("query exceeds this source's search length limit")
        params["q"] = query
    if limit is not None:
        if operation == "search" and limit > 50:
            raise ValueError("search limit must be between 1 and 50")
        params["limit"] = limit
    return f"{bases[source]}/{path}", params


def _valid_response(data: Any, source: Source, operation: Operation) -> bool:
    identity = "symbol" if source == "reddit_crypto" else "ticker"
    if operation == "asset":
        return (
            isinstance(data, dict)
            and isinstance(data.get(identity), str)
            and type(data.get("found", True if source == "x_stocks" else None)) is bool
        )
    if operation == "trending":
        rows = data
    else:
        fields_by_operation: dict[str, dict[str, tuple[type, ...]]] = {
            "compare": {"period_days": (int,), "tokens" if source == "reddit_crypto" else "stocks": (list,)},
            "search": {"query": (str,), "count": (int,), "period_days": (int,), "results": (list,)},
            "market_sentiment": {
                "buzz_score": (int, float, type(None)),
                "trade_count" if source == "polymarket_stocks" else "mentions": (int,),
            },
        }
        fields = fields_by_operation[operation]
        if not isinstance(data, dict) or any(
            key not in data or type(data[key]) not in types for key, types in fields.items()
        ):
            return False
        rows = data.get("results", data.get("tokens", data.get("stocks", [])))
    return isinstance(rows, list) and all(isinstance(row, dict) and isinstance(row.get(identity), str) for row in rows)


class AdanosToolkit(Toolkit):
    """Expose one ``adanos_query`` FunctionTool for Adanos research evidence.

    Credentials stay outside the model schema. Each invocation owns its HTTP
    client and makes one request, with no retries or pagination.

    Args:
        api_key: Server-side Adanos API key. If omitted, read ``ADANOS_API_KEY``
            at invocation time. Missing or unsafe keys fail without network I/O.
        middleware: Middleware applied to the tool.
    """

    __slots__ = ("_api_key",)

    def __init__(self, api_key: str | None = None, *, middleware: Iterable[ToolMiddleware] = ()) -> None:
        self._api_key = api_key
        super().__init__(tool(self._query, name="adanos_query"), name="adanos_toolkit", middleware=middleware)

    async def _query(
        self,
        source: Source,
        operation: Operation,
        asset: Annotated[str | None, Field(description="Symbol or News company key, required only for asset.")] = None,
        assets: Annotated[
            list[str] | None, Field(description="1-10 symbols or News company keys, only for compare.")
        ] = None,
        query: Annotated[str | None, Field(description="Search text, required only for search.")] = None,
        from_date: Annotated[
            str | None, Field(description="Inclusive UTC YYYY-MM-DD start. Not supported for search.")
        ] = None,
        to_date: Annotated[
            str | None, Field(description="Inclusive UTC YYYY-MM-DD end. Not supported for search.")
        ] = None,
        limit: Annotated[
            int | None, Field(strict=True, ge=1, le=100, description="Trending 1-100 or search 1-50 only.")
        ] = None,
    ) -> ToolResult:
        """Get source-labelled Adanos attention and sentiment evidence, not prices or trading advice.

        News assets are company keys, not share-class tickers. Preserve keys from
        that source's search results. Search summaries use a fixed seven-day window.
        Omitted dates use provider defaults, today for trending and market_sentiment,
        the last seven days for asset and compare. Null is unavailable data, not zero.
        """
        api_key = self._api_key if self._api_key is not None else os.getenv("ADANOS_API_KEY")
        if api_key is None or not re.fullmatch(r"[!-~]+", api_key):
            raise ValueError("configure a valid server-side ADANOS_API_KEY or constructor api_key")
        path, params = _request(source, operation, asset, assets, query, from_date, to_date, limit)
        try:
            async with httpx.AsyncClient(
                base_url="https://api.adanos.com",
                headers={"X-API-Key": api_key},
                timeout=30.0,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                response = await client.get(path, params=params)
        except httpx.HTTPError:
            raise RuntimeError("Adanos request failed due to a transport error or timeout") from None
        if not response.is_success:
            raise RuntimeError(f"Adanos request failed (HTTP {response.status_code})")
        try:
            data = response.json()
        except ValueError:
            raise RuntimeError("Adanos returned invalid JSON") from None
        if not _valid_response(data, source, operation):
            raise RuntimeError("Adanos returned an unexpected response shape")
        return ToolResult({
            "source": source,
            "operation": operation,
            "requested_period": {"from": from_date, "to": to_date},
            "data": data,
        })
