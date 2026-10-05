# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""FXMacroData extension for AG2.

FXMacroData serves official-source macroeconomic releases (CPI, GDP, payrolls,
policy rates, bond yields) with their publication timestamps, release calendars
and FX rates for 22 currencies through one REST API. This extension wraps four
read-only endpoints as agent tools.

Maintainer: fxmacrodata
Docs: https://docs.ag2.ai/docs/user-guide/extensions/tools/finance/fxmacrodata/
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any

import httpx
from pydantic import Field

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool

_USER_AGENT = "ag2-fxmacrodata-extension"

# Printable ASCII with no spaces. Checked up front so a malformed key fails with a
# message that does not contain it, instead of failing inside httpx header encoding.
_API_KEY_RE = re.compile(r"^[\x21-\x7e]+$")
_CURRENCY_RE = re.compile(r"^[A-Za-z]{3}$")
_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


class FXMacroDataError(RuntimeError):
    """Raised when FXMacroData answers with an error, a redirect, or a payload of the wrong shape."""


@dataclass(slots=True)
class FXMacroDataResponse:
    """One FXMacroData response, kept as the API published it."""

    endpoint: str
    data: dict[str, Any] = field(default_factory=dict)


def _currency(value: str, name: str) -> str:
    code = value.strip()
    if not _CURRENCY_RE.match(code):
        raise ValueError(f"{name} must be a three-letter currency code such as 'usd', got {value!r}")
    return code.lower()


def _slug(value: str) -> str:
    slug = value.strip().lower()
    if not _SLUG_RE.match(slug):
        raise ValueError(f"indicator must be a slug such as 'inflation' or 'policy_rate', got {value!r}")
    return slug


def _check_shape(path: str, raw: Any) -> dict[str, Any]:
    """Reject error bodies served with HTTP 200 and payloads that are not the documented shape."""
    if not isinstance(raw, dict):
        raise FXMacroDataError(f"Unexpected response from {path}: expected a JSON object")
    if ("detail" in raw or "error" in raw) and "data" not in raw:
        raise FXMacroDataError(f"FXMacroData error from {path}: {raw.get('detail') or raw.get('error')}")
    if not path.startswith("/v1/data_catalogue/"):
        rows = raw.get("data")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise FXMacroDataError(f"Unexpected response from {path}: expected a 'data' list of objects")
    return raw


async def _get(client_kwargs: dict[str, Any], path: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run one GET against the FXMacroData REST API.

    Redirects are not followed, so the ``X-API-Key`` header is never sent to another
    host or scheme.

    Raises:
        FXMacroDataError: On a redirect, a non-2xx status, a non-JSON body, or a payload
            of the wrong shape. The message carries the API's ``detail`` text.
    """
    async with httpx.AsyncClient(**client_kwargs) as client:
        response = await client.get(path, params={key: value for key, value in params.items() if value is not None})

    if response.is_redirect:
        raise FXMacroDataError(
            f"FXMacroData answered {path} with a redirect (HTTP {response.status_code}); not followed"
        )
    try:
        raw = response.json()
    except ValueError:
        raw = None
    if response.is_error:
        detail = raw.get("detail") if isinstance(raw, dict) else None
        raise FXMacroDataError(
            f"FXMacroData returned HTTP {response.status_code} for {path}: {detail or response.reason_phrase}"
        )
    if raw is None:
        raise FXMacroDataError(f"FXMacroData returned a non-JSON body for {path}")
    return _check_shape(path, raw)


class FXMacroDataToolkit(Toolkit):
    """Toolkit for official macroeconomic data, release calendars and FX rates from FXMacroData.

    Passing the toolkit to an agent registers ``fxmacrodata_catalogue``,
    ``fxmacrodata_announcements``, ``fxmacrodata_calendar`` and ``fxmacrodata_forex``.
    To use a subset, call the factory methods and pass the returned tools::

        toolkit = FXMacroDataToolkit(api_key=...)

        agent = Agent("a", config=config, tools=[toolkit])
        agent = Agent("a", config=config, tools=[toolkit.announcements(limit=5), toolkit.calendar()])

    The API key is optional. USD releases from the last 90 days, the USD release
    calendar and the data catalogue answer without one; other currencies, longer
    history and FX rates need a key.
    """

    __slots__ = ("_api_key", "_base_url", "_timeout")

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://api.fxmacrodata.com",
        timeout: float = 30.0,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        """Build the toolkit and its four default tools.

        Args:
            api_key: FXMacroData API key, sent as the ``X-API-Key`` header. Surrounding
                whitespace is stripped; ``None`` or an empty string runs keyless.
            base_url: API root. Trailing slashes are stripped.
            timeout: Per-request timeout in seconds.
            middleware: Middleware applied to every tool in the toolkit.

        Raises:
            ValueError: If *api_key* contains whitespace or non-printable characters.
                The key itself is not included in the message.
        """
        key = (api_key or "").strip()
        if key and not _API_KEY_RE.match(key):
            raise ValueError("api_key contains whitespace or non-printable characters")
        self._api_key = key or None
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

        super().__init__(
            self.catalogue(),
            self.announcements(),
            self.calendar(),
            self.forex(),
            name="fxmacrodata_toolkit",
            middleware=middleware,
        )

    def catalogue(
        self,
        *,
        name: str = "fxmacrodata_catalogue",
        description: str = (
            "List the macroeconomic indicators FXMacroData publishes for a currency, keyed by slug, "
            "with name, unit and source. Call this first when you do not know an indicator slug."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the data catalogue tool (``GET /v1/data_catalogue/{currency}``)."""
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def fxmacrodata_catalogue(
            currency: Annotated[str, Field(description="Three-letter currency code, such as 'usd' or 'eur'.")],
        ) -> ToolResult:
            """List the indicators available for a currency."""
            path = f"/v1/data_catalogue/{_currency(currency, 'currency')}"
            return ToolResult(FXMacroDataResponse(endpoint=path, data=await _get(client_kwargs, path, {})))

        return fxmacrodata_catalogue

    def announcements(
        self,
        *,
        limit: int | Variable | None = None,
        name: str = "fxmacrodata_announcements",
        description: str = (
            "Fetch released values for one indicator, newest first. Each row has the period it covers, "
            "the value, when it was published and the official source link."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the indicator history tool (``GET /v1/announcements/{currency}/{indicator}``).

        Args:
            limit: Default number of rows (1-100) when the model does not pass one.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.
        """
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def fxmacrodata_announcements(
            currency: Annotated[str, Field(description="Three-letter currency code, such as 'usd'.")],
            indicator: Annotated[str, Field(description="Indicator slug, such as 'inflation' or 'policy_rate'.")],
            ctx: Context,
            start_date: Annotated[str | None, Field(description="Optional start date, YYYY-MM-DD.")] = None,
            end_date: Annotated[str | None, Field(description="Optional end date, YYYY-MM-DD.")] = None,
            rows: Annotated[int | None, Field(description="Number of rows to return (1-100).", ge=1, le=100)] = None,
        ) -> ToolResult:
            """Fetch released values for one indicator."""
            path = f"/v1/announcements/{_currency(currency, 'currency')}/{_slug(indicator)}"
            params = {
                "start_date": start_date,
                "end_date": end_date,
                "limit": rows if rows is not None else resolve_variable(limit, ctx, param_name="limit"),
            }
            return ToolResult(FXMacroDataResponse(endpoint=path, data=await _get(client_kwargs, path, params)))

        return fxmacrodata_announcements

    def calendar(
        self,
        *,
        name: str = "fxmacrodata_calendar",
        description: str = (
            "Fetch scheduled economic releases for a currency, with times in UTC and in the publisher's "
            "local time. Optionally filter to one indicator."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the release calendar tool (``GET /v1/calendar/{currency}``)."""
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def fxmacrodata_calendar(
            currency: Annotated[str, Field(description="Three-letter currency code, such as 'usd'.")],
            indicator: Annotated[str | None, Field(description="Optional indicator slug filter.")] = None,
            start_date: Annotated[str | None, Field(description="Optional start date, YYYY-MM-DD.")] = None,
            end_date: Annotated[str | None, Field(description="Optional end date, YYYY-MM-DD.")] = None,
        ) -> ToolResult:
            """Fetch scheduled releases for a currency."""
            path = f"/v1/calendar/{_currency(currency, 'currency')}"
            params = {
                "indicator": _slug(indicator) if indicator else None,
                "start_date": start_date,
                "end_date": end_date,
            }
            return ToolResult(FXMacroDataResponse(endpoint=path, data=await _get(client_kwargs, path, params)))

        return fxmacrodata_calendar

    def forex(
        self,
        *,
        limit: int | Variable | None = None,
        name: str = "fxmacrodata_forex",
        description: str = "Fetch daily FX rates for a currency pair, such as EUR/USD. Needs an FXMacroData API key.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the FX rates tool (``GET /v1/forex/{base}/{quote}``).

        Args:
            limit: Default number of rows (1-100) when the model does not pass one.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.
        """
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def fxmacrodata_forex(
            base: Annotated[str, Field(description="Base currency, such as 'eur'.")],
            quote: Annotated[str, Field(description="Quote currency, such as 'usd'.")],
            ctx: Context,
            start_date: Annotated[str | None, Field(description="Optional start date, YYYY-MM-DD.")] = None,
            end_date: Annotated[str | None, Field(description="Optional end date, YYYY-MM-DD.")] = None,
            rows: Annotated[int | None, Field(description="Number of rows to return (1-100).", ge=1, le=100)] = None,
        ) -> ToolResult:
            """Fetch daily FX rates for a pair."""
            path = f"/v1/forex/{_currency(base, 'base')}/{_currency(quote, 'quote')}"
            params = {
                "start_date": start_date,
                "end_date": end_date,
                "limit": rows if rows is not None else resolve_variable(limit, ctx, param_name="limit"),
            }
            return ToolResult(FXMacroDataResponse(endpoint=path, data=await _get(client_kwargs, path, params)))

        return fxmacrodata_forex

    def _client_kwargs(self) -> dict[str, Any]:
        """Snapshot the connection settings as ``httpx.AsyncClient`` keyword arguments."""
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        return {
            "base_url": self._base_url,
            "headers": headers,
            "timeout": self._timeout,
            "follow_redirects": False,
        }
