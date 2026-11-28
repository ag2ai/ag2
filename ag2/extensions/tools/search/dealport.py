# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Dealport US private company, property, and owner intelligence extension for AG2.

Dealport (https://dealport.com) indexes 28M+ US private companies, commercial
buildings, 50-state US property parcels, corporate officers, and verified human
principals backed by primary public filings.

Maintainer: rkmasiello
Docs: https://docs.ag2.ai/docs/user-guide/extensions/tools/search/dealport/
"""

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Any, Literal
from urllib.parse import quote

import httpx
from pydantic import Field

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool

_API_KEY_ENV_VARS = ("DEALPORT_API_KEY", "DP_API_KEY")  # pragma: allowlist secret
_DEFAULT_BASE_URL = "https://app.dealport.com"
_DEFAULT_MAX_CHARACTERS = 10_000
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_BBL_RE = re.compile(r"^[1-5]\d{9}$")
_STATE_RE = re.compile(r"^[A-Za-z]{2}$")

NycBorough = Literal["MN", "BK", "QN", "BX", "SI"]


@dataclass(slots=True)
class DealportResult:
    """Structured outcome from a Dealport endpoint with optional content bounding."""

    endpoint: str
    cost_usd: float | None = None
    matching_count: int | None = None
    next_cursor: str | None = None
    data: Any = None
    truncated: bool = False

    @classmethod
    def from_api(
        cls,
        endpoint: str,
        raw: dict[str, Any],
        max_characters: int | None,
    ) -> "DealportResult":
        """Build a bounded result while preserving counts, cursors, and USD charges."""
        if max_characters is not None and max_characters < 0:
            raise ValueError("max_characters must be non-negative")

        raw_cost = raw.get("cost_usd")
        if not isinstance(raw_cost, (int, float)) or isinstance(raw_cost, bool):
            nested = raw.get("data")
            nested_cost = nested.get("cost_usd") if isinstance(nested, dict) else None
            cost_usd = (
                float(nested_cost)
                if isinstance(nested_cost, (int, float)) and not isinstance(nested_cost, bool)
                else None
            )
        else:
            cost_usd = float(raw_cost)

        raw_count = raw.get("matching_count")
        matching_count = raw_count if isinstance(raw_count, int) and not isinstance(raw_count, bool) else None

        raw_cursor = raw.get("next_cursor")
        next_cursor = raw_cursor if isinstance(raw_cursor, str) and raw_cursor else None

        payload = raw.get("data") if "data" in raw else raw.get("sample", raw)
        truncated = False
        if payload is not None and max_characters is not None:
            serialized = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
            if len(serialized) > max_characters:
                payload = serialized[:max_characters]
                truncated = True

        return cls(
            endpoint=endpoint,
            cost_usd=cost_usd,
            matching_count=matching_count,
            next_cursor=next_cursor,
            data=payload,
            truncated=truncated,
        )


def _resolve_api_key(explicit_key: str | None) -> str | None:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    for env_name in _API_KEY_ENV_VARS:
        val = os.environ.get(env_name, "").strip()
        if val:
            return val
    return None


def _require_api_key(explicit_key: str | None) -> str:
    key = _resolve_api_key(explicit_key)
    if not key:
        raise ValueError("Dealport API key is required for this tool; pass api_key=... or set DEALPORT_API_KEY.")
    return key


def _clean_state(state: str | None) -> str | None:
    if state is None:
        return None
    cleaned = state.strip().upper()
    if not _STATE_RE.match(cleaned):
        raise ValueError("state must be a two-letter US state code such as 'OH' or 'TX'")
    return cleaned


def _clean_record_id(record_id: str) -> str:
    cleaned = record_id.strip()
    if not cleaned or "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        raise ValueError("record_id must be a Dealport company UUID or 10-digit NYC BBL")
    if not (_UUID_RE.match(cleaned) or _BBL_RE.match(cleaned)):
        raise ValueError("record_id must be a Dealport company UUID or 10-digit NYC BBL")
    return quote(cleaned, safe="-")


def _clean_company_uuid(company_id: str) -> str:
    cleaned = company_id.strip()
    if not _UUID_RE.match(cleaned):
        raise ValueError("company_id must be a valid Dealport company UUID")
    return cleaned


def _raise_for_status(response: httpx.Response) -> None:
    """Raise ``httpx.HTTPStatusError`` with Dealport's message without echoing credentials."""
    if response.is_success:
        return
    detail = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict) and isinstance(err.get("message"), str):
                detail = err["message"]
            elif isinstance(err, str):
                detail = err
            elif isinstance(body.get("message"), str):
                detail = body["message"]
    except ValueError:
        detail = ""

    message = f"Dealport API returned HTTP {response.status_code}"
    if detail:
        message += f": {detail}"
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            message += f" (Retry-After: {retry_after})"
    raise httpx.HTTPStatusError(message, request=response.request, response=response)


async def _rest_request(
    client_kwargs: dict[str, Any],
    api_key: str,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    clean_params = {k: v for k, v in params.items() if v is not None} if params is not None else None
    clean_body = {k: v for k, v in json_body.items() if v is not None} if json_body is not None else None

    async with httpx.AsyncClient(**client_kwargs) as client:
        response = await client.request(method, path, params=clean_params, json=clean_body, headers=headers)
        _raise_for_status(response)
        decoded = response.json()
    if not isinstance(decoded, dict):
        raise ValueError("Dealport returned a non-object JSON response")
    return decoded


async def _keyless_mcp_preview(
    client_kwargs: dict[str, Any],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Call keyless ``preview_companies`` over Dealport's Streamable HTTP MCP endpoint."""
    clean_args = {k: v for k, v in arguments.items() if v is not None}
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    rpc_body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "preview_companies", "arguments": clean_args},
    }
    async with httpx.AsyncClient(**client_kwargs) as client:
        response = await client.post("/api/mcp", json=rpc_body, headers=headers)
        _raise_for_status(response)
        raw_text = response.text

    data_line = next((line[6:] for line in raw_text.splitlines() if line.startswith("data: ")), raw_text)
    envelope = json.loads(data_line)
    if not isinstance(envelope, dict):
        raise ValueError("Dealport MCP returned a non-object JSON response")
    if "error" in envelope and isinstance(envelope["error"], dict):
        msg = envelope["error"].get("message", "MCP error")
        raise ValueError(f"Dealport MCP preview failed: {msg}")
    result = envelope.get("result")
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            text = content[0].get("text")
            if isinstance(text, str):
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    parsed.setdefault("cost_usd", 0.0)
                    return parsed
    raise ValueError("Dealport MCP preview returned unexpected content")


class DealportToolkit(Toolkit):
    """Toolkit exposing Dealport US company, NYC building, Record, and owner tools.

    Passing the toolkit to an agent registers five focused tools:
      - ``dealport_preview_companies``: free ($0.00) US company market count and sample (works keyless)
      - ``dealport_search_companies``: search 28M+ cited US private company Records ($0.01/Record, $0.00 30-day repeat)
      - ``dealport_search_buildings``: search NYC commercial and multifamily building Records ($0.01/Record, $0.00 30-day repeat)
      - ``dealport_get_record``: open one cited company UUID or 10-digit NYC BBL Record ($0.01, $0.00 30-day repeat)
      - ``dealport_find_owner``: look up the verified human owner contact for a company UUID ($0.09 hit, $0.00 miss)

    Applications control ``limit`` and ``max_characters``, and may register only
    the individual factory tools they choose to fund. For Dealport's full 47-tool
    MCP surface (including 50-state parcel lookup, LLC officer unmasking, public
    filings, Projects, Playbooks, and double-opt-in outreach via Avery), connect
    ``MCPToolkit`` to ``https://app.dealport.com/api/mcp``.
    """

    __slots__ = ("_api_key", "_base_url", "_limit", "_max_characters", "_proxy", "_timeout", "_verify")

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 60.0,
        proxy: str | None = None,
        verify: bool = True,
        limit: int | Variable | None = None,
        max_characters: int | None = _DEFAULT_MAX_CHARACTERS,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        if max_characters is not None and max_characters < 0:
            raise ValueError("max_characters must be non-negative")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._proxy = proxy
        self._verify = verify
        self._limit = limit
        self._max_characters = max_characters

        super().__init__(
            self.preview_companies(max_characters=max_characters),
            self.search_companies(limit=limit, max_characters=max_characters),
            self.search_buildings(limit=limit, max_characters=max_characters),
            self.get_record(max_characters=max_characters),
            self.find_owner(max_characters=max_characters),
            name="dealport_toolkit",
            middleware=middleware,
        )

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "base_url": self._base_url,
            "timeout": self._timeout,
            "verify": self._verify,
        }
        if self._proxy is not None:
            kwargs["proxy"] = self._proxy
        return kwargs

    def preview_companies(
        self,
        *,
        max_characters: int | None = _DEFAULT_MAX_CHARACTERS,
        name: str = "dealport_preview_companies",
        description: str = (
            "Size a US private company market for $0.00 and return the matching count plus up to 5 sample companies."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the $0.00 US company market sizing preview tool."""
        api_key = self._api_key
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def dealport_preview_companies(
            state: Annotated[
                str | None,
                Field(default=None, description="Two-letter US state code (for example 'OH', 'TX', 'FL')."),
            ] = None,
            naics: Annotated[
                list[str] | None,
                Field(default=None, description="NAICS code prefixes or 6-digit codes (for example ['238220'])."),
            ] = None,
            q: Annotated[
                str | None,
                Field(default=None, description="Company name prefix filter."),
            ] = None,
            min_revenue: Annotated[
                float | None,
                Field(default=None, ge=0, description="Minimum annual revenue in USD."),
            ] = None,
            max_revenue: Annotated[
                float | None,
                Field(default=None, ge=0, description="Maximum annual revenue in USD."),
            ] = None,
            min_employees: Annotated[
                int | None,
                Field(default=None, ge=0, description="Minimum employee headcount."),
            ] = None,
            max_employees: Annotated[
                int | None,
                Field(default=None, ge=0, description="Maximum employee headcount."),
            ] = None,
        ) -> ToolResult:
            """Return matching US private company count and up to 5 sample companies ($0.00)."""
            cleaned_state = _clean_state(state)
            clean_naics = [code.strip() for code in naics if code and code.strip()] if naics else None
            clean_q = q.strip() if q and q.strip() else None

            if not any(
                val is not None
                for val in (cleaned_state, clean_naics, clean_q, min_revenue, max_revenue, min_employees, max_employees)
            ):
                return ToolResult({"error": "Provide at least one filter such as state, naics, or q."})

            body: dict[str, Any] = {
                "state": cleaned_state,
                "naics": clean_naics,
                "q": clean_q,
                "min_revenue": min_revenue,
                "max_revenue": max_revenue,
                "min_employees": min_employees,
                "max_employees": max_employees,
            }
            resolved_key = _resolve_api_key(api_key)
            if resolved_key:
                raw = await _rest_request(
                    client_kwargs,
                    resolved_key,
                    "POST",
                    "/api/v1/companies/preview",
                    json_body=body,
                )
            else:
                raw = await _keyless_mcp_preview(client_kwargs, body)
            return ToolResult(DealportResult.from_api("/api/v1/companies/preview", raw, max_characters))

        return dealport_preview_companies

    def search_companies(
        self,
        *,
        limit: int | Variable | None = None,
        max_characters: int | None = _DEFAULT_MAX_CHARACTERS,
        name: str = "dealport_search_companies",
        description: str = (
            "Search 28M+ cited US private company Records by state, NAICS, name prefix, domain, or size."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the US company search tool ($0.01 per new Record, $0.00 on 30-day repeat)."""
        api_key = self._api_key
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def dealport_search_companies(
            ctx: Context,
            state: Annotated[
                str | None,
                Field(default=None, description="Two-letter US state code (for example 'OH', 'TX', 'FL')."),
            ] = None,
            naics: Annotated[
                str | None,
                Field(
                    default=None, description="Comma-separated NAICS prefixes or 6-digit codes (for example '238220')."
                ),
            ] = None,
            q: Annotated[
                str | None,
                Field(default=None, description="Company name prefix filter."),
            ] = None,
            domain: Annotated[
                str | None,
                Field(default=None, description="Company website domain (for example 'acme.com')."),
            ] = None,
            min_revenue: Annotated[
                float | None,
                Field(default=None, ge=0, description="Minimum annual revenue in USD."),
            ] = None,
            max_revenue: Annotated[
                float | None,
                Field(default=None, ge=0, description="Maximum annual revenue in USD."),
            ] = None,
            cursor: Annotated[
                str | None,
                Field(default=None, description="Pagination cursor from a previous search result."),
            ] = None,
        ) -> ToolResult:
            """Search cited US private company Records."""
            cleaned_state = _clean_state(state)
            clean_naics = naics.strip() if naics and naics.strip() else None
            clean_q = q.strip() if q and q.strip() else None
            clean_domain = domain.strip() if domain and domain.strip() else None

            if not any(
                val is not None for val in (cleaned_state, clean_naics, clean_q, clean_domain, min_revenue, max_revenue)
            ):
                return ToolResult({"error": "Provide at least one filter such as state, naics, q, or domain."})

            resolved_limit = resolve_variable(limit, ctx, param_name="limit")
            if resolved_limit is not None and (resolved_limit < 1 or resolved_limit > 50):
                raise ValueError("limit must be between 1 and 50")

            key = _require_api_key(api_key)
            raw = await _rest_request(
                client_kwargs,
                key,
                "GET",
                "/api/v1/companies",
                params={
                    "state": cleaned_state,
                    "naics": clean_naics,
                    "q": clean_q,
                    "domain": clean_domain,
                    "min_revenue": min_revenue,
                    "max_revenue": max_revenue,
                    "limit": resolved_limit,
                    "cursor": cursor.strip() if cursor and cursor.strip() else None,
                },
            )
            return ToolResult(DealportResult.from_api("/api/v1/companies", raw, max_characters))

        return dealport_search_companies

    def search_buildings(
        self,
        *,
        limit: int | Variable | None = None,
        max_characters: int | None = _DEFAULT_MAX_CHARACTERS,
        name: str = "dealport_search_buildings",
        description: str = (
            "Search NYC commercial and multifamily building Records by borough, address, unit count, or assessed value."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the NYC commercial building search tool ($0.01 per new Record, $0.00 on 30-day repeat)."""
        api_key = self._api_key
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def dealport_search_buildings(
            ctx: Context,
            borough: Annotated[
                NycBorough | None,
                Field(default=None, description="NYC borough code: MN, BK, QN, BX, or SI."),
            ] = None,
            q: Annotated[
                str | None,
                Field(default=None, description="Street address or building query."),
            ] = None,
            min_units: Annotated[
                int | None,
                Field(default=None, ge=0, description="Minimum total residential and commercial units."),
            ] = None,
            max_units: Annotated[
                int | None,
                Field(default=None, ge=0, description="Maximum total residential and commercial units."),
            ] = None,
            cursor: Annotated[
                str | None,
                Field(default=None, description="Pagination cursor from a previous building search."),
            ] = None,
        ) -> ToolResult:
            """Search cited NYC commercial and multifamily building Records."""
            clean_q = q.strip() if q and q.strip() else None
            if borough is None and clean_q is None and min_units is None and max_units is None:
                return ToolResult({"error": "Provide at least one filter such as borough, q, or min_units."})

            resolved_limit = resolve_variable(limit, ctx, param_name="limit")
            if resolved_limit is not None and (resolved_limit < 1 or resolved_limit > 50):
                raise ValueError("limit must be between 1 and 50")

            key = _require_api_key(api_key)
            raw = await _rest_request(
                client_kwargs,
                key,
                "GET",
                "/api/v1/buildings",
                params={
                    "borough": borough,
                    "q": clean_q,
                    "min_units": min_units,
                    "max_units": max_units,
                    "limit": resolved_limit,
                    "cursor": cursor.strip() if cursor and cursor.strip() else None,
                },
            )
            return ToolResult(DealportResult.from_api("/api/v1/buildings", raw, max_characters))

        return dealport_search_buildings

    def get_record(
        self,
        *,
        max_characters: int | None = _DEFAULT_MAX_CHARACTERS,
        name: str = "dealport_get_record",
        description: str = ("Open a full cited Dealport company Record (UUID) or NYC building Record (10-digit BBL)."),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the single-Record retrieval tool ($0.01, $0.00 on 30-day repeat)."""
        api_key = self._api_key
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def dealport_get_record(
            record_id: Annotated[
                str,
                Field(description="Dealport company UUID or 10-digit NYC borough-block-lot (BBL) identifier."),
            ],
        ) -> ToolResult:
            """Open one cited company or building Record."""
            try:
                safe_segment = _clean_record_id(record_id)
            except ValueError as exc:
                return ToolResult({"error": str(exc)})

            key = _require_api_key(api_key)
            raw = await _rest_request(
                client_kwargs,
                key,
                "GET",
                f"/api/v1/records/{safe_segment}",
            )
            return ToolResult(DealportResult.from_api("/api/v1/records/{id}", raw, max_characters))

        return dealport_get_record

    def find_owner(
        self,
        *,
        max_characters: int | None = _DEFAULT_MAX_CHARACTERS,
        name: str = "dealport_find_owner",
        description: str = (
            "Resolve the verified human owner or named principal contact behind a Dealport company UUID."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the verified owner lookup tool ($0.09 on hit, $0.00 on miss or 30-day repeat)."""
        api_key = self._api_key
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def dealport_find_owner(
            company_id: Annotated[
                str,
                Field(description="Dealport company UUID from search or preview results."),
            ],
            person: Annotated[
                str | None,
                Field(default=None, description="Optional full name of a specific principal listed on the Record."),
            ] = None,
        ) -> ToolResult:
            """Look up verified human owner contact details for a company UUID."""
            try:
                safe_uuid = _clean_company_uuid(company_id)
            except ValueError as exc:
                return ToolResult({"error": str(exc)})

            key = _require_api_key(api_key)
            clean_person = person.strip() if person and person.strip() else None
            raw = await _rest_request(
                client_kwargs,
                key,
                "POST",
                "/api/v1/owners",
                json_body={"company_id": safe_uuid, "person": clean_person},
            )
            return ToolResult(DealportResult.from_api("/api/v1/owners", raw, max_characters))

        return dealport_find_owner
