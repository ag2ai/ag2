# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Glasser paid-data tools for AG2, using one key and pay-per-call billing.

Maintainer: glasserai
Docs: https://docs.ag2.ai/docs/user-guide/extensions/tools/search/glasser/
"""

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

import anyio
import httpx
from pydantic import Field

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool


def _pending(result: dict[str, Any]) -> bool:
    return result.get("status") in ("QUEUED", "RUNNING") or result.get("result_status") == "pending"


def _object(response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError("Glasser returned a non-object JSON response")
    return result


def _content(value: Any, remaining: int | None) -> tuple[Any, int | None, bool]:
    if value is None or remaining is None:
        return value, remaining, False
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= remaining:
        return value, remaining - len(text), False
    return text[:remaining], 0, True


@dataclass(slots=True)
class GlasserResult:
    """The Glasser outcome, with content bounded separately from billing and retrieval fields."""

    id: str | None = None
    status: str | None = None
    result_status: str | None = None
    provider: str | None = None
    charge_usd: str | None = None
    output: Any = None
    error: Any = None
    failure: Any = None
    provider_response: Any = None
    runs: Any = None
    run_url: str | None = None
    result_url: str | None = None
    truncated: bool = False

    @classmethod
    def from_api(cls, result: dict[str, Any], max_characters: int | None) -> "GlasserResult":
        """Limit content across four fields without cutting status, charges or retrieval URLs."""
        if max_characters is not None and max_characters < 0:
            raise ValueError("max_characters must be non-negative")
        content: dict[str, Any] = {}
        remaining = max_characters
        truncated = False
        for key in ("output", "error", "failure", "provider_response"):
            content[key], remaining, cut = _content(result.get(key), remaining)
            truncated |= cut
        return cls(
            id=result.get("id"),
            status=result.get("status"),
            result_status=result.get("result_status"),
            provider=result.get("provider"),
            charge_usd=result.get("charge_usd"),
            runs=result.get("runs"),
            run_url=result.get("run_url"),
            result_url=result.get("result_url"),
            truncated=truncated,
            **content,
        )


def _poll_url(client: httpx.AsyncClient, capability: str, path: str, result: dict[str, Any]) -> httpx.URL:
    if capability == "people_search":
        run_id = result.get("id")
        if not isinstance(run_id, str) or not run_id or any(c not in "0123456789abcdefABCDEF-" for c in run_id):
            raise ValueError("Glasser returned an invalid pending Run id")
        return client.base_url.join(f"v1/runs/{run_id}")

    result_url = result.get("result_url")
    if not isinstance(result_url, str):
        raise ValueError("Glasser returned no result_url for a pending result")
    poll_url = client.base_url.join(result_url)
    expected = client.base_url.join(f"{path}/result")
    if (
        (poll_url.scheme, poll_url.host, poll_url.port, poll_url.path)
        != (expected.scheme, expected.host, expected.port, expected.path)
        or poll_url.userinfo
        or poll_url.fragment
    ):
        raise ValueError("Glasser result_url must point to this capability on the configured API origin")
    return poll_url


@dataclass(frozen=True, slots=True)
class _GlasserClient:
    api_key: str = field(repr=False)
    base_url: str
    timeout: float
    wait_timeout: float
    poll_interval: float

    async def execute(
        self, capability: str, payload: dict[str, Any], ctx: Context, max_characters: int | None
    ) -> ToolResult:
        if max_characters is not None and max_characters < 0:
            raise ValueError("max_characters must be non-negative")
        # Scope an LLM's call ID to its stream: different conversations can reuse call IDs.
        key = (
            str(uuid5(NAMESPACE_URL, f"ag2-glasser:{ctx.stream.id}:{ctx.tool_call_id}:{capability}"))
            if ctx.tool_call_id
            else str(uuid4())
        )
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "User-Agent": "ag2-glasser-extension"},
            timeout=self.timeout,
            follow_redirects=False,
        ) as client:
            path = f"v1/solutions/gtm/{capability}"
            response = await client.post(
                path,
                json={name: value for name, value in payload.items() if value is not None},
                headers={"Idempotency-Key": key},
            )
            result = _object(response)
            if not _pending(result) or self.wait_timeout == 0:
                return ToolResult(GlasserResult.from_api(result, max_characters))

            poll_url = _poll_url(client, capability, path, result)
            # Only reads are repeated. On expiry retain the pending result and its retrieval information.
            with anyio.move_on_after(self.wait_timeout):
                while _pending(result):
                    await anyio.sleep(self.poll_interval)
                    try:
                        result = _object(await client.get(poll_url))
                    except (httpx.HTTPError, ValueError):
                        return ToolResult(GlasserResult.from_api(result, max_characters))
            return ToolResult(GlasserResult.from_api(result, max_characters))


class GlasserToolkit(Toolkit):
    """Six narrow tools backed by Glasser's GTM Solution API."""

    __slots__ = ("_client",)

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.glasser.ai",
        timeout: float = 60.0,
        wait_timeout: float = 120.0,
        poll_interval: float = 1.0,
        provider: str | Variable | None = None,
        limit: int | Variable | None = None,
        country: str | Variable | None = None,
        task_id: str | Variable | None = None,
        metadata: dict[str, str] | Variable | None = None,
        max_characters: int | Variable | None = 10000,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key is required")
        url = httpx.URL(base_url)
        if url.scheme not in ("http", "https") or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP(S) API root without credentials, query or fragment")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if not math.isfinite(wait_timeout) or wait_timeout < 0:
            raise ValueError("wait_timeout must be finite and non-negative")
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be finite and positive")
        self._client = _GlasserClient(api_key, base_url.rstrip("/") + "/", timeout, wait_timeout, poll_interval)
        super().__init__(
            self.web_search(
                provider=provider,
                limit=limit,
                country=country,
                task_id=task_id,
                metadata=metadata,
                max_characters=max_characters,
            ),
            self.company_profile(
                provider=provider,
                limit=limit,
                country=country,
                task_id=task_id,
                metadata=metadata,
                max_characters=max_characters,
            ),
            self.find_people(
                provider=provider, limit=limit, task_id=task_id, metadata=metadata, max_characters=max_characters
            ),
            self.keyword_overview(
                provider=provider,
                limit=limit,
                country=country,
                task_id=task_id,
                metadata=metadata,
                max_characters=max_characters,
            ),
            self.social_search(provider=provider, metadata=metadata, max_characters=max_characters),
            self.stock_quote(provider=provider, task_id=task_id, metadata=metadata, max_characters=max_characters),
            name="glasser_toolkit",
            middleware=middleware,
        )

    def web_search(
        self,
        *,
        provider: str | Variable | None = None,
        limit: int | Variable | None = None,
        country: str | Variable | None = None,
        task_id: str | Variable | None = None,
        metadata: dict[str, str] | Variable | None = None,
        max_characters: int | Variable | None = 10000,
        name: str = "glasser_web_search",
        description: str = "Search the web through Glasser.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        @tool(name=name, description=description, middleware=middleware)
        async def glasser_web_search(
            query: Annotated[str, Field(description="The web search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search the web through Glasser."""
            if not query.strip():
                return ToolResult({"error": "query must not be empty"})
            return await self._client.execute(
                "web_research",
                {
                    "action": "search",
                    "query": query,
                    "provider": resolve_variable(provider, ctx, param_name="provider"),
                    "limit": resolve_variable(limit, ctx, param_name="limit"),
                    "country": resolve_variable(country, ctx, param_name="country"),
                    "task_id": resolve_variable(task_id, ctx, param_name="task_id"),
                    "metadata": resolve_variable(metadata, ctx, param_name="metadata"),
                },
                ctx,
                resolve_variable(max_characters, ctx, param_name="max_characters"),
            )

        return glasser_web_search

    def company_profile(
        self,
        *,
        provider: str | Variable | None = None,
        limit: int | Variable | None = None,
        country: str | Variable | None = None,
        task_id: str | Variable | None = None,
        metadata: dict[str, str] | Variable | None = None,
        max_characters: int | Variable | None = 10000,
        name: str = "glasser_company_profile",
        description: str = "Look up a company profile by domain through Glasser.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        @tool(name=name, description=description, middleware=middleware)
        async def glasser_company_profile(
            domain: Annotated[str, Field(description="The company website domain.")],
            ctx: Context,
        ) -> ToolResult:
            """Look up a company profile through Glasser."""
            if not domain.strip():
                return ToolResult({"error": "domain must not be empty"})
            return await self._client.execute(
                "company_intelligence",
                {
                    "action": "enrich",
                    "domain": domain,
                    "provider": resolve_variable(provider, ctx, param_name="provider"),
                    "limit": resolve_variable(limit, ctx, param_name="limit"),
                    "country": resolve_variable(country, ctx, param_name="country"),
                    "task_id": resolve_variable(task_id, ctx, param_name="task_id"),
                    "metadata": resolve_variable(metadata, ctx, param_name="metadata"),
                },
                ctx,
                resolve_variable(max_characters, ctx, param_name="max_characters"),
            )

        return glasser_company_profile

    def find_people(
        self,
        *,
        provider: str | Variable | None = None,
        limit: int | Variable | None = None,
        task_id: str | Variable | None = None,
        metadata: dict[str, str] | Variable | None = None,
        max_characters: int | Variable | None = 10000,
        name: str = "glasser_find_people",
        description: str = "Find people by job, location or company through Glasser.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        @tool(name=name, description=description, middleware=middleware)
        async def glasser_find_people(
            ctx: Context,
            job_titles: Annotated[list[str] | None, Field(description="Job titles to match.")] = None,
            locations: Annotated[list[str] | None, Field(description="Locations to match.")] = None,
            company_domain: Annotated[str | None, Field(description="The company website domain.")] = None,
            keywords: Annotated[str | None, Field(description="Keywords to match.")] = None,
        ) -> ToolResult:
            """Find people through Glasser."""
            if not any((
                job_titles and any(title.strip() for title in job_titles),
                locations and any(location.strip() for location in locations),
                company_domain and company_domain.strip(),
                keywords and keywords.strip(),
            )):
                return ToolResult({"error": "supply job_titles, locations, company_domain or keywords"})
            return await self._client.execute(
                "people_search",
                {
                    "action": "search",
                    "job_titles": job_titles,
                    "locations": locations,
                    "company_domain": company_domain,
                    "keywords": keywords,
                    "provider": resolve_variable(provider, ctx, param_name="provider"),
                    "limit": resolve_variable(limit, ctx, param_name="limit"),
                    "task_id": resolve_variable(task_id, ctx, param_name="task_id"),
                    "metadata": resolve_variable(metadata, ctx, param_name="metadata"),
                },
                ctx,
                resolve_variable(max_characters, ctx, param_name="max_characters"),
            )

        return glasser_find_people

    def keyword_overview(
        self,
        *,
        provider: str | Variable | None = None,
        limit: int | Variable | None = None,
        country: str | Variable | None = None,
        task_id: str | Variable | None = None,
        metadata: dict[str, str] | Variable | None = None,
        max_characters: int | Variable | None = 10000,
        name: str = "glasser_keyword_overview",
        description: str = "Research SEO keywords through Glasser.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        @tool(name=name, description=description, middleware=middleware)
        async def glasser_keyword_overview(
            keywords: Annotated[list[str], Field(description="Keywords to research.")],
            ctx: Context,
        ) -> ToolResult:
            """Research SEO keywords through Glasser."""
            if not keywords or not any(keyword.strip() for keyword in keywords):
                return ToolResult({"error": "keywords must contain a non-empty value"})
            return await self._client.execute(
                "seo_research",
                {
                    "action": "keyword_overview",
                    "keywords": keywords,
                    "provider": resolve_variable(provider, ctx, param_name="provider"),
                    "limit": resolve_variable(limit, ctx, param_name="limit"),
                    "country": resolve_variable(country, ctx, param_name="country"),
                    "task_id": resolve_variable(task_id, ctx, param_name="task_id"),
                    "metadata": resolve_variable(metadata, ctx, param_name="metadata"),
                },
                ctx,
                resolve_variable(max_characters, ctx, param_name="max_characters"),
            )

        return glasser_keyword_overview

    def social_search(
        self,
        *,
        provider: str | Variable | None = None,
        metadata: dict[str, str] | Variable | None = None,
        max_characters: int | Variable | None = 10000,
        name: str = "glasser_social_search",
        description: str = "Search public social posts through Glasser.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        @tool(name=name, description=description, middleware=middleware)
        async def glasser_social_search(
            platform: Annotated[
                Literal["reddit", "x", "youtube", "tiktok", "instagram", "linkedin"],
                Field(description="The social platform."),
            ],
            query: Annotated[str, Field(description="The post search query.")],
            ctx: Context,
        ) -> ToolResult:
            """Search public social posts through Glasser."""
            if not query.strip():
                return ToolResult({"error": "query must not be empty"})
            return await self._client.execute(
                "social_research",
                {
                    "mode": "search",
                    "platform": platform,
                    "query": query,
                    "provider": resolve_variable(provider, ctx, param_name="provider"),
                    "metadata": resolve_variable(metadata, ctx, param_name="metadata"),
                },
                ctx,
                resolve_variable(max_characters, ctx, param_name="max_characters"),
            )

        return glasser_social_search

    def stock_quote(
        self,
        *,
        provider: str | Variable | None = None,
        task_id: str | Variable | None = None,
        metadata: dict[str, str] | Variable | None = None,
        max_characters: int | Variable | None = 10000,
        name: str = "glasser_stock_quote",
        description: str = "Get a stock quote through Glasser.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        @tool(name=name, description=description, middleware=middleware)
        async def glasser_stock_quote(
            symbol: Annotated[str, Field(description="The stock ticker symbol.")],
            ctx: Context,
        ) -> ToolResult:
            """Get a stock quote through Glasser."""
            if not symbol.strip():
                return ToolResult({"error": "symbol must not be empty"})
            return await self._client.execute(
                "market_data",
                {
                    "action": "stock_quote",
                    "symbol": symbol,
                    "provider": resolve_variable(provider, ctx, param_name="provider"),
                    "task_id": resolve_variable(task_id, ctx, param_name="task_id"),
                    "metadata": resolve_variable(metadata, ctx, param_name="metadata"),
                },
                ctx,
                resolve_variable(max_characters, ctx, param_name="max_characters"),
            )

        return glasser_stock_quote
