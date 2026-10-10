# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Keenable search and fetch extension for AG2.

Keenable is an independent web index with a REST API for web search and page
fetch. This extension wraps both endpoints as agent tools. It works without an
API key: keyless calls go to the public endpoints, and a ``KEENABLE_API_KEY``
only raises the rate limits.

Maintainer: ilya-bogin-keenable
Docs: https://docs.ag2.ai/docs/user-guide/extensions/tools/search/keenable/
"""

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Any

import httpx
from pydantic import Field

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool

_API_KEY_ENV_VAR = "KEENABLE_API_KEY"  # pragma: allowlist secret

# Keenable attributes traffic by this header and rejects keyless calls without it.
_APP_TITLE = "ag2"

# Left unset, Keenable returns several thousand characters of page text per result
# (5-8k seen live), so ten results would put ~15k tokens into the model's context.
# Snippets are capped here instead; ``keenable_fetch`` reads a whole page on demand.
_DEFAULT_SNIPPET_MAX_LENGTH = 1000


@dataclass(slots=True)
class KeenableSearchResult:
    title: str
    url: str
    snippet: str = ""
    acquired_at: str = ""


@dataclass(slots=True)
class KeenableSearchResponse:
    query: str
    results: list[KeenableSearchResult] = field(default_factory=list)


@dataclass(slots=True)
class KeenableFetchResult:
    url: str
    title: str = ""
    content: str = ""
    author: str = ""
    published_at: str = ""


def _text(raw: Any, key: str) -> str:
    """Return ``raw[key]`` when *raw* is a mapping holding a string there, else ``""``."""
    if not isinstance(raw, dict):
        return ""
    value = raw.get(key)
    return value if isinstance(value, str) else ""


def _items(raw: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Return the mappings inside the list at ``raw[key]``, skipping anything else."""
    items = raw.get(key)
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _timestamp(value: Any) -> str:
    """Render a Unix timestamp in seconds as an ISO 8601 UTC string, passing strings through."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return ""
    return ""


def _raise_for_status(response: httpx.Response, *, keyless: bool) -> None:
    """Raise ``httpx.HTTPStatusError`` carrying Keenable's own error message.

    The API key travels in a header, never in the URL, so neither the message
    nor the request URL attached to the error exposes it.
    """
    if response.is_success:
        return

    try:
        detail = _text(response.json(), "message")
    except ValueError:
        detail = ""

    message = f"Keenable API returned HTTP {response.status_code}"
    if detail:
        message += f": {detail}"
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            message += f" (Retry-After: {retry_after})"
        if keyless:
            message += f" Set {_API_KEY_ENV_VAR} to raise the keyless rate limit."
    raise httpx.HTTPStatusError(message, request=response.request, response=response)


async def _request(
    client_kwargs: dict[str, Any],
    api_key: str | None,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one request against the Keenable REST API.

    Args:
        client_kwargs: Keyword arguments for the ``httpx.AsyncClient`` to open.
        api_key: Explicit API key. When empty, ``KEENABLE_API_KEY`` is read from
            the environment; when that is unset too, the keyless ``/public``
            variant of *path* is called.
        method: HTTP method.
        path: Keyed endpoint path, resolved against the client's base URL.
        params: Query parameters.
        body: JSON request body. Entries with a ``None`` value are dropped.

    Returns:
        The decoded JSON object, or an empty dict when the payload is not an object.

    Raises:
        httpx.HTTPStatusError: If Keenable answers with a non-2xx status code.
    """
    key = api_key or os.environ.get(_API_KEY_ENV_VAR)
    headers = {"X-Keenable-Title": _APP_TITLE}
    if key:
        headers["X-API-Key"] = key
    else:
        path = f"{path}/public"

    payload = {k: v for k, v in body.items() if v is not None} if body is not None else None

    async with httpx.AsyncClient(**client_kwargs) as client:
        response = await client.request(method, path, params=params, json=payload, headers=headers)
        _raise_for_status(response, keyless=not key)
        raw = response.json()

    return raw if isinstance(raw, dict) else {}


class KeenableSearchToolkit(Toolkit):
    """Toolkit that exposes Keenable web search and page fetch as agent tools.

    The two tools mirror Keenable's REST endpoints:
      - ``keenable_search``: web search with ranked titles, URLs, and page text
      - ``keenable_fetch``: fetch one URL and return its content as markdown

    By default, passing the whole toolkit to an agent registers both tools.
    To use a subset, or to customise per-tool defaults, call the factory
    methods and pass the returned tools to the agent::

        toolkit = KeenableSearchToolkit()

        # both tools
        agent = Agent("a", config=config, tools=[toolkit])

        # only search, restricted to one site
        agent = Agent("a", config=config, tools=[toolkit.search(site="docs.python.org", max_results=5)])

    No API key is needed. When ``api_key`` is omitted, ``KEENABLE_API_KEY`` is
    read from the environment at call time; when neither is set, the toolkit
    calls Keenable's keyless endpoints, which have lower rate limits.
    """

    __slots__ = ("_api_key", "_base_url", "_timeout", "_proxy", "_verify")

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://api.keenable.ai",
        timeout: float = 60.0,
        proxy: str | None = None,
        verify: bool = True,
        max_results: int | Variable | None = None,
        site: str | Variable | None = None,
        published_after: str | Variable | None = None,
        snippet_max_length: int | Variable | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        """Build the toolkit and its two default tools.

        Args:
            api_key: Optional Keenable API key, sent as the ``X-API-Key`` header.
                Falls back to ``KEENABLE_API_KEY``; without either, the keyless
                endpoints are used.
            base_url: Keenable API root. Trailing slashes are stripped.
            timeout: Per-request timeout in seconds.
            proxy: Proxy URL for the outgoing HTTP connection.
            verify: Whether to verify the server's TLS certificate.
            max_results: Default number of search results (1-50, API default 10).
            site: Default domain to restrict search results to, e.g. ``"docs.python.org"``.
            published_after: Default lower bound on publication date, as ``YYYY-MM-DD``.
            snippet_max_length: Default approximate length of each result's
                ``snippet``, in characters (defaults to 1000). Keenable treats
                it as a hint, so a snippet can run slightly over.
            middleware: Middleware applied to every tool in the toolkit.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._proxy = proxy
        self._verify = verify

        super().__init__(
            self.search(
                max_results=max_results,
                site=site,
                published_after=published_after,
                snippet_max_length=snippet_max_length,
            ),
            self.fetch(),
            name="keenable_search_toolkit",
            middleware=middleware,
        )

    def search(
        self,
        *,
        max_results: int | Variable | None = None,
        site: str | Variable | None = None,
        published_after: str | Variable | None = None,
        snippet_max_length: int | Variable | None = None,
        name: str = "keenable_search",
        description: str = (
            "Search the web using Keenable. Returns ranked results with titles, URLs, and a snippet of each page's text."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the web search tool.

        Args:
            max_results: Number of results to return (1-50, API default 10).
            site: Domain to restrict results to, e.g. ``"docs.python.org"``.
            published_after: Only return pages published after this date (``YYYY-MM-DD``).
            snippet_max_length: Approximate length of each result's ``snippet``, in
                characters (defaults to 1000).
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A ``FunctionTool`` that queries Keenable's ``/v1/search`` endpoint.
        """
        api_key = self._api_key
        client_kwargs = self._client_kwargs()
        _snippet_max_length = _DEFAULT_SNIPPET_MAX_LENGTH if snippet_max_length is None else snippet_max_length

        @tool(name=name, description=description, middleware=middleware)
        async def keenable_search(
            query: Annotated[str, Field(description="The search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search the web using Keenable and return ranked results."""
            raw = await _request(
                client_kwargs,
                api_key,
                "POST",
                "/v1/search",
                body={
                    "query": query,
                    "max_results": resolve_variable(max_results, ctx, param_name="max_results"),
                    "site": resolve_variable(site, ctx, param_name="site"),
                    "published_after": resolve_variable(published_after, ctx, param_name="published_after"),
                    "snippet_max_length": resolve_variable(_snippet_max_length, ctx, param_name="snippet_max_length"),
                },
            )
            return ToolResult(
                KeenableSearchResponse(
                    query=query,
                    results=[
                        KeenableSearchResult(
                            title=_text(item, "title"),
                            url=_text(item, "url"),
                            # The page text is in `snippet`; `description` is kept only as a fallback.
                            snippet=_text(item, "snippet") or _text(item, "description"),
                            acquired_at=_text(item, "acquired_at"),
                        )
                        for item in _items(raw, "results")
                    ],
                )
            )

        return keenable_search

    def fetch(
        self,
        *,
        name: str = "keenable_fetch",
        description: str = (
            "Fetch a web page using Keenable. Returns the page's main content as markdown, "
            "with its title, author, and publication date when known."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the page fetch tool.

        Args:
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A ``FunctionTool`` that queries Keenable's ``/v1/fetch`` endpoint.
        """
        api_key = self._api_key
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def keenable_fetch(
            url: Annotated[str, Field(description="The http or https URL of the page to fetch.")],
        ) -> ToolResult:
            """Fetch one web page using Keenable and return its content as markdown."""
            raw = await _request(client_kwargs, api_key, "GET", "/v1/fetch", params={"url": url})
            return ToolResult(
                KeenableFetchResult(
                    url=_text(raw, "url") or url,
                    title=_text(raw, "title"),
                    content=_text(raw, "content"),
                    author=_text(raw, "author"),
                    published_at=_timestamp(raw.get("published_at")),
                )
            )

        return keenable_fetch

    def _client_kwargs(self) -> dict[str, Any]:
        """Snapshot the connection settings as ``httpx.AsyncClient`` keyword arguments."""
        return {
            "base_url": self._base_url,
            "timeout": self._timeout,
            "proxy": self._proxy,
            "verify": self._verify,
        }
