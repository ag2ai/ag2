# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Outcrawl search and scrape extension for AG2.

Outcrawl (https://outcrawl.ai) is a scrape and search API for AI agents. This
extension wraps its REST API as agent tools: web search, optionally returning
each result page as markdown in the same call, and reading a single URL as
markdown, HTML, or links.

Maintainer: heydryft
Docs: https://docs.ag2.ai/docs/user-guide/extensions/tools/search/outcrawl/
"""

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeAlias
from urllib.parse import urlparse

import httpx
from pydantic import Field

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool

ScrapeFormat: TypeAlias = Literal["markdown", "html", "links"]

_API_KEY_ENV_VAR = "OUTCRAWL_API_KEY"  # pragma: allowlist secret
_USER_AGENT = "ag2-outcrawl-extension"
_SAFE_URL_SCHEMES = {"http", "https"}


class OutcrawlAPIError(Exception):
    """Raised when the Outcrawl API answers with a non-2xx status code.

    Attributes:
        status_code: HTTP status code of the response.
        code: Outcrawl error code from the response body, such as ``"unauthorized"``.
    """

    def __init__(self, status_code: int, code: str | None, message: str) -> None:
        super().__init__(f"Outcrawl API error {status_code}" + (f" ({code})" if code else "") + f": {message}")
        self.status_code = status_code
        self.code = code


def _safe_url(url: str) -> bool:
    return urlparse(url).scheme.lower() in _SAFE_URL_SCHEMES


def _truncate(text: str | None, max_characters: int | None) -> tuple[str | None, bool]:
    if text is None or max_characters is None or len(text) <= max_characters:
        return text, False
    return text[:max_characters], True


def _truncate_links(links: Sequence[str] | None, max_characters: int | None) -> tuple[list[str], bool]:
    """Keep leading links while their combined length stays within ``max_characters``."""
    if not links:
        return [], False
    if max_characters is None:
        return list(links), False
    kept: list[str] = []
    used = 0
    for link in links:
        used += len(link)
        if used > max_characters:
            return kept, True
        kept.append(link)
    return kept, False


def _as_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _str(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    return value if isinstance(value, str) else None


def _int(raw: dict[str, Any], key: str) -> int | None:
    value = raw.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass(slots=True)
class OutcrawlSearchResult:
    url: str
    title: str | None = None
    snippet: str | None = None
    rank: int | None = None
    markdown: str | None = None
    truncated: bool = False


@dataclass(slots=True)
class OutcrawlSearchResponse:
    query: str
    results: list[OutcrawlSearchResult] = field(default_factory=list)


@dataclass(slots=True)
class OutcrawlScrapeResult:
    url: str
    title: str | None = None
    description: str | None = None
    language: str | None = None
    status_code: int | None = None
    markdown: str | None = None
    html: str | None = None
    links: list[str] = field(default_factory=list)
    truncated: bool = False


async def _post(client_kwargs: dict[str, Any], api_key: str | None, path: str, body: dict[str, Any]) -> dict[str, Any]:
    """Run one POST against the Outcrawl REST API.

    Args:
        client_kwargs: Keyword arguments for the ``httpx.AsyncClient`` to open.
        api_key: Explicit API key. When empty, ``OUTCRAWL_API_KEY`` is read from
            the environment.
        path: Endpoint path, resolved against the client's base URL.
        body: JSON request body.

    Returns:
        The decoded JSON object, or an empty dict when the payload is not an object.

    Raises:
        ValueError: If no API key is given and ``OUTCRAWL_API_KEY`` is unset.
        OutcrawlAPIError: If Outcrawl answers with a non-2xx status code.
    """
    key = api_key or os.environ.get(_API_KEY_ENV_VAR)
    if not key:
        raise ValueError(f"An Outcrawl API key is required: pass api_key or set {_API_KEY_ENV_VAR}")

    async with httpx.AsyncClient(**client_kwargs) as client:
        response = await client.post(path, json=body, headers={"Authorization": f"Bearer {key}"})

    try:
        raw = response.json()
    except ValueError:
        raw = None

    if response.is_error:
        error = raw if isinstance(raw, dict) else {}
        raise OutcrawlAPIError(
            response.status_code,
            _str(error, "code"),
            _str(error, "message") or response.reason_phrase or "request failed",
        )

    return raw if isinstance(raw, dict) else {}


class OutcrawlToolkit(Toolkit):
    """Toolkit that exposes Outcrawl Search and Scrape as agent tools.

    The two tools mirror Outcrawl's public APIs:
      - ``outcrawl_search``: web search; with ``scrape_results=True`` each
        result also carries the content of its page as markdown
      - ``outcrawl_scrape``: read one URL as markdown, HTML, or a list of links

    By default, passing the whole toolkit to an agent registers both tools.
    To use a subset, or to customise per-tool parameters, call the factory
    methods directly and pass the returned tools to the agent::

        toolkit = OutcrawlToolkit(api_key=...)

        # both tools
        agent = Agent("a", config=config, tools=[toolkit])

        # only search, returning page markdown for each result
        agent = Agent("a", config=config, tools=[toolkit.search(limit=3, scrape_results=True)])

    ``OUTCRAWL_API_KEY`` is read from the environment when a tool runs and
    ``api_key`` is omitted, so building the toolkit never requires a key.

    ``max_characters`` caps the page content each tool returns (markdown and
    HTML, and the combined length of links); results that were cut have
    ``truncated=True``.

    ``outcrawl_scrape`` returns ``{"error": ...}`` for a URL that is not
    http or https, so the model can correct it. Every other problem (a
    missing API key, Outcrawl API errors) is raised.
    """

    __slots__ = ("_api_key", "_base_url", "_timeout")

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://api.outcrawl.ai",
        timeout: float = 60.0,
        limit: int | Variable | None = None,
        scrape_results: bool | Variable | None = None,
        formats: Sequence[ScrapeFormat] | Variable | None = None,
        only_main_content: bool | Variable | None = None,
        max_characters: int | Variable | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        """Build the toolkit and its two default tools.

        Args:
            api_key: Outcrawl API key, sent as a bearer token. When omitted,
                ``OUTCRAWL_API_KEY`` is read from the environment when a tool runs.
            base_url: Outcrawl API root. Trailing slashes are stripped.
            timeout: Per-request HTTP timeout in seconds.
            limit: Default number of results for ``outcrawl_search``.
            scrape_results: Default for ``outcrawl_search``: include each result
                page as markdown.
            formats: Default output formats for ``outcrawl_scrape``.
            only_main_content: Default for ``outcrawl_scrape``: drop navigation,
                headers, and footers.
            max_characters: Default content cap for both tools.
            middleware: Middleware applied to every tool in the toolkit.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

        super().__init__(
            self.search(limit=limit, scrape_results=scrape_results, max_characters=max_characters),
            self.scrape(formats=formats, only_main_content=only_main_content, max_characters=max_characters),
            name="outcrawl_toolkit",
            middleware=middleware,
        )

    def search(
        self,
        *,
        limit: int | Variable | None = None,
        scrape_results: bool | Variable | None = None,
        max_characters: int | Variable | None = None,
        name: str = "outcrawl_search",
        description: str = (
            "Search the web using Outcrawl. Returns ranked results with URLs, titles, and snippets. "
            "Results may also include the content of each page as markdown."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Create the ``outcrawl_search`` tool.

        Args:
            limit: Maximum number of results. Outcrawl uses its own default when omitted.
            scrape_results: When true, each result includes the content of its
                page as markdown. Each result is then a full page, so keep
                ``limit`` low to bound the size of the tool result.
            max_characters: With ``scrape_results``, cut each result's markdown
                to this many characters and set ``truncated=True`` on it.
            name: Tool name exposed to the model.
            description: Tool description exposed to the model.
            middleware: Tool middleware applied to this tool.
        """
        api_key = self._api_key
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def outcrawl_search(
            query: Annotated[str, Field(description="The search query string.")],
            ctx: Context,
        ) -> ToolResult:
            """Search the web using Outcrawl and return ranked results."""
            body: dict[str, Any] = {"query": query}
            resolved_limit = resolve_variable(limit, ctx, param_name="limit")
            if resolved_limit is not None:
                body["limit"] = resolved_limit
            if resolve_variable(scrape_results, ctx, param_name="scrape_results"):
                body["scrape"] = True
                body["formats"] = ["markdown"]
            limit_chars = resolve_variable(max_characters, ctx, param_name="max_characters")

            raw = await _post(client_kwargs, api_key, "/v1/search", body)

            hits = raw.get("hits")
            return ToolResult(
                OutcrawlSearchResponse(
                    query=query,
                    results=[
                        _search_result(hit, i, limit_chars)
                        for i, hit in enumerate(hits if isinstance(hits, list) else [], start=1)
                        if isinstance(hit, dict)
                    ],
                )
            )

        return outcrawl_search

    def scrape(
        self,
        *,
        formats: Sequence[ScrapeFormat] | Variable | None = None,
        only_main_content: bool | Variable | None = None,
        max_characters: int | Variable | None = None,
        name: str = "outcrawl_scrape",
        description: str = (
            "Read a single web page using Outcrawl. Returns the page content (markdown by default) "
            "with the page title, description, and status code."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Create the ``outcrawl_scrape`` tool.

        Args:
            formats: Output formats to return: ``"markdown"``, ``"html"``, ``"links"``.
                Defaults to ``["markdown"]``.
            only_main_content: Drop headers, navigation, and footers. Outcrawl
                defaults to ``True``.
            max_characters: Cut the markdown and the HTML to this many
                characters each, keep links only while their combined length
                fits, and set ``truncated=True`` on the result.
            name: Tool name exposed to the model.
            description: Tool description exposed to the model.
            middleware: Tool middleware applied to this tool.
        """
        api_key = self._api_key
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def outcrawl_scrape(
            url: Annotated[str, Field(description="The http or https URL of the page to read.")],
            ctx: Context,
        ) -> ToolResult:
            """Scrape one web page using Outcrawl and return its content."""
            if not _safe_url(url):
                return ToolResult({"error": f"Only http/https URLs are supported; rejected: {url!r}"})

            body: dict[str, Any] = {
                "url": url,
                "formats": _as_list(resolve_variable(formats, ctx, param_name="formats")) or ["markdown"],
            }
            resolved_main_content = resolve_variable(only_main_content, ctx, param_name="only_main_content")
            if resolved_main_content is not None:
                body["onlyMainContent"] = resolved_main_content
            limit_chars = resolve_variable(max_characters, ctx, param_name="max_characters")

            raw = await _post(client_kwargs, api_key, "/v1/scrape", body)

            markdown, markdown_cut = _truncate(_str(raw, "markdown"), limit_chars)
            html, html_cut = _truncate(_str(raw, "html"), limit_chars)
            raw_links = raw.get("links")
            links, links_cut = _truncate_links(
                [link for link in raw_links if isinstance(link, str)] if isinstance(raw_links, list) else None,
                limit_chars,
            )
            metadata = raw.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            return ToolResult(
                OutcrawlScrapeResult(
                    url=_str(raw, "url") or url,
                    title=_str(metadata, "title"),
                    description=_str(metadata, "description"),
                    language=_str(metadata, "language"),
                    status_code=_int(raw, "statusCode"),
                    markdown=markdown,
                    html=html,
                    links=links,
                    truncated=markdown_cut or html_cut or links_cut,
                )
            )

        return outcrawl_scrape

    def _client_kwargs(self) -> dict[str, Any]:
        """Snapshot the connection settings as ``httpx.AsyncClient`` keyword arguments."""
        return {
            "base_url": self._base_url,
            "headers": {"User-Agent": _USER_AGENT},
            "timeout": self._timeout,
        }


def _search_result(hit: dict[str, Any], position: int, max_characters: int | None) -> OutcrawlSearchResult:
    markdown, truncated = _truncate(_str(hit, "markdown"), max_characters)
    return OutcrawlSearchResult(
        url=_str(hit, "url") or "",
        title=_str(hit, "title"),
        snippet=_str(hit, "snippet"),
        rank=_int(hit, "rank") or position,
        markdown=markdown,
        truncated=truncated,
    )
