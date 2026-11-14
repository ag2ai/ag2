# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""SpecStream search and extract extension for AG2.

SpecStream is a live web data engine with a Search API and an Extract API.
This extension wraps both endpoints as agent tools. Search answers a question
from the live web with an answer, rows and the source pages they come from;
extract reads pages as markdown or plain text.

Maintainer: Specstreamai
Docs: https://docs.ag2.ai/docs/user-guide/extensions/tools/search/specstream/
"""

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import httpx
from pydantic import Field

from ag2.annotations import Context, Variable
from ag2.events import ToolResult
from ag2.middleware import ToolMiddleware
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.final import Toolkit, tool
from ag2.tools.final.function_tool import FunctionTool

_API_KEY_ENV_VAR = "SPECSTREAM_API_KEY"  # pragma: allowlist secret


@dataclass(slots=True)
class SpecStreamSource:
    url: str
    title: str = ""
    site: str = ""
    snippet: str = ""
    published: str = ""


@dataclass(slots=True)
class SpecStreamSearchResponse:
    query: str
    kind: str = ""
    answer: str = ""
    answer_page: str = ""
    sources: list[SpecStreamSource] = field(default_factory=list)
    passages: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class SpecStreamExtractResult:
    url: str
    status: str = ""
    title: str = ""
    text: str = ""
    published: str = ""


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


def _raise_for_status(response: httpx.Response) -> None:
    """Raise ``httpx.HTTPStatusError`` carrying SpecStream's own error message.

    The API key travels in a header, never in the URL, so neither the message
    nor the request URL attached to the error exposes it.
    """
    if response.is_success:
        return

    try:
        body = response.json()
        detail = _text(body, "message") or _text(body, "detail") or _text(body.get("error"), "message")
    except (ValueError, AttributeError):
        detail = ""

    message = f"SpecStream API returned HTTP {response.status_code}"
    if detail:
        message += f": {detail}"
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
    """Run one request against the SpecStream REST API.

    Args:
        client_kwargs: Keyword arguments for the ``httpx.AsyncClient`` to open.
        api_key: Explicit API key. When empty, ``SPECSTREAM_API_KEY`` is read from the environment.
        method: HTTP method.
        path: Endpoint path, resolved against the client's base URL.
        params: Query parameters. Entries with a ``None`` value are dropped.
        body: JSON request body.

    Returns:
        The ``data`` object of the reply envelope (the whole body when it has none), or an empty dict when the payload is not an object.

    Raises:
        ValueError: If no API key is available.
        httpx.HTTPStatusError: If SpecStream answers with a non-2xx status code.
        ValueError: If the reply carries an error and no data.
    """
    key = api_key or os.environ.get(_API_KEY_ENV_VAR)
    if not key:
        raise ValueError(f"SpecStream needs an API key: pass api_key=... or set {_API_KEY_ENV_VAR}.")

    query = {k: v for k, v in params.items() if v is not None} if params is not None else None

    async with httpx.AsyncClient(**client_kwargs) as client:
        response = await client.request(
            method, path, params=query, json=body, headers={"Authorization": f"Bearer {key}"}
        )
        _raise_for_status(response)
        raw = response.json()

    return _unwrap(raw)


def _unwrap(raw: Any) -> dict[str, Any]:
    """Return the ``data`` object of SpecStream's reply envelope, or the body itself when it has no envelope.

    Raises:
        ValueError: If the envelope carries an error and no data.
    """
    if not isinstance(raw, dict):
        return {}
    data = raw.get("data") or raw
    if data is raw and raw.get("error"):
        raise ValueError(f"SpecStream error: {_text(raw.get('error'), 'message') or 'request failed'}")
    return data if isinstance(data, dict) else {}


class SpecStreamSearchToolkit(Toolkit):
    """Toolkit that exposes SpecStream web search and page extraction as agent tools.

    The two tools mirror SpecStream's REST endpoints:
      - ``specstream_search``: ``GET /v1/web-search``, an answer from the live web with its source pages
      - ``specstream_extract``: ``POST /v1/extract``, the main text of one page as markdown or plain text

    By default, passing the whole toolkit to an agent registers both tools.
    To use a subset, call the factory methods and pass the returned tools to the agent::

        toolkit = SpecStreamSearchToolkit()

        # both tools
        agent = Agent("a", config=config, tools=[toolkit])

        # only search, with page passages in each result
        agent = Agent("a", config=config, tools=[toolkit.search(passages=True)])

    An API key is required. When ``api_key`` is omitted, ``SPECSTREAM_API_KEY``
    is read from the environment at call time.
    """

    __slots__ = ("_api_key", "_base_url", "_timeout", "_proxy", "_verify")

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://api.specstream.ai",
        timeout: float = 60.0,
        proxy: str | None = None,
        verify: bool = True,
        passages: bool | Variable | None = None,
        extract_format: Literal["markdown", "text"] | Variable | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        """Build the toolkit and its two default tools.

        Args:
            api_key: SpecStream API key, sent as a bearer token. Falls back to ``SPECSTREAM_API_KEY``.
            base_url: SpecStream API root. Trailing slashes are stripped.
            timeout: Per-request timeout in seconds.
            proxy: Proxy URL for the outgoing HTTP connection.
            verify: Whether to verify the server's TLS certificate.
            passages: Default for returning the text of the pages closest to the query with each search.
            extract_format: Default output format of ``specstream_extract``: ``"markdown"`` (API default) or ``"text"``.
            middleware: Middleware applied to every tool in the toolkit.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._proxy = proxy
        self._verify = verify

        super().__init__(
            self.search(passages=passages),
            self.extract(format=extract_format),
            name="specstream_search_toolkit",
            middleware=middleware,
        )

    def search(
        self,
        *,
        passages: bool | Variable | None = None,
        name: str = "specstream_search",
        description: str = (
            "Search the live web using SpecStream. Write the query in everyday words. "
            "Returns an answer when the results agree on one, and the source pages with titles, URLs and snippets."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the web search tool.

        Args:
            passages: Also return the text of the pages closest to the query.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A ``FunctionTool`` that queries SpecStream's ``/v1/web-search`` endpoint.
        """
        api_key = self._api_key
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def specstream_search(
            query: Annotated[str, Field(description="The question or search request, 1 to 2000 characters.")],
            ctx: Context,
        ) -> ToolResult:
            """Search the live web using SpecStream and return the answer and its sources."""
            want_passages = resolve_variable(passages, ctx, param_name="passages")
            raw = await _request(
                client_kwargs,
                api_key,
                "GET",
                "/v1/web-search",
                params={"q": query, "passages": "true" if want_passages else None},
            )
            answer = raw.get("answer")
            return ToolResult(
                SpecStreamSearchResponse(
                    query=query,
                    kind=_text(raw, "kind"),
                    answer=_text(answer, "value"),
                    answer_page=_text(answer, "page"),
                    sources=[
                        SpecStreamSource(
                            url=_text(item, "url"),
                            title=_text(item, "title"),
                            site=_text(item, "site"),
                            snippet=_text(item, "snippet"),
                            published=_text(item, "published"),
                        )
                        for item in _items(raw, "sources")
                    ],
                    passages=[
                        {"url": _text(item, "url"), "title": _text(item, "title"), "text": _text(item, "text")}
                        for item in _items(raw, "passages")
                    ],
                )
            )

        return specstream_search

    def extract(
        self,
        *,
        format: Literal["markdown", "text"] | Variable | None = None,  # noqa: A002
        name: str = "specstream_extract",
        description: str = (
            "Read a web page using SpecStream. Returns the page's main text without menus and ads, "
            "with its title and publication date when known."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        """Build the page extraction tool.

        Args:
            format: ``"markdown"`` keeps headings, lists and links (API default); ``"text"`` is plain text.
            name: Tool name registered with the agent.
            description: Tool description shown to the model.
            middleware: Middleware applied to this tool.

        Returns:
            A ``FunctionTool`` that queries SpecStream's ``/v1/extract`` endpoint.
        """
        api_key = self._api_key
        client_kwargs = self._client_kwargs()

        @tool(name=name, description=description, middleware=middleware)
        async def specstream_extract(
            url: Annotated[str, Field(description="The http or https URL of the page to read.")],
            ctx: Context,
        ) -> ToolResult:
            """Read one web page using SpecStream and return its main text."""
            body: dict[str, Any] = {"urls": [url]}
            resolved = resolve_variable(format, ctx, param_name="format")
            if resolved is not None:
                body["format"] = resolved
            raw = await _request(client_kwargs, api_key, "POST", "/v1/extract", body=body)
            results = _items(raw, "results")
            item = results[0] if results else {}
            return ToolResult(
                SpecStreamExtractResult(
                    url=_text(item, "url") or url,
                    status=_text(item, "status"),
                    title=_text(item, "title"),
                    text=_text(item, "text"),
                    published=_text(item, "published"),
                )
            )

        return specstream_extract

    def _client_kwargs(self) -> dict[str, Any]:
        """Snapshot the connection settings as ``httpx.AsyncClient`` keyword arguments."""
        return {
            "base_url": self._base_url,
            "timeout": self._timeout,
            "proxy": self._proxy,
            "verify": self._verify,
        }
