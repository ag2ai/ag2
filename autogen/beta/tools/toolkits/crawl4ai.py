# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Crawl4AIToolkit — web-crawling tools for AG2 Beta agents.

Gives an agent two tools: ``crawl`` and ``crawl_many``.

``crawl`` fetches a single URL and returns its main content as Markdown.
``crawl_many`` fetches several URLs concurrently with an optional concurrency cap.

Requires the ``crawl4ai`` package::

    pip install crawl4ai

Basic usage::

    from autogen.beta import Agent
    from autogen.beta.tools import Crawl4AIToolkit
    from autogen.beta.config import OpenAIConfig

    crawler = Crawl4AIToolkit()
    agent = Agent("researcher", config=OpenAIConfig("gpt-4o-mini"), tools=[crawler])

Limit concurrent fetches to avoid overwhelming servers::

    crawler = Crawl4AIToolkit(max_concurrent=2)
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Annotated
from urllib.parse import urlparse

from pydantic import Field

from autogen.beta.middleware import ToolMiddleware
from autogen.beta.tools.final import Toolkit, tool
from autogen.beta.tools.final.function_tool import FunctionTool

__all__ = ("Crawl4AIToolkit",)

_DEFAULT_MAX_CONCURRENT = 3
_SAFE_SCHEMES = {"http", "https"}


def _safe_url(url: str) -> str:
    """Return url unchanged if scheme is http/https; otherwise return empty string."""
    scheme = urlparse(url).scheme.lower()
    return url if scheme in _SAFE_SCHEMES else ""


async def _fetch(url: str, *, include_links: bool, include_images: bool) -> str:
    """Fetch one URL and return a formatted result string."""
    if not _safe_url(url):
        return f"[invalid URL] Only http/https URLs are supported; got {url!r}"

    try:
        from crawl4ai import AsyncWebCrawler  # type: ignore[import]
    except ImportError:
        return f"[crawl4ai not installed] Cannot fetch {url!r}. Install it with: pip install crawl4ai"

    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
    except Exception as exc:
        return f"[error fetching {url!r}] {exc}"

    if not result.success:
        return f"[fetch failed for {url!r}] {getattr(result, 'error_message', 'unknown error')}"

    parts: list[str] = []
    if result.title:
        parts.append(f"# {result.title}")
    parts.append(f"URL: {url}")
    parts.append("")

    content = (result.markdown or "").strip()
    if content:
        parts.append(content)
    else:
        parts.append("(no content extracted)")

    if include_links:
        links = getattr(result, "links", {})
        internal = links.get("internal", [])
        external = links.get("external", [])
        if internal or external:
            parts.append("\n## Links")
            for lnk in internal[:20]:
                href = lnk.get("href", "") if isinstance(lnk, dict) else str(lnk)
                # Keep relative paths; reject non-http(s) absolute URLs
                if href and not (href.startswith(("/", "#")) or _safe_url(href)):
                    continue
                parts.append(f"- (internal) {href}")
            for lnk in external[:20]:
                href = lnk.get("href", "") if isinstance(lnk, dict) else str(lnk)
                if href and not _safe_url(href):
                    continue
                parts.append(f"- (external) {href}")

    if include_images:
        images = getattr(result, "media", {}).get("images", []) if hasattr(result, "media") else []
        if images:
            parts.append("\n## Images")
            for img in images[:10]:
                src = img.get("src", "") if isinstance(img, dict) else str(img)
                alt = img.get("alt", "") if isinstance(img, dict) else ""
                parts.append(f"- {src}" + (f" ({alt})" if alt else ""))

    return "\n".join(parts)


class Crawl4AIToolkit(Toolkit):
    """Web-crawling toolkit powered by ``crawl4ai``.

    Gives an agent two tools: ``crawl`` and ``crawl_many``.

    Requires ``crawl4ai`` (``pip install crawl4ai``).  A clear error is
    returned from the tool rather than raising at import time if the package
    is not installed.

    Args:
        max_concurrent: Maximum number of URLs fetched in parallel by
            ``crawl_many`` (default 3).
        middleware: Optional sequence of ``ToolMiddleware`` applied to every
            tool in the toolkit.
    """

    __slots__ = ("_max_concurrent",)

    def __init__(
        self,
        *,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self._max_concurrent = max_concurrent

        super().__init__(
            self.crawl(),
            self.crawl_many(),
            name="crawl4ai_toolkit",
            middleware=middleware,
        )

    def crawl(
        self,
        *,
        name: str = "crawl",
        description: str = (
            "Fetch a web page and return its main content as Markdown. "
            "Use this to read articles, documentation, or any public URL. "
            "Optionally include hyperlinks and images found on the page."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        @tool(name=name, description=description, middleware=middleware)
        async def _crawl(
            url: Annotated[str, Field(description="Full URL to fetch (must start with http:// or https://).")],
            include_links: Annotated[
                bool,
                Field(description="If true, append a list of hyperlinks found on the page."),
            ] = False,
            include_images: Annotated[
                bool,
                Field(description="If true, append a list of image URLs found on the page."),
            ] = False,
        ) -> str:
            return await _fetch(url, include_links=include_links, include_images=include_images)

        return _crawl

    def crawl_many(
        self,
        *,
        name: str = "crawl_many",
        description: str = (
            "Fetch multiple web pages concurrently and return their contents. "
            "Results are separated by a divider. "
            "Use this when you need to read several URLs at once."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        # Capture the toolkit default as a non-shadowed name for the closure.
        _default_concurrent = self._max_concurrent

        @tool(name=name, description=description, middleware=middleware)
        async def _crawl_many(
            urls: Annotated[
                list[str],
                Field(description="List of URLs to fetch (1–20 URLs).", min_length=1, max_length=20),
            ],
            max_concurrent: Annotated[
                int | None,
                Field(
                    description=("Maximum number of pages fetched in parallel. Uses the toolkit default when omitted."),
                    ge=1,
                    le=20,
                ),
            ] = None,
        ) -> str:
            concurrency = max_concurrent if max_concurrent is not None else _default_concurrent
            sem = asyncio.Semaphore(concurrency)

            async def bounded(u: str) -> str:
                async with sem:
                    return await _fetch(u, include_links=False, include_images=False)

            results = await asyncio.gather(*[bounded(u) for u in urls])
            return "\n\n---\n\n".join(results)

        return _crawl_many
