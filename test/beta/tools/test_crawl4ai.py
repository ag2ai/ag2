# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Crawl4AIToolkit — mock-based so crawl4ai need not be installed."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from autogen.beta import Context
from autogen.beta.tools.toolkits.crawl4ai import Crawl4AIToolkit, _fetch

# ---------------------------------------------------------------------------
# Helpers — fake crawl4ai module
# ---------------------------------------------------------------------------


def _make_crawl4ai_mock(
    *,
    success: bool = True,
    title: str = "Test Page",
    markdown: str = "Hello, world.",
    links: dict | None = None,
    media: dict | None = None,
    error_message: str = "",
) -> ModuleType:
    """Return a fake ``crawl4ai`` module with a controllable AsyncWebCrawler."""
    result = SimpleNamespace(
        success=success,
        title=title,
        markdown=markdown,
        links=links or {"internal": [], "external": []},
        media=media or {"images": []},
        error_message=error_message,
    )

    class FakeCrawler:
        async def __aenter__(self) -> FakeCrawler:
            return self

        async def __aexit__(self, *_: object) -> None:
            pass

        async def arun(self, url: str) -> SimpleNamespace:
            return result

    mod = ModuleType("crawl4ai")
    mod.AsyncWebCrawler = FakeCrawler  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestCrawl4AIToolkitConstruction:
    @pytest.mark.asyncio
    async def test_default_tool_names(self, async_mock: AsyncMock) -> None:
        toolkit = Crawl4AIToolkit()
        schemas = list(await toolkit.schemas(Context(async_mock)))
        names = {s.function.name for s in schemas}
        assert names == {"crawl", "crawl_many"}

    def test_custom_max_concurrent(self) -> None:
        toolkit = Crawl4AIToolkit(max_concurrent=5)
        assert toolkit._max_concurrent == 5

    def test_importable_from_tools_package(self) -> None:
        from autogen.beta.tools import Crawl4AIToolkit as Alias  # noqa: F401

        assert Alias is Crawl4AIToolkit


# ---------------------------------------------------------------------------
# _fetch helper — with mocked crawl4ai
# ---------------------------------------------------------------------------


class TestFetchHelper:
    @pytest.mark.asyncio
    async def test_fetch_returns_content(self) -> None:
        fake_mod = _make_crawl4ai_mock(title="My Page", markdown="Some content here.")
        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await _fetch("https://example.com", include_links=False, include_images=False)
        assert "My Page" in result
        assert "Some content here." in result
        assert "https://example.com" in result

    @pytest.mark.asyncio
    async def test_fetch_no_title(self) -> None:
        fake_mod = _make_crawl4ai_mock(title="", markdown="Body only.")
        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await _fetch("https://example.com", include_links=False, include_images=False)
        assert "Body only." in result
        assert result.startswith("URL:")

    @pytest.mark.asyncio
    async def test_fetch_failed_result(self) -> None:
        fake_mod = _make_crawl4ai_mock(success=False, error_message="timeout")
        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await _fetch("https://example.com", include_links=False, include_images=False)
        assert "fetch failed" in result
        assert "timeout" in result

    @pytest.mark.asyncio
    async def test_fetch_missing_crawl4ai(self) -> None:
        # Remove crawl4ai from sys.modules to simulate not-installed
        original = sys.modules.pop("crawl4ai", None)
        try:
            result = await _fetch("https://example.com", include_links=False, include_images=False)
        finally:
            if original is not None:
                sys.modules["crawl4ai"] = original
        assert "not installed" in result

    @pytest.mark.asyncio
    async def test_fetch_exception_during_crawl(self) -> None:
        class ErrorCrawler:
            async def __aenter__(self) -> ErrorCrawler:
                return self

            async def __aexit__(self, *_: object) -> None:
                pass

            async def arun(self, url: str) -> None:
                raise RuntimeError("network error")

        mod = ModuleType("crawl4ai")
        mod.AsyncWebCrawler = ErrorCrawler  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"crawl4ai": mod}):
            result = await _fetch("https://example.com", include_links=False, include_images=False)
        assert "error fetching" in result
        assert "network error" in result

    @pytest.mark.asyncio
    async def test_fetch_include_links(self) -> None:
        links = {
            "internal": [{"href": "/about"}, {"href": "/contact"}],
            "external": [{"href": "https://other.com"}],
        }
        fake_mod = _make_crawl4ai_mock(markdown="Content.", links=links)
        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await _fetch("https://example.com", include_links=True, include_images=False)
        assert "/about" in result
        assert "https://other.com" in result
        assert "## Links" in result

    @pytest.mark.asyncio
    async def test_fetch_include_images(self) -> None:
        media = {"images": [{"src": "https://example.com/img.png", "alt": "logo"}]}
        fake_mod = _make_crawl4ai_mock(markdown="Content.", media=media)
        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await _fetch("https://example.com", include_links=False, include_images=True)
        assert "img.png" in result
        assert "logo" in result
        assert "## Images" in result

    @pytest.mark.asyncio
    async def test_fetch_empty_markdown(self) -> None:
        fake_mod = _make_crawl4ai_mock(markdown="")
        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await _fetch("https://example.com", include_links=False, include_images=False)
        assert "no content extracted" in result

    @pytest.mark.asyncio
    async def test_fetch_rejects_non_http_scheme(self) -> None:
        result = await _fetch("javascript:alert(1)", include_links=False, include_images=False)
        assert "invalid URL" in result

    @pytest.mark.asyncio
    async def test_fetch_rejects_file_scheme(self) -> None:
        result = await _fetch("file:///etc/passwd", include_links=False, include_images=False)
        assert "invalid URL" in result

    @pytest.mark.asyncio
    async def test_fetch_links_filters_unsafe_hrefs(self) -> None:
        links = {
            "internal": [{"href": "javascript:void(0)"}, {"href": "/safe"}],
            "external": [{"href": "https://other.com"}, {"href": "data:text/html,<h1>xss</h1>"}],
        }
        fake_mod = _make_crawl4ai_mock(markdown="Content.", links=links)
        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await _fetch("https://example.com", include_links=True, include_images=False)
        assert "/safe" in result
        assert "https://other.com" in result
        assert "javascript:" not in result
        assert "data:" not in result


# ---------------------------------------------------------------------------
# crawl tool
# ---------------------------------------------------------------------------


class TestCrawlTool:
    @pytest.mark.asyncio
    async def test_crawl_calls_fetch(self) -> None:
        fake_mod = _make_crawl4ai_mock(markdown="Article text.")
        toolkit = Crawl4AIToolkit()
        crawl_tool = toolkit.crawl()
        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await crawl_tool.model.call(url="https://example.com")
        assert "Article text." in result

    @pytest.mark.asyncio
    async def test_crawl_with_links(self) -> None:
        links = {"internal": [{"href": "/page"}], "external": []}
        fake_mod = _make_crawl4ai_mock(markdown="Body.", links=links)
        toolkit = Crawl4AIToolkit()
        crawl_tool = toolkit.crawl()
        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await crawl_tool.model.call(url="https://example.com", include_links=True)
        assert "/page" in result

    @pytest.mark.asyncio
    async def test_crawl_default_no_links_no_images(self) -> None:
        links = {"internal": [{"href": "/hidden"}], "external": []}
        fake_mod = _make_crawl4ai_mock(markdown="Body.", links=links)
        toolkit = Crawl4AIToolkit()
        crawl_tool = toolkit.crawl()
        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await crawl_tool.model.call(url="https://example.com")
        assert "/hidden" not in result


# ---------------------------------------------------------------------------
# crawl_many tool
# ---------------------------------------------------------------------------


class TestCrawlManyTool:
    @pytest.mark.asyncio
    async def test_crawl_many_returns_multiple_results(self) -> None:
        fake_mod = _make_crawl4ai_mock(markdown="Page content.")
        toolkit = Crawl4AIToolkit()
        many_tool = toolkit.crawl_many()
        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await many_tool.model.call(urls=["https://a.com", "https://b.com", "https://c.com"])
        assert result.count("---") == 2
        assert result.count("Page content.") == 3

    @pytest.mark.asyncio
    async def test_crawl_many_single_url(self) -> None:
        fake_mod = _make_crawl4ai_mock(markdown="Solo.")
        toolkit = Crawl4AIToolkit()
        many_tool = toolkit.crawl_many()
        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await many_tool.model.call(urls=["https://example.com"])
        assert "Solo." in result
        assert "---" not in result

    @pytest.mark.asyncio
    async def test_crawl_many_respects_toolkit_default_concurrency(self) -> None:
        call_count = 0
        active = 0
        peak = 0

        class CountingCrawler:
            async def __aenter__(self) -> CountingCrawler:
                return self

            async def __aexit__(self, *_: object) -> None:
                pass

            async def arun(self, url: str) -> SimpleNamespace:
                nonlocal call_count, active, peak
                call_count += 1
                active += 1
                peak = max(peak, active)
                import asyncio

                await asyncio.sleep(0)
                active -= 1
                return SimpleNamespace(success=True, title="T", markdown="M", links={}, media={}, error_message="")

        mod = ModuleType("crawl4ai")
        mod.AsyncWebCrawler = CountingCrawler  # type: ignore[attr-defined]

        toolkit = Crawl4AIToolkit(max_concurrent=2)
        many_tool = toolkit.crawl_many()
        with patch.dict(sys.modules, {"crawl4ai": mod}):
            await many_tool.model.call(urls=["https://a.com", "https://b.com", "https://c.com"])
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_crawl_many_per_call_max_concurrent_override(self) -> None:
        fake_mod = _make_crawl4ai_mock(markdown="X.")
        toolkit = Crawl4AIToolkit(max_concurrent=10)
        many_tool = toolkit.crawl_many()
        with patch.dict(sys.modules, {"crawl4ai": fake_mod}):
            result = await many_tool.model.call(urls=["https://a.com", "https://b.com"], max_concurrent=1)
        assert result.count("X.") == 2
