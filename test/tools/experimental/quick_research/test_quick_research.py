# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autogen.import_utils import run_for_optional_imports
from autogen.llm_config import LLMConfig
from autogen.tools.experimental.quick_research import QuickResearchTool
from autogen.tools.experimental.quick_research.quick_research import (
    _clean_html,
    _crawl_and_summarise,
    _reg_dom,
    _research_single_query,
    _split_tokens,
    _tavily_search,
)


@run_for_optional_imports(["tavily", "crawl4ai", "tiktoken", "tldextract", "bs4"], "quick-research")
class TestQuickResearchTool:
    """Test suite for the QuickResearchTool class."""

    # ──────────────────────────── fixtures ────────────────────────────────────

    @pytest.fixture
    def llm_config(self) -> LLMConfig:
        return LLMConfig({"model": "gpt-4o-mini", "api_key": "test_key"})

    @pytest.fixture
    def mock_tavily_response(self) -> dict[str, Any]:
        return {
            "results": [
                {
                    "title": "AG2 Framework Overview",
                    "url": "https://ag2.ai/overview",
                    "content": "AG2 is an open-source framework for AI agents.",
                },
                {
                    "title": "AG2 Documentation",
                    "url": "https://docs.ag2.ai",
                    "content": "Official AG2 documentation site.",
                },
            ]
        }

    @pytest.fixture
    def mock_crawl_html(self) -> str:
        return """
        <html>
        <head><style>body { color: red; }</style></head>
        <body>
            <script>alert('xss')</script>
            <h1>AG2 Framework</h1>
            <p>AG2 is an open-source framework for building AI agents.</p>
            <a href="/docs">Docs</a>
            <a href="https://other-site.com/page">External link</a>
            <img src="photo.jpg" />
        </body>
        </html>
        """

    # ──────────────────────────── initialization ─────────────────────────────

    def test_initialization_missing_tavily_key(self, llm_config: LLMConfig, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing Tavily API key raises ValueError."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(ValueError, match="tavily_api_key must be provided"):
            QuickResearchTool(llm_config=llm_config, tavily_api_key=None)

    def test_initialization_success(self, llm_config: LLMConfig) -> None:
        """Test successful initialization with API keys."""
        tool = QuickResearchTool(llm_config=llm_config, tavily_api_key="test_tavily_key")
        assert tool.name == "quick_research"
        assert "Research multiple queries" in tool.description
        assert tool.tavily_api_key == "test_tavily_key"
        assert callable(tool.func)

    def test_initialization_custom_params(self, llm_config: LLMConfig) -> None:
        """Test initialization with custom parameters."""
        tool = QuickResearchTool(
            llm_config=llm_config,
            tavily_api_key="key",
            num_results_per_query=5,
        )
        assert tool.num_results_per_query == 5

    # ──────────────────────────── tool schema ────────────────────────────────

    def test_tool_schema(self, llm_config: LLMConfig) -> None:
        """Test the tool's JSON schema structure."""
        tool = QuickResearchTool(llm_config=llm_config, tavily_api_key="test_key")
        schema = tool.tool_schema

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "quick_research"

        params = schema["function"]["parameters"]
        assert "queries" in params["properties"]
        assert "queries" in params["required"]

        # Injected dependencies should NOT appear in the schema
        for injected in ["tavily_api_key", "config_list", "num_results_per_query"]:
            assert injected not in params["properties"]

    # ──────────────────────────── HTML cleaning ──────────────────────────────

    def test_clean_html(self, mock_crawl_html: str) -> None:
        """Test HTML cleaning removes scripts, styles, and off-site links."""
        cleaned = _clean_html(mock_crawl_html, "https://ag2.ai")
        assert "alert" not in cleaned, "Scripts should be removed"
        assert "color: red" not in cleaned, "Styles should be removed"
        assert "AG2 Framework" in cleaned, "Text content should be preserved"
        assert "open-source framework" in cleaned, "Paragraph text should be preserved"

    def test_clean_html_preserves_same_site_links(self) -> None:
        """Test that same-site links are preserved."""
        html = '<html><body><a href="https://example.com/page">Link</a></body></html>'
        cleaned = _clean_html(html, "https://example.com")
        assert "Link" in cleaned

    def test_clean_html_removes_media_tags(self) -> None:
        """Test that img, audio, video tags are removed."""
        html = """
        <html><body>
            <p>Text</p>
            <img src="photo.jpg" alt="Photo" />
            <video src="video.mp4"></video>
            <audio src="audio.mp3"></audio>
        </body></html>
        """
        cleaned = _clean_html(html, "https://example.com")
        assert "Text" in cleaned
        assert "photo.jpg" not in cleaned
        assert "video.mp4" not in cleaned

    # ──────────────────────────── domain extraction ──────────────────────────

    def test_reg_dom(self) -> None:
        """Test registered domain extraction."""
        assert _reg_dom("www.example.com") == "example.com"
        assert _reg_dom("sub.example.co.uk") == "example.co.uk"
        assert _reg_dom("docs.ag2.ai") == "ag2.ai"

    # ──────────────────────────── token splitting ────────────────────────────

    def test_split_tokens_short_text(self) -> None:
        """Test that short text produces a single chunk."""
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        chunks = _split_tokens("Hello world, this is a short text.", enc)
        assert len(chunks) == 1
        assert "Hello world" in chunks[0]

    def test_split_tokens_long_text(self) -> None:
        """Test that long text is split into multiple chunks."""
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        # Create text that exceeds MAX_CHUNK_TOKENS
        text = "hello " * 20_000
        chunks = _split_tokens(text, enc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    # ──────────────────────────── tavily search ──────────────────────────────

    @patch("autogen.tools.experimental.quick_research.quick_research.TavilyClient")
    def test_tavily_search(self, mock_client_cls: MagicMock, mock_tavily_response: dict[str, Any]) -> None:
        """Test Tavily search returns formatted results."""
        mock_client = MagicMock()
        mock_client.search.return_value = mock_tavily_response
        mock_client_cls.return_value = mock_client

        results = _tavily_search("AG2 framework", "test_key", num_results=2)

        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["title"] == "AG2 Framework Overview"
        assert results[0]["url"] == "https://ag2.ai/overview"
        mock_client.search.assert_called_once()

    @patch("autogen.tools.experimental.quick_research.quick_research.TavilyClient")
    def test_tavily_search_empty_results(self, mock_client_cls: MagicMock) -> None:
        """Test Tavily search handles empty results."""
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        mock_client_cls.return_value = mock_client

        results = _tavily_search("nonexistent query", "test_key")
        assert results == []

    # ──────────────────────────── crawl + summarise ──────────────────────────

    @pytest.mark.asyncio
    async def test_crawl_and_summarise_timeout(self) -> None:
        """Test that crawl timeout returns None page_content."""
        import asyncio

        async def slow_crawler(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(100)

        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        config_list = [{"model": "gpt-4o-mini", "api_key": "test_key"}]

        with patch("autogen.tools.experimental.quick_research.quick_research.AsyncWebCrawler") as mock_crawler_cls:
            mock_crawler = AsyncMock()
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock(return_value=False)
            mock_crawler.arun = slow_crawler
            mock_crawler_cls.return_value = mock_crawler

            result = await _crawl_and_summarise(
                "https://example.com",
                config_list,
                enc,
                summarise=True,
            )

        assert result["url"] == "https://example.com"
        assert result["page_content"] is None

    @pytest.mark.asyncio
    async def test_crawl_and_summarise_no_summarise(self) -> None:
        """Test crawl without summarization returns raw text."""
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        config_list = [{"model": "gpt-4o-mini", "api_key": "test_key"}]

        mock_result = MagicMock()
        mock_result.html = "<html><body><p>Raw page content here.</p></body></html>"

        with patch("autogen.tools.experimental.quick_research.quick_research.AsyncWebCrawler") as mock_crawler_cls:
            mock_crawler = AsyncMock()
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock(return_value=False)
            mock_crawler.arun = AsyncMock(return_value=mock_result)
            mock_crawler_cls.return_value = mock_crawler

            result = await _crawl_and_summarise(
                "https://example.com",
                config_list,
                enc,
                summarise=False,
            )

        assert result["url"] == "https://example.com"
        assert result["page_content"] is not None
        assert "Raw page content here" in result["page_content"]

    # ──────────────────────────── research single query ──────────────────────

    @pytest.mark.asyncio
    async def test_research_single_query(self) -> None:
        """Test single query research pipeline with mocked search and crawl."""
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        config_list = [{"model": "gpt-4o-mini", "api_key": "test_key"}]

        mock_search_results = [
            {"title": "Result 1", "url": "https://example.com/1"},
            {"title": "Result 2", "url": "https://example.com/2"},
        ]

        mock_crawl_result = {
            "url": "https://example.com/1",
            "fetched_at": "2025-01-01T00:00:00+00:00",
            "page_content": "Summary of result 1",
            "timing": {"crawl": 1.0, "clean": 0.1, "summarize": 0.5, "total": 1.6},
        }

        with (
            patch(
                "autogen.tools.experimental.quick_research.quick_research._tavily_search",
                return_value=mock_search_results,
            ),
            patch(
                "autogen.tools.experimental.quick_research.quick_research._crawl_and_summarise",
                return_value=mock_crawl_result,
            ),
        ):
            result = await _research_single_query(
                "test query",
                tavily_api_key="test_key",
                config_list=config_list,
                enc=enc,
                num_results=2,
            )

        assert result["query"] == "test query"
        assert len(result["sources"]) == 2
        assert result["sources"][0]["title"] == "Result 1"
        assert result["sources"][0]["summary"] == "Summary of result 1"

    # ──────────────────────────── full tool invocation ────────────────────────

    @pytest.mark.asyncio
    async def test_quick_research_empty_queries(self, llm_config: LLMConfig) -> None:
        """Test that empty queries return empty list."""
        tool = QuickResearchTool(llm_config=llm_config, tavily_api_key="test_key")
        result = await tool(queries=[])
        assert result == "[]"

    @pytest.mark.asyncio
    async def test_quick_research_caps_queries(self, llm_config: LLMConfig) -> None:
        """Test that queries are capped at MAX_QUERIES."""
        mock_research_result = {"query": "q", "sources": []}

        mock_research = AsyncMock(return_value=mock_research_result)

        with patch(
            "autogen.tools.experimental.quick_research.quick_research._research_single_query",
            mock_research,
        ):
            tool = QuickResearchTool(llm_config=llm_config, tavily_api_key="test_key")
            queries = [f"query {i}" for i in range(10)]
            result_json = await tool(queries=queries)
            results = json.loads(result_json)

            # Should cap at MAX_QUERIES (5)
            assert len(results) <= 5
            assert mock_research.call_count <= 5

    @pytest.mark.asyncio
    async def test_quick_research_full_pipeline(self, llm_config: LLMConfig) -> None:
        """Test full pipeline with mocked Tavily and crawl4ai."""
        mock_search_results = [{"title": "Test Page", "url": "https://example.com/test"}]
        mock_crawl_result = {
            "url": "https://example.com/test",
            "fetched_at": "2025-01-01T00:00:00+00:00",
            "page_content": "Summarized content about the topic.",
            "timing": {"crawl": 1.0, "clean": 0.1, "summarize": 0.5, "total": 1.6},
        }

        with (
            patch(
                "autogen.tools.experimental.quick_research.quick_research._tavily_search",
                return_value=mock_search_results,
            ),
            patch(
                "autogen.tools.experimental.quick_research.quick_research._crawl_and_summarise",
                return_value=mock_crawl_result,
            ),
        ):
            tool = QuickResearchTool(llm_config=llm_config, tavily_api_key="test_key")
            result_json = await tool(queries=["What is AG2?"])

        results = json.loads(result_json)
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["query"] == "What is AG2?"
        assert len(results[0]["sources"]) == 1
        assert results[0]["sources"][0]["title"] == "Test Page"
        assert results[0]["sources"][0]["summary"] == "Summarized content about the topic."

    @pytest.mark.asyncio
    async def test_quick_research_handles_crawl_failures(self, llm_config: LLMConfig) -> None:
        """Test that failed crawls are gracefully skipped."""
        mock_search_results = [
            {"title": "Good Page", "url": "https://example.com/good"},
            {"title": "Bad Page", "url": "https://example.com/bad"},
        ]

        async def mock_crawl_side_effect(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
            if "bad" in url:
                raise ConnectionError("Failed to crawl")
            return {
                "url": url,
                "fetched_at": "2025-01-01T00:00:00+00:00",
                "page_content": "Good content",
                "timing": {"crawl": 1.0, "clean": 0.1, "summarize": 0.5, "total": 1.6},
            }

        with (
            patch(
                "autogen.tools.experimental.quick_research.quick_research._tavily_search",
                return_value=mock_search_results,
            ),
            patch(
                "autogen.tools.experimental.quick_research.quick_research._crawl_and_summarise",
                side_effect=mock_crawl_side_effect,
            ),
        ):
            tool = QuickResearchTool(llm_config=llm_config, tavily_api_key="test_key")
            result_json = await tool(queries=["test query"])

        results = json.loads(result_json)
        assert len(results) == 1
        # Only the good page should be in sources
        assert len(results[0]["sources"]) == 1
        assert results[0]["sources"][0]["title"] == "Good Page"
