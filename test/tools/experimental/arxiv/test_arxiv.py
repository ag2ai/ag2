# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from autogen.import_utils import run_for_optional_imports
from autogen.tools.experimental.arxiv.arxiv import (
    ArxivArticle,
    ArxivClient,
    ArxivSearchTool,
)


class FakeAuthor:
    """Simulates an arxiv result author object."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakeResult:
    """Simulates an arxiv.Result object."""

    def __init__(
        self,
        title: str = "Test Paper",
        authors: list[str] | None = None,
        summary: str = "This is a test summary.",
        published: datetime | None = None,
        entry_id: str = "http://arxiv.org/abs/2301.00001v1",
        pdf_url: str = "http://arxiv.org/pdf/2301.00001v1",
        doi: str | None = None,
        primary_category: str = "cs.AI",
    ) -> None:
        self.title = title
        self.authors = [FakeAuthor(name) for name in (authors or ["Author One", "Author Two"])]
        self.summary = summary
        self.published = published or datetime(2023, 1, 15)
        self.entry_id = entry_id
        self.pdf_url = pdf_url
        self.doi = doi
        self.primary_category = primary_category


@run_for_optional_imports("arxiv", "arxiv")
class TestArxivClient:
    """Test suite for the ArxivClient class."""

    @patch("autogen.tools.experimental.arxiv.arxiv.arxiv")
    def test_search_success(self, mock_arxiv_module: MagicMock) -> None:
        """Test that search returns a list of ArxivArticle on success."""
        fake_result = FakeResult()
        mock_client_instance = MagicMock()
        mock_client_instance.results.return_value = [fake_result]
        mock_arxiv_module.Client.return_value = mock_client_instance
        mock_arxiv_module.SortCriterion.Relevance = "relevance_criterion"
        mock_arxiv_module.Search.return_value = MagicMock()

        client = ArxivClient()
        results = client.search("transformer attention")

        assert len(results) == 1
        assert isinstance(results[0], ArxivArticle)
        assert results[0].title == "Test Paper"
        assert results[0].authors == "Author One, Author Two"
        assert results[0].published == "2023-01"
        assert results[0].primary_category == "cs.AI"

    @patch("autogen.tools.experimental.arxiv.arxiv.arxiv")
    def test_search_empty_results(self, mock_arxiv_module: MagicMock) -> None:
        """Test that search returns an empty list when no results found."""
        mock_client_instance = MagicMock()
        mock_client_instance.results.return_value = []
        mock_arxiv_module.Client.return_value = mock_client_instance
        mock_arxiv_module.SortCriterion.Relevance = "relevance_criterion"
        mock_arxiv_module.Search.return_value = MagicMock()

        client = ArxivClient()
        results = client.search("some obscure query")

        assert results == []

    @patch("autogen.tools.experimental.arxiv.arxiv.arxiv")
    def test_search_invalid_sort_by(self, mock_arxiv_module: MagicMock) -> None:
        """Test that search raises ValueError for invalid sort_by."""
        mock_arxiv_module.Client.return_value = MagicMock()

        client = ArxivClient()
        with pytest.raises(ValueError, match="Invalid sort_by value"):
            client.search("test", sort_by="invalid")

    @patch("autogen.tools.experimental.arxiv.arxiv.arxiv")
    def test_search_truncates_summary(self, mock_arxiv_module: MagicMock) -> None:
        """Test that long summaries are truncated."""
        long_summary = "A" * 5000
        fake_result = FakeResult(summary=long_summary)
        mock_client_instance = MagicMock()
        mock_client_instance.results.return_value = [fake_result]
        mock_arxiv_module.Client.return_value = mock_client_instance
        mock_arxiv_module.SortCriterion.Relevance = "relevance_criterion"
        mock_arxiv_module.Search.return_value = MagicMock()

        client = ArxivClient(truncate=100)
        results = client.search("test")

        assert len(results[0].summary) == 100

    @patch("autogen.tools.experimental.arxiv.arxiv.arxiv")
    def test_search_with_sort_by_date(self, mock_arxiv_module: MagicMock) -> None:
        """Test search with submittedDate sort criterion."""
        fake_result = FakeResult()
        mock_client_instance = MagicMock()
        mock_client_instance.results.return_value = [fake_result]
        mock_arxiv_module.Client.return_value = mock_client_instance
        mock_arxiv_module.SortCriterion.SubmittedDate = "submitted_date_criterion"
        mock_arxiv_module.Search.return_value = MagicMock()

        client = ArxivClient()
        results = client.search("test", sort_by="submittedDate")

        assert len(results) == 1
        mock_arxiv_module.Search.assert_called_once()


@run_for_optional_imports("arxiv", "arxiv")
class TestArxivSearchTool:
    """Test suite for the ArxivSearchTool class."""

    @pytest.fixture
    def tool(self) -> ArxivSearchTool:
        """Provide an ArxivSearchTool instance for testing.

        Returns:
            ArxivSearchTool: A configured tool instance with verbose off.
        """
        return ArxivSearchTool(verbose=False)

    def test_search_success(self, tool: ArxivSearchTool) -> None:
        """Test successful search returns a list of ArxivArticle."""
        fake_articles = [
            ArxivArticle(
                title="Test Paper",
                authors="Author One, Author Two",
                summary="Test summary",
                published="2023-01",
                entry_id="http://arxiv.org/abs/2301.00001v1",
                pdf_url="http://arxiv.org/pdf/2301.00001v1",
                doi="",
                primary_category="cs.AI",
            )
        ]
        with patch.object(tool.arxiv_client, "search", return_value=fake_articles):
            result = tool.search("transformer attention")
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0].title == "Test Paper"
            assert result[0].authors == "Author One, Author Two"

    def test_search_no_results(self, tool: ArxivSearchTool) -> None:
        """Test search with no results returns an informative message."""
        with patch.object(tool.arxiv_client, "search", return_value=[]):
            result = tool.search("some obscure query")
            assert result == "No arXiv search results found"

    def test_search_exception(self, tool: ArxivSearchTool) -> None:
        """Test search handles exceptions gracefully."""
        with patch.object(tool.arxiv_client, "search", side_effect=Exception("connection error")):
            result = tool.search("test query")
            assert isinstance(result, str)
            assert result.startswith("arXiv search failed: ")

    def test_search_truncates_query(self, tool: ArxivSearchTool) -> None:
        """Test that very long queries are truncated."""
        long_query = "A" * 500
        with patch.object(tool.arxiv_client, "search", return_value=[]) as mock_search:
            tool.search(long_query)
            called_query = mock_search.call_args[1]["query"]
            assert len(called_query) == 300

    def test_tool_name_and_description(self, tool: ArxivSearchTool) -> None:
        """Test tool has correct name and description."""
        assert tool.name == "arxiv-search"
        assert "arXiv" in tool.description

    def test_verbose_mode(self) -> None:
        """Test that verbose mode can be enabled."""
        tool = ArxivSearchTool(verbose=True)
        assert tool.verbose is True
