# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import requests

from autogen.tools.experimental.semantic_scholar.semantic_scholar import (
    Paper,
    SemanticScholarClient,
    SemanticScholarSearchTool,
    _format_paper,
    _parse_paper,
)


# Sample raw paper data matching the Semantic Scholar API response format.
SAMPLE_RAW_PAPER = {
    "paperId": "abc123",
    "title": "Attention Is All You Need",
    "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
    "authors": [
        {"authorId": "1", "name": "Ashish Vaswani"},
        {"authorId": "2", "name": "Noam Shazeer"},
    ],
    "year": 2017,
    "citationCount": 90000,
    "url": "https://www.semanticscholar.org/paper/abc123",
    "externalIds": {"DOI": "10.5555/3295222.3295349", "ArXiv": "1706.03762", "CorpusId": "215416146"},
}

SAMPLE_RAW_PAPER_MINIMAL = {
    "paperId": "def456",
    "title": "A Minimal Paper",
    "abstract": None,
    "authors": None,
    "year": None,
    "citationCount": None,
    "url": None,
    "externalIds": None,
}


class TestParsePaper:
    """Test suite for the _parse_paper helper function."""

    def test_parse_full_paper(self) -> None:
        paper = _parse_paper(SAMPLE_RAW_PAPER)
        assert paper.paper_id == "abc123"
        assert paper.title == "Attention Is All You Need"
        assert len(paper.authors) == 2
        assert "Ashish Vaswani" in paper.authors
        assert paper.year == 2017
        assert paper.citation_count == 90000
        assert paper.external_ids["DOI"] == "10.5555/3295222.3295349"

    def test_parse_minimal_paper(self) -> None:
        paper = _parse_paper(SAMPLE_RAW_PAPER_MINIMAL)
        assert paper.paper_id == "def456"
        assert paper.title == "A Minimal Paper"
        assert paper.abstract is None
        assert paper.authors == []
        assert paper.year is None
        assert paper.citation_count is None
        assert paper.external_ids == {}


class TestFormatPaper:
    """Test suite for the _format_paper helper function."""

    def test_format_full_paper(self) -> None:
        paper = _parse_paper(SAMPLE_RAW_PAPER)
        formatted = _format_paper(paper)
        assert "Title: Attention Is All You Need" in formatted
        assert "Authors: Ashish Vaswani, Noam Shazeer" in formatted
        assert "Year: 2017" in formatted
        assert "Citations: 90000" in formatted
        assert "Abstract:" in formatted
        assert "URL:" in formatted
        assert "External IDs:" in formatted

    def test_format_minimal_paper(self) -> None:
        paper = _parse_paper(SAMPLE_RAW_PAPER_MINIMAL)
        formatted = _format_paper(paper)
        assert "Title: A Minimal Paper" in formatted
        assert "Authors:" not in formatted
        assert "Year:" not in formatted
        assert "Abstract:" not in formatted


class TestSemanticScholarClient:
    """Test suite for the SemanticScholarClient class."""

    @patch("autogen.tools.experimental.semantic_scholar.semantic_scholar.requests.get")
    def test_search_papers_success(self, mock_get: MagicMock) -> None:
        fake_json = {"data": [SAMPLE_RAW_PAPER]}
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = fake_json
        mock_get.return_value = mock_response

        client = SemanticScholarClient()
        results = client.search_papers("attention transformers")
        assert len(results) == 1
        assert results[0]["title"] == "Attention Is All You Need"

    @patch("autogen.tools.experimental.semantic_scholar.semantic_scholar.requests.get")
    def test_search_papers_empty(self, mock_get: MagicMock) -> None:
        fake_json = {"data": []}
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = fake_json
        mock_get.return_value = mock_response

        client = SemanticScholarClient()
        results = client.search_papers("nonexistent query xyz")
        assert results == []

    @patch("autogen.tools.experimental.semantic_scholar.semantic_scholar.requests.get")
    def test_search_papers_http_error(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("HTTP Error")
        mock_get.return_value = mock_response

        client = SemanticScholarClient()
        with pytest.raises(requests.HTTPError):
            client.search_papers("test query")

    @patch("autogen.tools.experimental.semantic_scholar.semantic_scholar.requests.get")
    def test_get_paper_success(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = SAMPLE_RAW_PAPER
        mock_get.return_value = mock_response

        client = SemanticScholarClient()
        result = client.get_paper("abc123")
        assert result["title"] == "Attention Is All You Need"

    @patch("autogen.tools.experimental.semantic_scholar.semantic_scholar.requests.get")
    def test_get_paper_not_found(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        client = SemanticScholarClient()
        with pytest.raises(requests.HTTPError):
            client.get_paper("nonexistent_id")

    def test_api_key_header(self) -> None:
        client = SemanticScholarClient(api_key="test-key-123")
        assert client.headers["x-api-key"] == "test-key-123"

    def test_no_api_key_header(self) -> None:
        client = SemanticScholarClient()
        assert "x-api-key" not in client.headers


class TestSemanticScholarSearchTool:
    """Test suite for the SemanticScholarSearchTool class."""

    @pytest.fixture
    def tool(self) -> SemanticScholarSearchTool:
        """Provide a SemanticScholarSearchTool instance for testing.

        Returns:
            SemanticScholarSearchTool: A configured tool instance with verbose off.
        """
        return SemanticScholarSearchTool(verbose=False)

    def test_search_papers_success(self, tool: SemanticScholarSearchTool) -> None:
        with patch.object(
            tool.client,
            "search_papers",
            return_value=[SAMPLE_RAW_PAPER],
        ):
            result = tool.search_papers("attention mechanism")
            assert isinstance(result, list)
            assert len(result) == 1
            assert "Attention Is All You Need" in result[0]
            assert "Ashish Vaswani" in result[0]

    def test_search_papers_no_results(self, tool: SemanticScholarSearchTool) -> None:
        with patch.object(tool.client, "search_papers", return_value=[]):
            result = tool.search_papers("completely nonexistent paper xyz123")
            assert result == "No papers found matching the query on Semantic Scholar"

    def test_search_papers_exception(self, tool: SemanticScholarSearchTool) -> None:
        with patch.object(tool.client, "search_papers", side_effect=Exception("API timeout")):
            result = tool.search_papers("test query")
            assert isinstance(result, str)
            assert result.startswith("Semantic Scholar search failed:")

    def test_get_paper_success(self, tool: SemanticScholarSearchTool) -> None:
        with patch.object(
            tool.client,
            "get_paper",
            return_value=SAMPLE_RAW_PAPER,
        ):
            result = tool.get_paper("abc123")
            assert isinstance(result, str)
            assert "Attention Is All You Need" in result

    def test_get_paper_exception(self, tool: SemanticScholarSearchTool) -> None:
        with patch.object(tool.client, "get_paper", side_effect=Exception("404 Not Found")):
            result = tool.get_paper("nonexistent_id")
            assert isinstance(result, str)
            assert result.startswith("Semantic Scholar paper lookup failed:")

    def test_tool_name_and_description(self, tool: SemanticScholarSearchTool) -> None:
        assert tool.name == "semantic-scholar-search"
        assert "Semantic Scholar" in tool.description

    def test_max_results_capped(self) -> None:
        tool = SemanticScholarSearchTool(max_results=500)
        assert tool.max_results == 100

    def test_query_truncation(self, tool: SemanticScholarSearchTool) -> None:
        long_query = "a" * 500
        with patch.object(tool.client, "search_papers", return_value=[]) as mock_search:
            tool.search_papers(long_query)
            called_query = mock_search.call_args[1]["query"]
            assert len(called_query) == 300
