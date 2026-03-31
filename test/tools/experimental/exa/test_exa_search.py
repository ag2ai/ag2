# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from autogen.tools.experimental.exa import ExaSearchTool


class _MockResult:
    """Mock object matching exa_py Result attributes."""

    def __init__(
        self,
        url: str = "",
        title: str = "",
        text: str | None = None,
        published_date: str | None = None,
        author: str | None = None,
        score: float | None = None,
    ):
        self.url = url
        self.title = title
        self.text = text
        self.published_date = published_date
        self.author = author
        self.score = score


class _MockSearchResponse:
    """Mock object matching exa_py SearchResponse."""

    def __init__(self, results: list[_MockResult]):
        self.results = results


class TestExaSearchTool:
    """Test suite for the ExaSearchTool class."""

    @pytest.fixture
    def mock_search_response(self) -> _MockSearchResponse:
        """Provide a mock search response fixture."""
        return _MockSearchResponse(
            results=[
                _MockResult(
                    url="https://example.com/article",
                    title="Example Article",
                    text="This is the article content.",
                    published_date="2025-01-15",
                    author="Test Author",
                    score=0.95,
                ),
            ]
        )

    @pytest.fixture
    def mock_similar_response(self) -> _MockSearchResponse:
        """Provide a mock find_similar response fixture."""
        return _MockSearchResponse(
            results=[
                _MockResult(
                    url="https://similar.com/page",
                    title="Similar Page",
                    text="Content of a similar page.",
                    score=0.88,
                ),
            ]
        )

    @pytest.mark.parametrize("use_internal_auth", [True, False])
    def test_initialization(self, use_internal_auth: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test the initialization of ExaSearchTool."""
        if use_internal_auth:
            monkeypatch.delenv("EXA_API_KEY", raising=False)
            with pytest.raises(ValueError) as exc_info:
                ExaSearchTool(exa_api_key=None)
            assert "exa_api_key must be provided" in str(exc_info.value)
        else:
            tool = ExaSearchTool(exa_api_key="valid_key")
            assert tool.name == "exa_search"
            assert "Exa" in tool.description
            assert tool.exa_api_key == "valid_key"

    def test_tool_schema(self) -> None:
        """Test the validation of the tool's JSON schema."""
        tool = ExaSearchTool(exa_api_key="test_key")
        schema = tool.tool_schema
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "exa_search"

        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert "num_results" in params["properties"]
        assert "search_type" in params["properties"]
        assert "include_text" in params["properties"]
        assert "include_domains" in params["properties"]
        assert "exclude_domains" in params["properties"]
        assert "start_published_date" in params["properties"]
        assert "end_published_date" in params["properties"]
        assert "category" in params["properties"]
        assert params["required"] == ["query"]

    @pytest.mark.parametrize(
        ("search_params", "expected_error"),
        [
            ({"exa_api_key": None}, "exa_api_key must be provided"),
        ],
    )
    def test_parameter_validation(
        self, search_params: dict[str, Any], expected_error: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test validation of tool parameters."""
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        with pytest.raises(ValueError) as exc_info:
            ExaSearchTool(**search_params)
        assert expected_error in str(exc_info.value)

    @patch("autogen.tools.experimental.exa.exa_search._execute_exa_search")
    def test_search_success(self, mock_execute: MagicMock, mock_search_response: _MockSearchResponse) -> None:
        """Test successful execution of a search query."""
        mock_execute.return_value = [
            {
                "title": "Example Article",
                "url": "https://example.com/article",
                "text": "This is the article content.",
                "published_date": "2025-01-15",
                "author": "Test Author",
                "score": 0.95,
            }
        ]

        tool = ExaSearchTool(exa_api_key="valid_test_key")
        result = tool(query="test query", exa_api_key="valid_test_key")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["title"] == "Example Article"
        assert result[0]["url"] == "https://example.com/article"
        assert result[0]["text"] == "This is the article content."
        assert result[0]["published_date"] == "2025-01-15"
        assert result[0]["author"] == "Test Author"
        assert result[0]["score"] == 0.95

        mock_execute.assert_called_once_with(
            query="test query",
            exa_api_key="valid_test_key",
            num_results=10,
            search_type="auto",
            include_text=True,
            include_domains=None,
            exclude_domains=None,
            start_published_date=None,
            end_published_date=None,
            category=None,
        )

    @patch("autogen.tools.experimental.exa.exa_search._execute_exa_search")
    def test_search_with_filters(self, mock_execute: MagicMock) -> None:
        """Test search with domain and date filters."""
        mock_execute.return_value = [
            {
                "title": "Filtered Result",
                "url": "https://arxiv.org/paper",
                "text": "Research content.",
                "score": 0.92,
            }
        ]

        tool = ExaSearchTool(exa_api_key="test_key")
        result = tool(
            query="machine learning",
            exa_api_key="test_key",
            num_results=5,
            search_type="neural",
            include_domains=["arxiv.org"],
            start_published_date="2024-01-01",
            category="research paper",
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["url"] == "https://arxiv.org/paper"

        mock_execute.assert_called_once_with(
            query="machine learning",
            exa_api_key="test_key",
            num_results=5,
            search_type="neural",
            include_text=True,
            include_domains=["arxiv.org"],
            exclude_domains=None,
            start_published_date="2024-01-01",
            end_published_date=None,
            category="research paper",
        )

    @patch("autogen.tools.experimental.exa.exa_search._execute_exa_search")
    def test_search_empty_results(self, mock_execute: MagicMock) -> None:
        """Test search that returns no results."""
        mock_execute.return_value = []

        tool = ExaSearchTool(exa_api_key="test_key")
        result = tool(query="obscure query", exa_api_key="test_key")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_search_invalid_query(self) -> None:
        """Test that an invalid (None) query raises a Pydantic ValidationError."""
        tool = ExaSearchTool(exa_api_key="test_key")
        with pytest.raises(ValidationError) as exc_info:
            tool(query=None, exa_api_key="test_key")  # type: ignore[arg-type]
        assert "Input should be a valid string" in str(exc_info.value)

    @patch("autogen.tools.experimental.exa.exa_search.Exa")
    def test_execute_exa_search_sets_integration_header(self, mock_exa_class: MagicMock) -> None:
        """Test that the x-exa-integration header is set to 'ag2'."""
        mock_client = MagicMock()
        mock_client.headers = {"x-api-key": "test_key"}
        mock_client.search_and_contents.return_value = _MockSearchResponse(results=[])
        mock_exa_class.return_value = mock_client

        from autogen.tools.experimental.exa.exa_search import _execute_exa_search

        _execute_exa_search(query="test", exa_api_key="test_key")

        assert mock_client.headers["x-exa-integration"] == "ag2"
        mock_client.search_and_contents.assert_called_once()

    @patch("autogen.tools.experimental.exa.exa_search.Exa")
    def test_execute_exa_search_formats_results(self, mock_exa_class: MagicMock) -> None:
        """Test that raw API results are formatted correctly."""
        mock_client = MagicMock()
        mock_client.headers = {}
        mock_client.search_and_contents.return_value = _MockSearchResponse(
            results=[
                _MockResult(
                    url="https://example.com",
                    title="Test",
                    text="Content here.",
                    published_date="2025-03-01",
                    author="Author",
                    score=0.9,
                ),
                _MockResult(
                    url="https://example.org",
                    title="Another",
                    text=None,
                    score=0.7,
                ),
            ]
        )
        mock_exa_class.return_value = mock_client

        from autogen.tools.experimental.exa.exa_search import _execute_exa_search

        results = _execute_exa_search(query="test", exa_api_key="key", include_text=True)
        assert len(results) == 2
        assert results[0]["title"] == "Test"
        assert results[0]["text"] == "Content here."
        assert results[0]["published_date"] == "2025-03-01"
        assert results[0]["author"] == "Author"
        assert results[0]["score"] == 0.9
        assert results[1]["title"] == "Another"
        assert "text" not in results[1]

    @patch("autogen.tools.experimental.exa.exa_search.Exa")
    def test_execute_exa_find_similar(self, mock_exa_class: MagicMock) -> None:
        """Test the find_similar helper function."""
        mock_client = MagicMock()
        mock_client.headers = {}
        mock_client.find_similar_and_contents.return_value = _MockSearchResponse(
            results=[
                _MockResult(
                    url="https://similar.com/page",
                    title="Similar Page",
                    text="Similar content.",
                    score=0.88,
                ),
            ]
        )
        mock_exa_class.return_value = mock_client

        from autogen.tools.experimental.exa.exa_search import _execute_exa_find_similar

        results = _execute_exa_find_similar(url="https://example.com", exa_api_key="key")
        assert len(results) == 1
        assert results[0]["title"] == "Similar Page"
        assert results[0]["url"] == "https://similar.com/page"
        assert results[0]["score"] == 0.88

        mock_client.find_similar_and_contents.assert_called_once_with(
            "https://example.com",
            num_results=10,
            text=True,
            exclude_source_domain=True,
        )
        assert mock_client.headers["x-exa-integration"] == "ag2"
