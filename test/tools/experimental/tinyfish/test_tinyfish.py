# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import Mock, patch

import pytest

from autogen.tools.experimental.tinyfish import TinyFishTool


class TestTinyFishTool:
    """Test suite for the TinyFishTool class."""

    @pytest.fixture
    def mock_response(self) -> Mock:
        """Provide a mock TinyFish response fixture."""
        response = Mock()
        response.status = "COMPLETED"
        response.result = {
            "company_name": "Acme Corp",
            "description": "A technology company",
        }
        response.error = None
        return response

    @pytest.mark.parametrize("use_internal_auth", [True, False])
    def test_initialization(self, use_internal_auth: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test the initialization of TinyFishTool."""
        if use_internal_auth:
            monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
            with pytest.raises(ValueError) as exc_info:
                TinyFishTool(tinyfish_api_key=None)
            assert "tinyfish_api_key must be provided" in str(exc_info.value)
        else:
            tool = TinyFishTool(tinyfish_api_key="valid_key")
            assert tool.name == "tinyfish_scrape"
            assert "TinyFish" in tool.description
            assert tool.tinyfish_api_key == "valid_key"

    def test_tool_schema(self) -> None:
        """Test the validation of the tool's JSON schema."""
        tool = TinyFishTool(tinyfish_api_key="test_key")
        expected_schema = {
            "function": {
                "name": "tinyfish_scrape",
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to scrape."},
                        "goal": {
                            "type": "string",
                            "description": "A natural language description of what information to extract from the page.",
                        },
                    },
                    "required": ["url", "goal"],
                },
            },
            "type": "function",
        }
        assert tool.tool_schema == expected_schema

    @pytest.mark.parametrize(
        ("init_params", "expected_error"),
        [
            ({"tinyfish_api_key": None}, "tinyfish_api_key must be provided"),
        ],
    )
    def test_parameter_validation(
        self, init_params: dict[str, Any], expected_error: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test validation of tool parameters."""
        monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
        with pytest.raises(ValueError) as exc_info:
            TinyFishTool(**init_params)
        assert expected_error in str(exc_info.value)

    @patch("autogen.tools.experimental.tinyfish.tinyfish_tool._execute_tinyfish_scrape")
    def test_execute_scrape_success(self, mock_execute: Mock) -> None:
        """Test successful execution of a TinyFish scrape."""
        mock_execute.return_value = {
            "company_name": "Acme Corp",
            "description": "A technology company",
        }

        tool = TinyFishTool(tinyfish_api_key="valid_test_key")
        result = tool(url="https://example.com", goal="Extract company info", tinyfish_api_key="valid_test_key")

        assert isinstance(result, dict)
        assert result["url"] == "https://example.com"
        assert result["goal"] == "Extract company info"
        assert result["data"]["company_name"] == "Acme Corp"

        mock_execute.assert_called_once_with(
            url="https://example.com",
            goal="Extract company info",
            tinyfish_api_key="valid_test_key",
        )

    @patch("autogen.tools.experimental.tinyfish.tinyfish_tool._execute_tinyfish_scrape")
    def test_execute_scrape_error(self, mock_execute: Mock) -> None:
        """Test that errors are handled gracefully."""
        mock_execute.side_effect = Exception("Connection failed")

        tool = TinyFishTool(tinyfish_api_key="test_key")
        result = tool(url="https://example.com", goal="Extract data", tinyfish_api_key="test_key")

        assert isinstance(result, dict)
        assert result["url"] == "https://example.com"
        assert "error" in result
        assert "Connection failed" in result["error"]

    @patch("autogen.tools.experimental.tinyfish.tinyfish_tool._execute_tinyfish_scrape")
    def test_execute_scrape_empty_result(self, mock_execute: Mock) -> None:
        """Test handling of empty/no results."""
        mock_execute.return_value = {"status": "no_result", "error": "No result returned."}

        tool = TinyFishTool(tinyfish_api_key="test_key")
        result = tool(url="https://example.com", goal="Extract data", tinyfish_api_key="test_key")

        assert isinstance(result, dict)
        assert result["data"]["status"] == "no_result"
