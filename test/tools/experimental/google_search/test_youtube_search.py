# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict
from unittest.mock import patch

import pytest

from autogen import AssistantAgent
from autogen.import_utils import run_for_optional_imports
from autogen.tools.experimental import YoutubeSearchTool
from autogen.tools.experimental.google_search.youtube_search import (
    _youtube_search,
)

from ....conftest import Credentials


class TestYoutubeSearchTool:
    def test_init(self) -> None:
        youtube_search_tool = YoutubeSearchTool(youtube_api_key="api_key")

        assert youtube_search_tool.name == "youtube_search"
        assert (
            youtube_search_tool.description
            == "Search for YouTube videos based on a query, optionally including detailed information."
        )
        assert youtube_search_tool.youtube_api_key == "api_key"

    def test_init_no_api_key(self) -> None:
        with pytest.raises(ValueError, match="youtube_api_key must be provided"):
            YoutubeSearchTool(youtube_api_key=None)

    @pytest.fixture
    def mock_search_response(self) -> Dict[str, Any]:
        return {
            "items": [
                {
                    "id": {"kind": "youtube#video", "videoId": "video1"},
                    "snippet": {
                        "title": "Test Video 1",
                        "description": "This is test video 1",
                        "publishedAt": "2023-01-01T00:00:00Z",
                        "channelTitle": "Test Channel 1",
                    },
                },
                {
                    "id": {"kind": "youtube#video", "videoId": "video2"},
                    "snippet": {
                        "title": "Test Video 2",
                        "description": "This is test video 2",
                        "publishedAt": "2023-01-02T00:00:00Z",
                        "channelTitle": "Test Channel 2",
                    },
                },
            ]
        }

    @pytest.fixture
    def mock_video_details(self) -> Dict[str, Any]:
        return {
            "items": [
                {
                    "id": "video1",
                    "snippet": {
                        "title": "Test Video 1",
                        "description": "This is test video 1",
                        "publishedAt": "2023-01-01T00:00:00Z",
                        "channelTitle": "Test Channel 1",
                    },
                    "contentDetails": {"duration": "PT10M30S", "definition": "hd"},
                    "statistics": {"viewCount": "1000", "likeCount": "100", "commentCount": "50"},
                },
                {
                    "id": "video2",
                    "snippet": {
                        "title": "Test Video 2",
                        "description": "This is test video 2",
                        "publishedAt": "2023-01-02T00:00:00Z",
                        "channelTitle": "Test Channel 2",
                    },
                    "contentDetails": {"duration": "PT5M15S", "definition": "sd"},
                    "statistics": {"viewCount": "500", "likeCount": "50", "commentCount": "25"},
                },
            ]
        }

    def test_youtube_search_basic(self, mock_search_response: Dict[str, Any]) -> None:
        with (
            patch(
                "autogen.tools.experimental.google_search.youtube_search._execute_search_query",
                return_value=mock_search_response,
            ),
            patch(
                "autogen.tools.experimental.google_search.youtube_search._get_video_details",
                return_value={"items": []},
            ),
        ):
            results = _youtube_search(
                query="test query",
                youtube_api_key="api_key",
                max_results=2,
                include_video_details=False,
            )

            assert len(results) == 2
            assert results[0]["title"] == "Test Video 1"
            assert results[0]["description"] == "This is test video 1"
            assert results[0]["url"] == "https://www.youtube.com/watch?v=video1"
            assert results[1]["title"] == "Test Video 2"

    def test_youtube_search_with_details(
        self, mock_search_response: Dict[str, Any], mock_video_details: Dict[str, Any]
    ) -> None:
        with (
            patch(
                "autogen.tools.experimental.google_search.youtube_search._execute_search_query",
                return_value=mock_search_response,
            ),
            patch(
                "autogen.tools.experimental.google_search.youtube_search._get_video_details",
                return_value=mock_video_details,
            ),
        ):
            results = _youtube_search(
                query="test query",
                youtube_api_key="api_key",
                max_results=2,
                include_video_details=True,
            )

            assert len(results) == 2
            assert results[0]["title"] == "Test Video 1"
            assert results[0]["viewCount"] == "1000"
            assert results[0]["likeCount"] == "100"
            assert results[0]["duration"] == "PT10M30S"

    def _test_end_to_end(
        self,
        youtube_search_tool: YoutubeSearchTool,
        credentials: Credentials,
        expected_search_result: Dict[str, Any],
        expected_details_result: Dict[str, Any],
    ) -> None:
        assistant = AssistantAgent(
            name="assistant",
            llm_config=credentials.llm_config,
        )

        youtube_search_tool.register_for_llm(assistant)

        with (
            patch(
                "autogen.tools.experimental.google_search.youtube_search._execute_search_query",
                return_value=expected_search_result,
            ) as mock_search,
            patch(
                "autogen.tools.experimental.google_search.youtube_search._get_video_details",
                return_value=expected_details_result,
            ) as mock_details,
        ):
            assistant.run(
                "Find YouTube videos about machine learning",
                tools=assistant.tools,
                max_turns=3,
                user_input=False,
            )
            assert mock_search.called
            assert mock_details.called

    @run_for_optional_imports("openai", "openai")
    def test_end_to_end_openai(
        self,
        credentials_gpt_4o_mini: Credentials,
        mock_search_response: Dict[str, Any],
        mock_video_details: Dict[str, Any],
    ) -> None:
        youtube_search_tool = YoutubeSearchTool(youtube_api_key="api_key")
        self._test_end_to_end(
            youtube_search_tool=youtube_search_tool,
            credentials=credentials_gpt_4o_mini,
            expected_search_result=mock_search_response,
            expected_details_result=mock_video_details,
        )

    @run_for_optional_imports(["google", "vertexai", "PIL", "jsonschema", "jsonschema"], "gemini")
    def test_end_to_end_gemini(
        self,
        credentials_gemini_flash_exp: Credentials,
        mock_search_response: Dict[str, Any],
        mock_video_details: Dict[str, Any],
    ) -> None:
        youtube_search_tool = YoutubeSearchTool(youtube_api_key="api_key")
        self._test_end_to_end(
            youtube_search_tool=youtube_search_tool,
            credentials=credentials_gemini_flash_exp,
            expected_search_result=mock_search_response,
            expected_details_result=mock_video_details,
        )
