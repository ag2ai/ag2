# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import pytest

from autogen.import_utils import optional_import_block, skip_on_missing_imports
from autogen.tools.experimental.crawl4ai import Crawl4AITool

from ....conftest import Credentials

with optional_import_block():
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
    from crawl4ai.extraction_strategy import LLMExtractionStrategy


@pytest.mark.crawl4ai  # todo: remove me after we merge the PR that ads it automatically
@skip_on_missing_imports(["crawl4ai"], "crawl4ai")
class TestCrawl4AITool:
    def _use_imports(self) -> None:
        self._AsyncWebCrawler = AsyncWebCrawler
        self._BrowserConfig = BrowserConfig
        self._CrawlerRunConfig = CrawlerRunConfig
        self._CacheMode = CacheMode
        self._LLMExtractionStrategy = LLMExtractionStrategy

    @pytest.mark.asyncio
    async def test_without_llm(self) -> None:
        tool_without_llm = Crawl4AITool()
        assert isinstance(tool_without_llm, Crawl4AITool)
        assert tool_without_llm.name == "crawl4ai"
        assert tool_without_llm.description == "Crawl a website and extract information."
        assert callable(tool_without_llm.func)
        expected_schema = {
            "function": {
                "description": "Crawl a website and extract information.",
                "name": "crawl4ai",
                "parameters": {
                    "properties": {
                        "url": {"description": "The url to crawl and extract information from.", "type": "string"}
                    },
                    "required": ["url"],
                    "type": "object",
                },
            },
            "type": "function",
        }
        assert tool_without_llm.tool_schema == expected_schema

        result = await tool_without_llm(url="https://docs.ag2.ai/docs/Home")
        assert isinstance(result, str)

    def test_get_crawl_config(self, mock_credentials: Credentials) -> None:
        config = Crawl4AITool._get_crawl_config(mock_credentials.llm_config)
        assert isinstance(config, CrawlerRunConfig)
        assert config.extraction_strategy.provider == f"openai/{mock_credentials.model}"

    @pytest.mark.openai
    @pytest.mark.asyncio
    async def test_with_llm(self, credentials_gpt_4o_mini: Credentials) -> None:
        tool_with_llm = Crawl4AITool(llm_config=credentials_gpt_4o_mini.llm_config)
        assert isinstance(tool_with_llm, Crawl4AITool)

        result = await tool_with_llm(url="https://docs.ag2.ai/docs/Home")
        assert isinstance(result, str)
