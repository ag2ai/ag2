# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import pytest

from autogen.import_utils import optional_import_block, skip_on_missing_imports

with optional_import_block():
    from crawl4ai import AsyncWebCrawler

from autogen.tools.experimental.crawl4ai import Crawl4AITool


@pytest.mark.crawl4ai  # todo: remove me after we merge the PR that ads it automatically
@skip_on_missing_imports(["crawl4ai"], "crawl4ai")
class TestCrawl4AITool:
    def _use_imports(self) -> None:
        self._AsyncWebCrawler = AsyncWebCrawler

    @pytest.mark.asyncio
    async def test__init__(self) -> None:
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
