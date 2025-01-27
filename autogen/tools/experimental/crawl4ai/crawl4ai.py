# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any

from ....import_utils import optional_import_block
from ... import Tool

with optional_import_block():
    from crawl4ai import AsyncWebCrawler

__all__ = ["Crawl4AITool"]


class Crawl4AITool(Tool):
    def __init__(self) -> None:
        async def crawl4ai(
            url: Annotated[str, "The url to crawl and extract information from."],
        ) -> Any:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(
                    url=url,
                )
                return result.markdown

        super().__init__(
            name="crawl4ai",
            description="Crawl a website and extract information.",
            func_or_tool=crawl4ai,
        )
