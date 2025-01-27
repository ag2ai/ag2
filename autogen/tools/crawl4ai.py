# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from crawl4ai import AsyncWebCrawler

from . import Tool


class Crawl4AI(Tool):
    def __init__(self) -> None:
        async def crawl4ai(url: str) -> Any:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(
                    url=url,
                )
                return result.markdown()

        super().__init__(
            name="crawl4ai",
            description="Crawl a website and extract information.",
            func_or_tool=crawl4ai,
        )
