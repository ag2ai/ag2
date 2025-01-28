# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any, Optional

from ....import_utils import optional_import_block
from ... import Tool
from ...dependency_injection import Depends, on

with optional_import_block():
    from crawl4ai import AsyncWebCrawler

__all__ = ["Crawl4AITool"]


class Crawl4AITool(Tool):
    def __init__(
        self,
        llm_config: Optional[dict[str, Any]] = None,
    ) -> None:
        async def crawl4ai(
            url: Annotated[str, "The url to crawl and extract information from."],
            llm_config: Annotated[Optional[dict[str, Any]], Depends[on(llm_config)]],
        ) -> Any:
            async with AsyncWebCrawler() as crawler:
                if llm_config is not None:
                    raise ValueError("llm_config is not supported yet.")
                result = await crawler.arun(
                    url=url,
                )
                return result.markdown

        super().__init__(
            name="crawl4ai",
            description="Crawl a website and extract information.",
            func_or_tool=crawl4ai,
        )
