# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Annotated, Any, Optional

from ....import_utils import optional_import_block
from ... import Tool
from ...dependency_injection import Depends, on

with optional_import_block():
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
    from crawl4ai.extraction_strategy import LLMExtractionStrategy

__all__ = ["Crawl4AITool"]


class Crawl4AITool(Tool):
    def __init__(
        self,
        llm_config: Optional[dict[str, Any]] = None,
    ) -> None:
        async def crawl4ai_helper(  # type: ignore[no-any-unimported]
            url: str,
            browser_cfg: Optional[BrowserConfig] = None,
            crawl_config: Optional[CrawlerRunConfig] = None,
        ) -> Any:
            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                )

            if crawl_config is None:
                response = result.markdown
            else:
                response = result.extracted_content if result.success else result.error_message

            return response

        async def crawl4ai(
            url: Annotated[str, "The url to crawl and extract information from."],
            llm_config: Annotated[Optional[dict[str, Any]], Depends(on(llm_config))],
        ) -> Any:
            if llm_config is None:
                return await crawl4ai_helper(url=url)
            else:
                browser_cfg = BrowserConfig(headless=True)
                crawl_config = Crawl4AITool._get_crawl_config(llm_config)

                return await crawl4ai_helper(url=url, browser_cfg=browser_cfg, crawl_config=crawl_config)

        super().__init__(
            name="crawl4ai",
            description="Crawl a website and extract information.",
            func_or_tool=crawl4ai,
        )

    @staticmethod
    def _get_crawl_config(llm_config: dict[str, Any]) -> CrawlerRunConfig:  # type: ignore[no-any-unimported]
        if "config_list" not in llm_config:
            if "model" in llm_config:
                model = llm_config["model"]
                api_type = "openai"
                api_key = os.getenv("OPENAI_API_KEY")
            raise ValueError("llm_config must be a valid config dictionary.")
        else:
            try:
                model = llm_config["config_list"][0]["model"]
                api_type = llm_config["config_list"][0].get("api_type", "openai")
                api_key = llm_config["config_list"][0]["api_key"]

            except (KeyError, TypeError):
                raise ValueError("llm_config must be a valid config dictionary.")

        provider = f"{api_type}/{model}"

        # 1. Define the LLM extraction strategy
        llm_strategy = LLMExtractionStrategy(
            provider=provider,
            api_token=api_key,
            # schema=Product.schema_json(),            # Or use model_json_schema()
            # extraction_type="schema",
            instruction="Get the most relevant information from the page.",
            chunk_token_threshold=1000,
            overlap_rate=0.0,
            apply_chunking=True,
            input_format="markdown",  # or "html", "fit_markdown"
            extra_args={"temperature": 0.0, "max_tokens": 800},
        )

        # 2. Build the crawler config
        crawl_config = CrawlerRunConfig(extraction_strategy=llm_strategy, cache_mode=CacheMode.BYPASS)

        return crawl_config
