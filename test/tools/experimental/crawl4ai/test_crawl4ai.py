# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

import pytest
from pydantic import BaseModel

from autogen.import_utils import optional_import_block, skip_on_missing_imports
from autogen.tools.experimental.crawl4ai import Crawl4AITool

from ....conftest import Credentials

with optional_import_block():
    from crawl4ai import CrawlerRunConfig


@skip_on_missing_imports(["crawl4ai"], "crawl4ai")
class TestCrawl4AITool:
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

    @pytest.mark.parametrize(
        "config_list",
        [
            [
                {"api_type": "openai", "model": "gpt-4o-mini", "api_key": "test"},
            ],
            [
                {"api_type": "deepseek", "model": "deepseek-model", "api_key": "test", "base_url": "test"},
            ],
            [
                {
                    "api_type": "azure",
                    "model": "gpt-4o-mini",
                    "api_key": "test",
                    "base_url": "test",
                    "api_version": "test",
                },
            ],
            [
                {"api_type": "google", "model": "gemini", "api_key": "test"},
            ],
            [
                {"api_type": "anthropic", "model": "sonnet", "api_key": "test"},
            ],
            [{"api_type": "ollama", "model": "mistral:7b"}],
        ],
    )
    def test_get_provider_and_api_key(self, config_list: list[dict[str, Any]]) -> None:
        lite_llm_config = Crawl4AITool._get_lite_llm_config({"config_list": config_list})

        api_type = config_list[0]["api_type"]
        model = config_list[0]["model"]
        api_type = api_type if api_type != "google" else "gemini"
        provider = f"{api_type}/{model}"

        if api_type == "ollama":
            assert lite_llm_config == {"provider": provider}
        else:
            assert all(key in lite_llm_config for key in ["provider", "api_token"])
            assert lite_llm_config["provider"] == provider

        if api_type == "deepseek" or api_type == "azure":
            assert "base_url" in lite_llm_config

    @pytest.mark.parametrize(
        "use_extraction_model",
        [
            False,
            True,
        ],
    )
    def test_get_crawl_config(self, mock_credentials: Credentials, use_extraction_model: bool) -> None:
        class Product(BaseModel):
            name: str
            price: str

        extraction_model = Product if use_extraction_model else None

        config = Crawl4AITool._get_crawl_config(
            mock_credentials.llm_config, instruction="dummy", extraction_model=extraction_model
        )
        assert isinstance(config, CrawlerRunConfig)
        assert config.extraction_strategy.provider == f"openai/{mock_credentials.model}"

        if use_extraction_model:
            assert config.extraction_strategy.schema == Product.model_json_schema()
        else:
            assert config.extraction_strategy.schema is None

    @pytest.mark.openai
    @pytest.mark.asyncio
    async def test_with_llm(self, credentials_gpt_4o_mini: Credentials) -> None:
        tool_with_llm = Crawl4AITool(llm_config=credentials_gpt_4o_mini.llm_config)
        assert isinstance(tool_with_llm, Crawl4AITool)

        result = await tool_with_llm(
            url="https://docs.ag2.ai/docs/Home", instruction="Get the most relevant information from the page."
        )
        assert isinstance(result, str)

    @pytest.mark.openai
    @pytest.mark.asyncio
    async def test_with_llm_and_extraction_schema(self, credentials_gpt_4o_mini: Credentials) -> None:
        class Product(BaseModel):
            name: str
            price: str

        tool_with_llm = Crawl4AITool(
            llm_config=credentials_gpt_4o_mini.llm_config,
            extraction_model=Product,
        )
        assert isinstance(tool_with_llm, Crawl4AITool)

        result = await tool_with_llm(
            url="https://www.ikea.com/gb/en/",
            instruction="Extract all product objects with 'name' and 'price' from the content.",
        )
        assert isinstance(result, str)

    @pytest.mark.parametrize(
        ("llm_strategy_kwargs", "llm_config_provided", "expected_error"),
        [
            (None, True, None),
            ({"some_param": "dummy_value"}, True, None),
            (
                {"provider": "openai/gpt-4o", "api_token": "dummy_token"},
                False,
                "llm_strategy_kwargs can only be provided if llm_config is also provided.",
            ),
            (
                {"schema": "dummy_schema"},
                True,
                "'schema' should not be provided in llm_strategy_kwargs.",
            ),
            (
                {"instruction": "dummy_instruction"},
                True,
                "'instruction' should not be provided in llm_strategy_kwargs.",
            ),
        ],
    )
    def test_validate_llm_strategy_kwargs(
        self, llm_strategy_kwargs: Optional[dict[str, Any]], llm_config_provided: bool, expected_error: Optional[str]
    ) -> None:
        if expected_error is None:
            Crawl4AITool._validate_llm_strategy_kwargs(
                llm_strategy_kwargs=llm_strategy_kwargs, llm_config_provided=llm_config_provided
            )
            return

        with pytest.raises(
            ValueError,
            match=expected_error,
        ):
            Crawl4AITool._validate_llm_strategy_kwargs(
                llm_strategy_kwargs=llm_strategy_kwargs, llm_config_provided=llm_config_provided
            )
