# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from importlib import metadata
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from autogen.interop import LiteLLmConfigFactory
from autogen.interop.litellm.litellm_config_factory import get_crawl4ai_version, is_crawl4ai_v05_or_higher


class TestLiteLLmConfigFactory:
    def test_number_of_factories(self) -> None:
        assert len(LiteLLmConfigFactory._factories) == 3

    @pytest.mark.parametrize(
        ("config_list", "expected_legacy", "expected_llm_config", "expected_strategy"),
        [
            (
                [{"api_type": "openai", "model": "gpt-4o-mini", "api_key": ""}],
                {"api_token": "", "provider": "openai/gpt-4o-mini"},
                {"api_token": "", "provider": "openai/gpt-4o-mini"},
                {},
            ),
            (
                [
                    {"api_type": "deepseek", "model": "deepseek-model", "api_key": "", "base_url": "test-url"},
                ],
                {"base_url": "test-url", "api_token": "", "provider": "deepseek/deepseek-model"},
                {"base_url": "test-url", "api_token": "", "provider": "deepseek/deepseek-model"},
                {},
            ),
            (
                [
                    {
                        "api_type": "azure",
                        "model": "gpt-4o-mini",
                        "api_key": "",
                        "base_url": "test",
                        "api_version": "test",
                    },
                ],
                {"base_url": "test", "api_version": "test", "api_token": "", "provider": "azure/gpt-4o-mini"},
                {"base_url": "test", "api_token": "", "provider": "azure/gpt-4o-mini"},
                {"api_version": "test"},
            ),
            (
                [
                    {"api_type": "google", "model": "gemini", "api_key": ""},
                ],
                {"api_token": "", "provider": "gemini/gemini"},
                {"api_token": "", "provider": "gemini/gemini"},
                {},
            ),
            (
                [
                    {"api_type": "anthropic", "model": "sonnet", "api_key": ""},
                ],
                {"api_token": "", "provider": "anthropic/sonnet"},
                {"api_token": "", "provider": "anthropic/sonnet"},
                {},
            ),
            (
                [{"api_type": "ollama", "model": "mistral:7b"}],
                {"provider": "ollama/mistral:7b"},
                {"provider": "ollama/mistral:7b"},
                {},
            ),
            (
                [{"api_type": "ollama", "model": "mistral:7b", "client_host": "http://127.0.0.1:11434"}],
                {"api_base": "http://127.0.0.1:11434", "provider": "ollama/mistral:7b"},
                {"base_url": "http://127.0.0.1:11434", "provider": "ollama/mistral:7b"},
                {},
            ),
        ],
    )
    def test_get_provider_and_api_key(
        self,
        config_list: list[dict[str, Any]],
        expected_legacy: dict[str, Any],
        expected_llm_config: dict[str, Any],
        expected_strategy: dict[str, Any],
    ) -> None:
        adapter = LiteLLmConfigFactory.create_lite_llm_config({"config_list": config_list})
        assert adapter.as_legacy_kwargs() == expected_legacy
        assert adapter.as_llm_config_kwargs() == expected_llm_config
        assert adapter.as_strategy_kwargs() == expected_strategy


class TestCrawl4aiCompatibility:
    """Test suite for crawl4ai version compatibility fix."""

    def test_get_crawl4ai_version_when_installed(self) -> None:
        """Test version detection when crawl4ai is installed."""
        with patch("autogen.interop.litellm.litellm_config_factory.metadata.version", return_value="0.5.0"):
            version = get_crawl4ai_version()
            assert version == "0.5.0"

    def test_get_crawl4ai_version_when_not_installed(self) -> None:
        """Test version detection when crawl4ai is not installed."""
        with patch(
            "autogen.interop.litellm.litellm_config_factory.metadata.version",
            side_effect=metadata.PackageNotFoundError("crawl4ai"),
        ):
            with patch.dict("sys.modules", {"crawl4ai": None}):
                version = get_crawl4ai_version()
                assert version is None

    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ("0.5.0", True),
            ("0.5.1", True),
            ("0.6.0", True),
            ("0.8.0", True),  # Latest version from PyPI
            ("0.4.247", False),
            ("0.4.999", False),
            ("0.3.0", False),
            ("0.0.1", False),
            (None, False),
        ],
    )
    def test_is_crawl4ai_v05_or_higher(self, version: str | None, expected: bool) -> None:
        """Test version comparison logic."""
        with patch("autogen.interop.litellm.litellm_config_factory.get_crawl4ai_version", return_value=version):
            result = is_crawl4ai_v05_or_higher()
            assert result == expected

    def test_is_crawl4ai_v05_or_higher_invalid_version(self) -> None:
        """Test version comparison with invalid version string."""
        with patch("autogen.interop.litellm.litellm_config_factory.get_crawl4ai_version", return_value="invalid"):
            result = is_crawl4ai_v05_or_higher()
            assert result is False

    def test_config_adaptation_with_multiple_parameters(self) -> None:
        """Test config adaptation with multiple parameters that should be moved to llmConfig."""
        config_list = [
            {
                "api_type": "azure",
                "model": "gpt-4o-mini",
                "api_key": "test-key",
                "base_url": "https://test.openai.azure.com/",
                "api_version": "2023-12-01-preview",
            }
        ]

        expected_legacy = {
            "api_token": "test-key",
            "provider": "azure/gpt-4o-mini",
            "base_url": "https://test.openai.azure.com/",
            "api_version": "2023-12-01-preview",
        }

        expected_llm_config = {
            "api_token": "test-key",
            "provider": "azure/gpt-4o-mini",
            "base_url": "https://test.openai.azure.com/",
        }

        adapter = LiteLLmConfigFactory.create_lite_llm_config({"config_list": config_list})
        assert adapter.as_legacy_kwargs() == expected_legacy
        assert adapter.as_llm_config_kwargs() == expected_llm_config
        assert adapter.as_strategy_kwargs() == {"api_version": "2023-12-01-preview"}

    def test_config_adaptation_preserves_other_parameters(self) -> None:
        """Test that config adaptation preserves parameters that shouldn't be moved to llmConfig."""
        config_list = [
            {
                "api_type": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test-key",
                "tags": ["test-tag"],
            }
        ]

        adapter = LiteLLmConfigFactory.create_lite_llm_config({"config_list": config_list})
        assert adapter.as_strategy_kwargs()["tags"] == ["test-tag"]
        assert adapter.as_llm_config_kwargs()["provider"] == "openai/gpt-4o-mini"
        assert adapter.as_llm_config_kwargs()["api_token"] == "test-key"

    @pytest.mark.parametrize(
        ("api_type", "model", "expected_provider"),
        [
            ("openai", "gpt-4o-mini", "openai/gpt-4o-mini"),
            ("anthropic", "claude-sonnet-4-5", "anthropic/claude-sonnet-4-5"),
            ("google", "gemini-pro", "gemini/gemini-pro"),  # Note: google gets converted to gemini
            ("azure", "gpt-4", "azure/gpt-4"),
            ("ollama", "llama2", "ollama/llama2"),
        ],
    )
    def test_provider_format_in_adapted_config(self, api_type: str, model: str, expected_provider: str) -> None:
        """Test that provider format is correct in adapted config for different API types."""
        config_list = [{"api_type": api_type, "model": model, "api_key": "test-key"}]

        adapter = LiteLLmConfigFactory.create_lite_llm_config({"config_list": config_list})
        assert adapter.as_llm_config_kwargs()["provider"] == expected_provider

    def test_backward_compatibility_no_crawl4ai(self) -> None:
        """Test that the fix doesn't break anything when crawl4ai is not installed."""
        config_list = [{"api_type": "openai", "model": "gpt-4o-mini", "api_key": "test-key"}]

        adapter = LiteLLmConfigFactory.create_lite_llm_config({"config_list": config_list})
        assert adapter.as_legacy_kwargs() == {"api_token": "test-key", "provider": "openai/gpt-4o-mini"}
