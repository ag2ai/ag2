# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from autogen.interop import LiteLLmConfigFactory
from autogen.interop.litellm.litellm_config_factory import get_crawl4ai_version, is_crawl4ai_v05_or_higher


class TestLiteLLmConfigFactory:
    def test_number_of_factories(self) -> None:
        assert len(LiteLLmConfigFactory._factories) == 3

    @pytest.mark.parametrize(
        ("config_list", "expected"),
        [
            (
                [{"api_type": "openai", "model": "gpt-4o-mini", "api_key": ""}],
                {"api_token": "", "provider": "openai/gpt-4o-mini"},
            ),
            (
                [
                    {"api_type": "deepseek", "model": "deepseek-model", "api_key": "", "base_url": "test-url"},
                ],
                {"base_url": "test-url", "api_token": "", "provider": "deepseek/deepseek-model"},
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
            ),
            (
                [
                    {"api_type": "google", "model": "gemini", "api_key": ""},
                ],
                {"api_token": "", "provider": "gemini/gemini"},
            ),
            (
                [
                    {"api_type": "anthropic", "model": "sonnet", "api_key": ""},
                ],
                {"api_token": "", "provider": "anthropic/sonnet"},
            ),
            (
                [{"api_type": "ollama", "model": "mistral:7b"}],
                {"provider": "ollama/mistral:7b"},
            ),
            (
                [{"api_type": "ollama", "model": "mistral:7b", "client_host": "http://127.0.0.1:11434"}],
                {"api_base": "http://127.0.0.1:11434", "provider": "ollama/mistral:7b"},
            ),
        ],
    )
    def test_get_provider_and_api_key(self, config_list: list[dict[str, Any]], expected: dict[str, Any]) -> None:
        lite_llm_config = LiteLLmConfigFactory.create_lite_llm_config({"config_list": config_list})
        assert lite_llm_config == expected


class TestCrawl4aiCompatibility:
    """Test suite for crawl4ai version compatibility fix."""

    def test_get_crawl4ai_version_when_installed(self) -> None:
        """Test version detection when crawl4ai is installed."""
        # Mock crawl4ai being installed with version 0.5.0
        mock_crawl4ai = MagicMock()
        mock_crawl4ai.__version__ = "0.5.0"

        with patch.dict("sys.modules", {"crawl4ai": mock_crawl4ai}):
            version = get_crawl4ai_version()
            assert version == "0.5.0"

    def test_get_crawl4ai_version_when_not_installed(self) -> None:
        """Test version detection when crawl4ai is not installed."""
        with patch.dict("sys.modules", {"crawl4ai": None}):
            version = get_crawl4ai_version()
            assert version is None

    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ("0.5.0", True),
            ("0.5.1", True),
            ("0.6.0", True),
            ("0.6.3", True),  # Latest version from PyPI
            ("0.4.247", False),
            ("0.4.999", False),
            ("0.3.0", False),
            ("0.0.1", False),
            (None, False),
        ],
    )
    def test_is_crawl4ai_v05_or_higher(self, version: Optional[str], expected: bool) -> None:
        """Test version comparison logic."""
        with patch("autogen.interop.litellm.litellm_config_factory.get_crawl4ai_version", return_value=version):
            result = is_crawl4ai_v05_or_higher()
            assert result == expected

    def test_is_crawl4ai_v05_or_higher_invalid_version(self) -> None:
        """Test version comparison with invalid version string."""
        with patch("autogen.interop.litellm.litellm_config_factory.get_crawl4ai_version", return_value="invalid"):
            result = is_crawl4ai_v05_or_higher()
            assert result is False

    @pytest.mark.parametrize(
        ("crawl4ai_version", "config_list", "expected"),
        [
            # Test with crawl4ai >=0.5 - should adapt config
            (
                "0.5.0",
                [{"api_type": "openai", "model": "gpt-4o-mini", "api_key": "test-key"}],
                {"llmConfig": {"api_token": "test-key", "provider": "openai/gpt-4o-mini"}},
            ),
            (
                "0.5.1",
                [{"api_type": "openai", "model": "gpt-4o-mini", "api_key": "test-key"}],
                {"llmConfig": {"api_token": "test-key", "provider": "openai/gpt-4o-mini"}},
            ),
            (
                "1.0.0",
                [{"api_type": "openai", "model": "gpt-4o-mini", "api_key": "test-key"}],
                {"llmConfig": {"api_token": "test-key", "provider": "openai/gpt-4o-mini"}},
            ),
            # Test with crawl4ai <0.5 - should use legacy format
            (
                "0.4.247",
                [{"api_type": "openai", "model": "gpt-4o-mini", "api_key": "test-key"}],
                {"api_token": "test-key", "provider": "openai/gpt-4o-mini"},
            ),
            (
                "0.4.999",
                [{"api_type": "openai", "model": "gpt-4o-mini", "api_key": "test-key"}],
                {"api_token": "test-key", "provider": "openai/gpt-4o-mini"},
            ),
            (
                "0.3.0",
                [{"api_type": "openai", "model": "gpt-4o-mini", "api_key": "test-key"}],
                {"api_token": "test-key", "provider": "openai/gpt-4o-mini"},
            ),
            # Test when crawl4ai is not installed - should use legacy format
            (
                None,
                [{"api_type": "openai", "model": "gpt-4o-mini", "api_key": "test-key"}],
                {"api_token": "test-key", "provider": "openai/gpt-4o-mini"},
            ),
        ],
    )
    def test_config_adaptation_based_on_crawl4ai_version(
        self, crawl4ai_version: Optional[str], config_list: list[dict[str, Any]], expected: dict[str, Any]
    ) -> None:
        """Test that config is properly adapted based on crawl4ai version."""
        with patch(
            "autogen.interop.litellm.litellm_config_factory.get_crawl4ai_version", return_value=crawl4ai_version
        ):
            lite_llm_config = LiteLLmConfigFactory.create_lite_llm_config({"config_list": config_list})
            assert lite_llm_config == expected

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

        expected_v05 = {
            "llmConfig": {
                "api_token": "test-key",
                "provider": "azure/gpt-4o-mini",
                "base_url": "https://test.openai.azure.com/",
                "api_version": "2023-12-01-preview",
            }
        }

        expected_legacy = {
            "api_token": "test-key",
            "provider": "azure/gpt-4o-mini",
            "base_url": "https://test.openai.azure.com/",
            "api_version": "2023-12-01-preview",
        }

        # Test with crawl4ai >=0.5
        with patch("autogen.interop.litellm.litellm_config_factory.get_crawl4ai_version", return_value="0.5.0"):
            lite_llm_config = LiteLLmConfigFactory.create_lite_llm_config({"config_list": config_list})
            assert lite_llm_config == expected_v05

        # Test with crawl4ai <0.5
        with patch("autogen.interop.litellm.litellm_config_factory.get_crawl4ai_version", return_value="0.4.247"):
            lite_llm_config = LiteLLmConfigFactory.create_lite_llm_config({"config_list": config_list})
            assert lite_llm_config == expected_legacy

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

        # Test with crawl4ai >=0.5 - tags should remain at top level
        with patch("autogen.interop.litellm.litellm_config_factory.get_crawl4ai_version", return_value="0.5.0"):
            lite_llm_config = LiteLLmConfigFactory.create_lite_llm_config({"config_list": config_list})
            assert "tags" in lite_llm_config
            assert lite_llm_config["tags"] == ["test-tag"]
            assert "llmConfig" in lite_llm_config
            assert "provider" in lite_llm_config["llmConfig"]
            assert "api_token" in lite_llm_config["llmConfig"]

    @pytest.mark.parametrize(
        ("api_type", "model", "expected_provider"),
        [
            ("openai", "gpt-4o-mini", "openai/gpt-4o-mini"),
            ("anthropic", "claude-3-sonnet", "anthropic/claude-3-sonnet"),
            ("google", "gemini-pro", "gemini/gemini-pro"),  # Note: google gets converted to gemini
            ("azure", "gpt-4", "azure/gpt-4"),
            ("ollama", "llama2", "ollama/llama2"),
        ],
    )
    def test_provider_format_in_adapted_config(self, api_type: str, model: str, expected_provider: str) -> None:
        """Test that provider format is correct in adapted config for different API types."""
        config_list = [{"api_type": api_type, "model": model, "api_key": "test-key"}]

        with patch("autogen.interop.litellm.litellm_config_factory.get_crawl4ai_version", return_value="0.5.0"):
            lite_llm_config = LiteLLmConfigFactory.create_lite_llm_config({"config_list": config_list})
            assert lite_llm_config["llmConfig"]["provider"] == expected_provider

    def test_backward_compatibility_no_crawl4ai(self) -> None:
        """Test that the fix doesn't break anything when crawl4ai is not installed."""
        config_list = [{"api_type": "openai", "model": "gpt-4o-mini", "api_key": "test-key"}]

        # Mock crawl4ai not being installed
        with patch("autogen.interop.litellm.litellm_config_factory.get_crawl4ai_version", return_value=None):
            lite_llm_config = LiteLLmConfigFactory.create_lite_llm_config({"config_list": config_list})
            # Should use legacy format
            assert lite_llm_config == {"api_token": "test-key", "provider": "openai/gpt-4o-mini"}
            assert "llmConfig" not in lite_llm_config
