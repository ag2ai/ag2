# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import functools
import os
import re
from typing import Any

import pytest

from autogen import LLMConfig


class Secrets:
    _secrets: set[str] = set()

    @staticmethod
    def add_secret(secret: str) -> None:
        Secrets._secrets.add(secret)
        Secrets.get_secrets_patten.cache_clear()

    @staticmethod
    @functools.lru_cache(None)
    def get_secrets_patten(x: int = 5) -> re.Pattern[str]:
        """
        Builds a regex pattern to match substrings of length `x` or greater derived from any secret in the list.

        Args:
            data (str): The string to be checked.
            x (int): The minimum length of substrings to match.

        Returns:
            re.Pattern: Compiled regex pattern for matching substrings.
        """
        substrings: set[str] = set()
        for secret in Secrets._secrets:
            for length in range(x, len(secret) + 1):
                substrings.update(secret[i : i + length] for i in range(len(secret) - length + 1))

        return re.compile("|".join(re.escape(sub) for sub in sorted(substrings, key=len, reverse=True)))

    @staticmethod
    def sanitize_secrets(data: str, x: int = 5) -> str:
        """
        Censors substrings of length `x` or greater derived from any secret in the list.

        Args:
            data (str): The string to be censored.
            x (int): The minimum length of substrings to match.

        Returns:
            str: The censored string.
        """
        if len(Secrets._secrets) == 0:
            return data

        pattern = Secrets.get_secrets_patten(x)

        return re.sub(pattern, "*****", data)


class Credentials:
    """Credentials for the OpenAI API."""

    def __init__(self, llm_config: LLMConfig) -> None:
        self.llm_config = llm_config
        Secrets.add_secret(self.api_key)

    def sanitize(self) -> LLMConfig:
        llm_config = self.llm_config.copy()
        for config in llm_config["config_list"]:
            if "api_key" in config:
                config["api_key"] = "********"
        return llm_config

    def __repr__(self) -> str:
        return repr(self.sanitize())

    def __str___(self) -> str:
        return str(self.sanitize())

    @property
    def config_list(self) -> list[dict[str, Any]]:
        return [c.model_dump() for c in self.llm_config.config_list]

    @property
    def api_key(self) -> str:
        return self.config_list[0]["api_key"]  # type: ignore[no-any-return]

    @property
    def api_type(self) -> str:
        return self.config_list[0]["api_type"]  # type: ignore[no-any-return]

    @property
    def model(self) -> str:
        return self.config_list[0]["model"]  # type: ignore[no-any-return]


def build_config_from_env(
    api_type: str,
    env_var_name: str,
    model: str,
    base_url: str | None = None,
    api_version: str | None = None,
) -> dict[str, Any] | None:
    """Build a single config entry from environment variables.

    Args:
        api_type: The API type (e.g., 'openai', 'azure', 'google', 'anthropic')
        env_var_name: The environment variable name containing the API key
        model: The model name to use
        base_url: Optional base URL for the API
        api_version: Optional API version

    Returns:
        A configuration dictionary or None if the API key is not found
    """
    api_key = os.getenv(env_var_name)
    if not api_key:
        return None

    config = {
        "api_key": api_key,
        "model": model,
        "api_type": api_type,
    }

    if base_url:
        config["base_url"] = base_url
    if api_version:
        config["api_version"] = api_version

    return config


def get_credentials_from_env_vars(
    filter_dict: dict[str, Any] | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> Credentials:
    """Build credentials from individual environment variables instead of OAI_CONFIG_LIST.

    This function constructs a config list from GitHub secrets/environment variables:
    - OPENAI_API_KEY for OpenAI
    - AZURE_OPENAI_API_KEY + AZURE_OPENAI_API_BASE for Azure
    - GEMINI_API_KEY for Google/Gemivariablesni
    - ANTHROPIC_API_KEY for Anthropic
    - DEEPSEEK_API_KEY for DeepSeek
    - CEREBRAS_API_KEY for Cerebras
    - AWS credentials for Bedrock

    Args:
        filter_dict: Filter criteria for configurations (supports api_type and tags)
        temperature: Temperature for LLM
        **kwargs: Additional keyword arguments for e.g, (api_type, model, base_url, api_version)

    Returns:
        Credentials object with configurations built from environment
    """

    config_list = []

    # OpenAI configuration
    openai_config = build_config_from_env(
        api_type="openai",
        env_var_name="OPENAI_API_KEY",
        model="gpt-4o",
    )
    if openai_config:
        config_list.append({**openai_config, "tags": ["gpt-4o"]})
        config_list.append({**openai_config, "model": "gpt-4o-mini", "tags": ["gpt-4o-mini"]})
        config_list.append({**openai_config, "model": "gpt-4-turbo", "tags": ["gpt-4-turbo"]})
        config_list.append({**openai_config, "model": "gpt-4o-realtime-preview", "tags": ["gpt-4o-realtime"]})
        config_list.append({**openai_config, "model": "o1-mini", "tags": ["o1-mini"]})
        config_list.append({**openai_config, "model": "o1", "tags": ["o1"]})
        config_list.append({**openai_config, "model": "o4-mini", "tags": ["o4-mini"]})

    # Azure OpenAI configuration
    azure_base = os.getenv("AZURE_OPENAI_API_BASE")
    if azure_base:  # Only add if we have a base URL
        azure_config = build_config_from_env(
            api_type="azure",
            env_var_name="AZURE_OPENAI_API_KEY",
            model="gpt-4",
            base_url=azure_base,
            api_version="2024-02-01",
        )
        if azure_config:
            config_list.append({**azure_config, "tags": ["gpt-4"]})
            config_list.append({**azure_config, "model": "gpt-35-turbo", "tags": ["gpt-3.5-turbo"]})
            config_list.append({
                **azure_config,
                "model": "gpt-35-turbo-instruct",
                "tags": ["gpt-35-turbo-instruct", "gpt-3.5-turbo-instruct"],
            })

    # Google/Gemini configuration
    gemini_config = build_config_from_env(
        api_type="google",
        env_var_name="GEMINI_API_KEY",
        model="gemini-2.5-pro",
    )
    if gemini_config:
        config_list.append({**gemini_config, "tags": ["gemini-pro"]})
        config_list.append({**gemini_config, "model": "gemini-2.5-flash", "tags": ["gemini-2.5-flash", "gemini-flash"]})

    # Anthropic configuration
    anthropic_config = build_config_from_env(
        api_type="anthropic",
        env_var_name="ANTHROPIC_API_KEY",
        model="claude-sonnet-4-5-20250929",
    )
    if anthropic_config:
        config_list.append({**anthropic_config, "tags": ["anthropic-claude-sonnet"]})
        config_list.append({
            **anthropic_config,
            "model": "claude-sonnet-4-5-20250929",
            "tags": ["anthropic-claude-sonnet"],
        })
        config_list.append({**anthropic_config, "model": "claude-3-opus-20240229", "tags": ["anthropic-claude-opus"]})

    # DeepSeek configuration
    deepseek_config = build_config_from_env(
        api_type="openai",  # DeepSeek uses OpenAI-compatible API
        env_var_name="DEEPSEEK_API_KEY",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )
    if deepseek_config:
        config_list.append({**deepseek_config, "tags": ["deepseek-chat"]})
        config_list.append({**deepseek_config, "model": "deepseek-reasoner", "tags": ["deepseek-reasoner"]})

    # Cerebras configuration
    cerebras_config = build_config_from_env(
        api_type="openai",  # Cerebras uses OpenAI-compatible API
        env_var_name="CEREBRAS_API_KEY",
        model="llama3.1-8b",
        base_url="https://api.cerebras.ai/v1",
    )
    if cerebras_config:
        config_list.append({**cerebras_config, "tags": ["cerebras"]})

    # Ollama configuration (if running locally)
    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    if os.getenv("OLLAMA_API_KEY") or ollama_base != "http://localhost:11434/v1":
        ollama_config = {
            "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),  # Ollama doesn't require a key
            "model": "llama2",
            "api_type": "openai",
            "base_url": ollama_base,
        }
        config_list.append(ollama_config)

    # Bedrock configuration (uses AWS credentials)
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        bedrock_config = {
            "api_key": "bedrock",  # Placeholder, uses AWS credentials
            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "api_type": "bedrock",
        }
        config_list.append(bedrock_config)

    if not config_list:
        pytest.skip("No API keys found in environment variables")

    # Apply filters
    from autogen.llm_config.utils import filter_config

    if filter_dict:
        filtered_config_list = filter_config(config_list, filter_dict)
        if not filtered_config_list:
            pytest.skip(f"No configurations match the filter criteria: {filter_dict}")
        config_list = filtered_config_list

    llm_config = LLMConfig(*config_list, temperature=temperature)
    return Credentials(llm_config)
