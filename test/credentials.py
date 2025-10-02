# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import functools
import re
from typing import Any

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
