# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from contextvars import ContextVar
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, HttpUrl

current_llm_config: ContextVar[dict[str, list[dict[str, Any]]]] = ContextVar("current_llm_config")


class LLMProvider(str, Enum):
    openai = "openai"
    bedrock = "bedrock"
    anthropic = "anthropic"
    cerebras = "cerebras"
    cohere = "cohere"
    deepseek = "deepseek"
    google = "google"
    groq = "groq"
    mistral = "mistral"
    ollama = "ollama"
    together = "together"


class IndividualLLMConfig(BaseModel):
    api_type: LLMProvider = LLMProvider.openai
    model: str
    api_key: str
    base_url: Optional[HttpUrl] = None
    tags: Optional[list[str]] = None


class LLMConfig(BaseModel):
    config_list: list[IndividualLLMConfig]

    def __enter__(self):
        # Store previous context and set self as current
        self._token = current_llm_config.set(self.model_dump_json())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        current_llm_config.reset(self._token)
