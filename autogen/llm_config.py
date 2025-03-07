# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl

if TYPE_CHECKING:
    from .oai.client import ModelClient

# class LLMProvider(str, Enum):
#     openai = "openai"
#     bedrock = "bedrock"
#     anthropic = "anthropic"
#     cerebras = "cerebras"
#     cohere = "cohere"
#     deepseek = "deepseek"
#     google = "google"
#     groq = "groq"
#     mistral = "mistral"
#     ollama = "ollama"
#     together = "together"

LLMProvider = Literal[
    "openai",
    "azure",
    "bedrock",
    "anthropic",
    "cerebras",
    "cohere",
    "deepseek",
    "google",
    "groq",
    "mistral",
    "ollama",
    "together",
]


# class IndividualLLMConfig(BaseModel):
#     api_type: LLMProvider = "openai"
#     model: str
#     api_key: str
#     base_url: Optional[HttpUrl] = None
#     tags: Optional[list[str]] = None


class LLMConfig(BaseModel):
    # class variable not touched by BaseModel
    _current_llm_config: ContextVar[dict[str, list[dict[str, Any]]]] = ContextVar("current_llm_config")

    # used by BaseModel to create instance variables
    config_list: Annotated[list["LLMConfigEntry"], Field(default_factory=list)]
    temperature: Optional[float] = None
    check_every_ms: Optional[int] = None
    max_new_tokens: Optional[int] = None
    seed: Optional[int] = None
    allow_format_str_template: Optional[bool] = None
    response_format: Optional[str] = None
    timeout: Optional[int] = None
    cache_seed: Optional[int] = None

    def __enter__(self):
        # Store previous context and set self as current
        self._token = LLMConfig._current_llm_config.set(self.model_dump_json(exclude_none=True))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        LLMConfig._current_llm_config.reset(self._token)


class LLMConfigEntry(BaseModel, ABC):
    api_type: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[HttpUrl] = None
    tags: Annotated[list[str], Field(default_factory=list)]

    @property
    @abstractmethod
    def create_client(self) -> "ModelClient": ...

    def model_dump(self) -> dict[str, Any]:
        return BaseModel.model_dump(self, exclude_none=True)

    def model_dump_json(self) -> str:
        return BaseModel.model_dump_json(self, exclude_none=True)


class OpenAILLMConfigEntry(LLMConfigEntry):
    api_type: LLMProvider = "openai"

    def create_client(self) -> "ModelClient":
        raise NotImplementedError


class AzureOpenAILLMConfigEntry(LLMConfigEntry):
    api_type: LLMProvider = "azure"

    def create_client(self) -> "ModelClient":
        raise NotImplementedError


class GeminiLLMConfigEntry(LLMConfigEntry):
    api_type: LLMProvider = "gemini"

    def create_client(self) -> "ModelClient":
        raise NotImplementedError
