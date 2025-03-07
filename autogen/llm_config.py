# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, Type, Union

from pydantic import BaseModel, Field, HttpUrl

if TYPE_CHECKING:
    from .oai.client import ModelClient


# class LLMProvider(str, Enum):
#     openai = "openai"
#     azure = "azure"
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


# LLMProvider = Literal[
#     "openai",
#     "azure",
#     "bedrock",
#     "anthropic",
#     "cerebras",
#     "cohere",
#     "deepseek",
#     "google",
#     "groq",
#     "mistral",
#     "ollama",
#     "together",
# ]


_current_llm_config: ContextVar[dict[str, list[dict[str, Any]]]] = ContextVar("current_llm_config")


def get_LLMConfig():
    class _LLMConfig(BaseModel):
        # class variable not touched by BaseModel

        # used by BaseModel to create instance variables
        config_list: Annotated[list[LLMConfigItem()], Field(default_factory=list)]
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
            self._token = _current_llm_config.set(self.model_dump_json(exclude_none=True))
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            _current_llm_config.reset(self._token)

        def model_dump(self, *args, exclude_none: bool = True, **kwargs) -> dict[str, Any]:
            return BaseModel.model_dump(self, exclude_none=exclude_none, *args, **kwargs)

        def model_dump_json(self, *args, exclude_none: bool = True, **kwargs) -> str:
            return BaseModel.model_dump_json(self, exclude_none=exclude_none, *args, **kwargs)

    return _LLMConfig


class LLMConfigEntry(BaseModel, ABC):
    # api_type: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[HttpUrl] = None
    tags: Annotated[list[str], Field(default_factory=list)]

    @property
    @abstractmethod
    def create_client(self) -> "ModelClient": ...

    def model_dump(self, *args, exclude_none: bool = True, **kwargs) -> dict[str, Any]:
        return BaseModel.model_dump(self, exclude_none=exclude_none, *args, **kwargs)

    def model_dump_json(self, *args, exclude_none: bool = True, **kwargs) -> str:
        return BaseModel.model_dump_json(self, exclude_none=exclude_none, *args, **kwargs)


_llm_config_classes: list[Type[LLMConfigEntry]] = []


def LLMConfigItem():
    return Annotated[Union[*_llm_config_classes], Field(discriminator="api_type")]


# ToDo: Add a decorator to auto gene


def register_llm_config(cls: Type[LLMConfigEntry]) -> Type[LLMConfigEntry]:
    if isinstance(cls, type) and issubclass(cls, LLMConfigEntry):
        _llm_config_classes.append(cls)
    else:
        raise TypeError(f"Expected a subclass of LLMConfigEntry, got {cls}")
    return cls


@register_llm_config
class OpenAILLMConfigEntry(LLMConfigEntry):
    api_type: Literal["openai"] = "openai"

    def create_client(self) -> "ModelClient":
        raise NotImplementedError


@register_llm_config
class AzureOpenAILLMConfigEntry(LLMConfigEntry):
    api_type: Literal["azure"] = "azure"

    def create_client(self) -> "ModelClient":
        raise NotImplementedError


@register_llm_config
class GeminiLLMConfigEntry(LLMConfigEntry):
    api_type: Literal["google"] = "google"

    def create_client(self) -> "ModelClient":
        raise NotImplementedError
