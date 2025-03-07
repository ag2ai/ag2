# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import functools
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, Type, Union

from pydantic import BaseModel, Field, HttpUrl

if TYPE_CHECKING:
    from .oai.client import ModelClient


current_llm_config: ContextVar["LLMConfig"] = ContextVar("current_llm_config")


class LLMConfig:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._model = self._get_base_model_class()(*args, **kwargs)

    # used by BaseModel to create instance variables
    def __enter__(self) -> "LLMConfig":
        # Store previous context and set self as current
        self._token = current_llm_config.set(self)
        return self

    def __exit__(self, exc_type: Type[Exception], exc_val: Exception, exc_tb: Any) -> None:
        current_llm_config.reset(self._token)

    # @functools.wraps(BaseModel.model_dump)
    def model_dump(self, *args: Any, exclude_none: bool = True, **kwargs: Any) -> dict[str, Any]:
        return self._model.model_dump(*args, exclude_none=exclude_none, **kwargs)

    # @functools.wraps(BaseModel.model_dump_json)
    def model_dump_json(self, *args: Any, exclude_none: bool = True, **kwargs: Any) -> str:
        return self._model.model_dump_json(*args, exclude_none=exclude_none, **kwargs)

    # @functools.wraps(BaseModel.model_validate)
    def model_validate(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.model_validate(*args, **kwargs)

    @functools.wraps(BaseModel.model_validate_json)
    def model_validate_json(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.model_validate_json(*args, **kwargs)

    @functools.wraps(BaseModel.model_validate_strings)
    def model_validate_strings(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.model_validate_strings(*args, **kwargs)

    def __eq__(self, value: Any) -> bool:
        return hasattr(value, "_model") and self._model == value._model

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return getattr(self._model, key, default)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self._model, key)
        except AttributeError:
            raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}")

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self._model, key, value)

    def __getattr__(self, name: Any) -> Any:
        try:
            return getattr(self._model, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    _base_model_classes: dict[tuple[Type["LLMConfigEntry"]], Type[BaseModel]] = {}

    @classmethod
    def _get_base_model_class(cls) -> Type["BaseModel"]:
        def _get_cls(llm_config_classes: tuple[Type[LLMConfigEntry]]) -> Type[BaseModel]:
            if llm_config_classes in LLMConfig._base_model_classes:
                return LLMConfig._base_model_classes[llm_config_classes]

            class _LLMConfig(BaseModel):
                temperature: Optional[float] = None
                check_every_ms: Optional[int] = None
                max_new_tokens: Optional[int] = None
                seed: Optional[int] = None
                allow_format_str_template: Optional[bool] = None
                response_format: Optional[str] = None
                timeout: Optional[int] = None
                cache_seed: Optional[int] = None

                config_list: Annotated[  # type: ignore[valid-type]
                    list[Annotated[Union[*llm_config_classes], Field(discriminator="api_type")]],
                    Field(default_factory=list),
                ]

            LLMConfig._base_model_classes[llm_config_classes] = _LLMConfig

            return _LLMConfig

        return _get_cls(tuple(_llm_config_classes))  # type: ignore[arg-type]


class LLMConfigEntry(BaseModel, ABC):
    api_type: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[HttpUrl] = None
    # tags: Annotated[list[str], Field(default_factory=list)]
    tags: list[str] = Field(default_factory=list)

    @abstractmethod
    def create_client(self) -> "ModelClient": ...

    def model_dump(self, *args: Any, exclude_none: bool = True, **kwargs: Any) -> dict[str, Any]:
        return BaseModel.model_dump(self, exclude_none=exclude_none, *args, **kwargs)

    def model_dump_json(self, *args: Any, exclude_none: bool = True, **kwargs: Any) -> str:
        return BaseModel.model_dump_json(self, exclude_none=exclude_none, *args, **kwargs)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}")

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)


_llm_config_classes: list[Type[LLMConfigEntry]] = []


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
