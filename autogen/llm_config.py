# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import functools
import json
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import TYPE_CHECKING, Annotated, Any, Optional, Type, TypeVar, Union

from pydantic import AnyUrl, BaseModel, Field, SecretStr, field_serializer

if TYPE_CHECKING:
    from _collections_abc import dict_items, dict_keys, dict_values

    from .oai.client import ModelClient

    _KT = TypeVar("_KT")
    _VT = TypeVar("_VT")

__all__ = [
    "LLMConfig",
    "LLMConfigEntry",
    "register_llm_config",
]


def _add_default_api_type(d: dict[str, Any]) -> dict[str, Any]:
    if "api_type" not in d:
        d["api_type"] = "openai"
    return d


class LLMConfig:
    _current_llm_config: ContextVar["LLMConfig"] = ContextVar("current_llm_config")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "config_list" in kwargs:
            kwargs["config_list"] = [
                _add_default_api_type(v) if isinstance(v, dict) else v for v in kwargs["config_list"]
            ]
        self._model = self._get_base_model_class()(*args, **kwargs)

    # used by BaseModel to create instance variables
    def __enter__(self) -> "LLMConfig":
        # Store previous context and set self as current
        self._token = LLMConfig._current_llm_config.set(self)
        return self

    def __exit__(self, exc_type: Type[Exception], exc_val: Exception, exc_tb: Any) -> None:
        LLMConfig._current_llm_config.reset(self._token)

    @classmethod
    def get_current_llm_config(cls) -> "Optional[LLMConfig]":
        try:
            return LLMConfig._current_llm_config.get()
        except LookupError:
            return None

    # @functools.wraps(BaseModel.model_dump)
    def model_dump(self, *args: Any, exclude_none: bool = True, **kwargs: Any) -> dict[str, Any]:
        d = self._model.model_dump(*args, exclude_none=exclude_none, **kwargs)
        return {k: v for k, v in d.items() if not (isinstance(v, list) and len(v) == 0)}

    # @functools.wraps(BaseModel.model_dump_json)
    def model_dump_json(self, *args: Any, exclude_none: bool = True, **kwargs: Any) -> str:
        # return self._model.model_dump_json(*args, exclude_none=exclude_none, **kwargs)
        d = self.model_dump(*args, exclude_none=exclude_none, **kwargs)
        return json.dumps(d)

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

    def _getattr(self, o: object, name: str) -> Any:
        val = getattr(o, name)
        if isinstance(val, list) and name == "config_list":
            return [v.model_dump() if isinstance(v, LLMConfigEntry) else v for v in val]
        return val

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        val = getattr(self._model, key, default)
        if isinstance(val, list) and key == "config_list":
            return [v.model_dump() if isinstance(v, LLMConfigEntry) else v for v in val]
        return val

    def __getitem__(self, key: str) -> Any:
        try:
            return self._getattr(self._model, key)
        except AttributeError:
            raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}")

    def __setitem__(self, key: str, value: Any) -> None:
        try:
            setattr(self._model, key, value)
        except ValueError:
            raise ValueError(f"'{self.__class__.__name__}' object has no field '{key}'")

    def __getattr__(self, name: Any) -> Any:
        try:
            return self._getattr(self._model, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __contains__(self, key: str) -> bool:
        return hasattr(self._model, key)

    def __repr__(self) -> str:
        return repr(self._model).replace("_LLMConfig", self.__class__.__name__)

    def items(self) -> "dict_items[_KT, _VT]":
        d = self.model_dump()
        return d.items()  # type: ignore[return-value]

    def keys(self) -> "dict_keys[_KT, _VT]":
        d = self.model_dump()
        return d.keys()  # type: ignore[return-value]

    def values(self) -> "dict_values[_KT, _VT]":
        d = self.model_dump()
        return d.values()  # type: ignore[return-value]

    _base_model_classes: dict[tuple[Type["LLMConfigEntry"], ...], Type[BaseModel]] = {}

    @classmethod
    def _get_base_model_class(cls) -> Type["BaseModel"]:
        def _get_cls(llm_config_classes: tuple[Type[LLMConfigEntry], ...]) -> Type[BaseModel]:
            if llm_config_classes in LLMConfig._base_model_classes:
                return LLMConfig._base_model_classes[llm_config_classes]

            class _LLMConfig(BaseModel):
                temperature: Optional[float] = None
                check_every_ms: Optional[int] = None
                max_new_tokens: Optional[int] = None
                seed: Optional[int] = None
                allow_format_str_template: Optional[bool] = None
                response_format: Optional[Union[str, dict[str, Any], BaseModel, Type[BaseModel]]] = None
                timeout: Optional[int] = None
                cache_seed: Optional[int] = None

                tools: list[Any] = Field(default_factory=list)
                functions: list[Any] = Field(default_factory=list)

                config_list: Annotated[  # type: ignore[valid-type]
                    list[Annotated[Union[llm_config_classes], Field(discriminator="api_type")]],
                    Field(default_factory=list, min_length=1),
                ]

            LLMConfig._base_model_classes[llm_config_classes] = _LLMConfig

            return _LLMConfig

        return _get_cls(tuple(_llm_config_classes))


class LLMConfigEntry(BaseModel, ABC):
    api_type: str
    model: str = Field(..., min_length=1)
    api_key: Optional[SecretStr] = None
    api_version: Optional[str] = None
    base_url: Optional[AnyUrl] = None
    model_client_cls: Optional[str] = None
    response_format: Optional[Union[str, dict[str, Any], BaseModel, Type[BaseModel]]] = None
    tags: list[str] = Field(default_factory=list)

    @abstractmethod
    def create_client(self) -> "ModelClient": ...

    @field_serializer("base_url")
    def serialize_base_url(self, v: Any) -> Any:
        return str(v)

    @field_serializer("api_key", when_used="unless-none")
    def serialize_api_key(self, v: SecretStr) -> Any:
        return v.get_secret_value()

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
