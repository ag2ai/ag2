# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, List, Mapping, Optional, Type, Union

from httpx import Client as httpxClient
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, ValidationInfo, field_serializer, field_validator
from typing_extensions import Required, TypedDict

from .client import ModelClient


class LLMConfigEntryDict(TypedDict, total=False):
    api_type: Required[str]
    model: str
    max_tokens: Optional[int]
    top_p: Optional[float]
    temperature: Optional[float]

    api_key: Optional[SecretStr]
    api_version: Optional[str]
    base_url: Optional[HttpUrl]
    voice: Optional[str]
    http_client: Optional[httpxClient]
    model_client_cls: Optional[str]
    response_format: Optional[Union[str, dict[str, Any], BaseModel, Type[BaseModel]]]
    default_headers: Optional[Mapping[str, Any]]
    tags: List[str]


class ApplicationConfig(BaseModel):
    max_tokens: Optional[int] = Field(default=None, ge=0)
    top_p: Optional[float] = Field(default=None, ge=0)
    temperature: Optional[float] = None

    @field_validator("top_p", mode="before")
    @classmethod
    def check_top_p(cls, v: Optional[float], info: ValidationInfo) -> Optional[float]:
        if v is not None and info.data.get("temperature") is not None:
            raise ValueError("temperature and top_p cannot be set at the same time.")
        return v


class LLMConfigEntry(ApplicationConfig, ABC):
    api_type: str
    model: str = Field(..., min_length=1)

    api_key: Optional[SecretStr] = None
    api_version: Optional[str] = None
    base_url: Optional[HttpUrl] = None
    voice: Optional[str] = None
    model_client_cls: Optional[str] = None
    http_client: Optional[httpxClient] = None
    response_format: Optional[Union[str, dict[str, Any], BaseModel, Type[BaseModel]]] = None
    default_headers: Optional[Mapping[str, Any]] = None
    tags: List[str] = Field(default_factory=list)

    # Following field is configuration for pydantic to disallow extra fields
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def apply_application_config(self, application_config: ApplicationConfig) -> "LLMConfigEntry":
        """Apply application level configurations."""
        # TODO: should create a new instance instead of mutating current one
        self.max_tokens = self.max_tokens or application_config.max_tokens
        self.top_p = self.top_p or application_config.top_p
        self.temperature = self.temperature or application_config.temperature
        return self

    @abstractmethod
    def create_client(self) -> "ModelClient": ...

    @field_validator("base_url", mode="before")
    @classmethod
    def check_base_url(cls, v: Union[HttpUrl, str, None], info: ValidationInfo) -> Optional[str]:
        if v is None:  # Handle None case explicitly
            return None
        if not str(v).startswith("https://") and not str(v).startswith("http://"):
            return f"http://{str(v)}"
        return str(v)

    @field_serializer("base_url", when_used="unless-none")  # Ensure serializer also respects None
    def serialize_base_url(self, v: Optional[HttpUrl]) -> Optional[str]:
        return str(v) if v is not None else None

    @field_serializer("api_key", when_used="unless-none")
    def serialize_api_key(self, v: SecretStr) -> str:
        return v.get_secret_value()

    def model_dump(self, *args: Any, exclude_none: bool = True, **kwargs: Any) -> dict[str, Any]:
        return BaseModel.model_dump(self, exclude_none=exclude_none, *args, **kwargs)

    def model_dump_json(self, *args: Any, exclude_none: bool = True, **kwargs: Any) -> str:
        return BaseModel.model_dump_json(self, exclude_none=exclude_none, *args, **kwargs)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        val = getattr(self, key, default)
        if isinstance(val, SecretStr):
            return val.get_secret_value()
        return val

    def __getitem__(self, key: str) -> Any:
        try:
            val = getattr(self, key)
            if isinstance(val, SecretStr):
                return val.get_secret_value()
            return val
        except AttributeError:
            raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}")

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def items(self) -> Iterable[tuple[str, Any]]:
        d = self.model_dump()
        return d.items()

    def keys(self) -> Iterable[str]:
        d = self.model_dump()
        return d.keys()

    def values(self) -> Iterable[Any]:
        d = self.model_dump()
        return d.values()

    def __repr__(self) -> str:
        # Override to eliminate none values from the repr
        d = self.model_dump()
        r = [f"{k}={repr(v)}" for k, v in d.items()]

        s = f"{self.__class__.__name__}({', '.join(r)})"

        # Replace any keys ending with '_key' or '_token' values with stars for security
        # This regex will match any key ending with '_key' or '_token' and its value, and replace the value with stars
        # It also captures the type of quote used (single or double) and reuses it in the replacement
        s = re.sub(r'(\w+_(key|token)\s*=\s*)([\'"]).*?\3', r"\1\3**********\3", s, flags=re.IGNORECASE)

        return s

    def __str__(self) -> str:
        return repr(self)
