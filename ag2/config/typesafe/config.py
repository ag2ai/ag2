# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypedDict

import httpx2
from typesafe_sdk import RetryPolicy
from typesafe_sdk.constants import DEFAULT_MODEL
from typing_extensions import Unpack

from ag2.config.config import ModelConfig, ModelProvider

from .typesafe_client import TypeSafeClient

if TYPE_CHECKING:
    from ag2.files.protocol import FilesClient


class TypeSafeConfigOverrides(TypedDict, total=False):
    model: str
    api_key: str | None
    base_url: str | None
    timeout: float | httpx2.Timeout | None
    retry: RetryPolicy | None
    headers: Mapping[str, str] | None
    http_client: httpx2.AsyncClient | None
    boolean_threshold: float
    criteria: Mapping[str, str] | None


@dataclass(slots=True)
class TypeSafeConfig(ModelConfig):
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    timeout: float | httpx2.Timeout | None = None
    retry: RetryPolicy | None = None
    headers: Mapping[str, str] | None = None
    http_client: httpx2.AsyncClient | None = None
    boolean_threshold: float = 0.5
    criteria: Mapping[str, str] | None = None

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.TYPESAFE

    def create_files_client(self) -> "FilesClient":
        raise NotImplementedError(f"{type(self).__name__} does not support Files API.")

    def copy(self, /, **overrides: Unpack[TypeSafeConfigOverrides]) -> "TypeSafeConfig":
        return replace(self, **overrides)

    def create(self) -> TypeSafeClient:
        return TypeSafeClient(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            retry=self.retry,
            headers=self.headers,
            http_client=self.http_client,
            boolean_threshold=self.boolean_threshold,
            criteria=self.criteria,
        )


__all__ = ["TypeSafeConfig", "TypeSafeConfigOverrides"]
