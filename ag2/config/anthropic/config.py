# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypedDict, overload

import httpx2
from anthropic.types import ModelParam
from typing_extensions import Unpack, deprecated

from ag2.config.config import ModelConfig, ModelProvider

from ._compat import SAMPLING_DEPRECATION, SAMPLING_FIELDS, sampling_extra_body
from .anthropic_client import AnthropicClient, CreateOptions, Speed
from .files import AnthropicFilesClient


class AnthropicConfigOverrides(TypedDict, total=False):
    model: ModelParam | str
    api_key: str | None
    base_url: str | None
    max_tokens: int
    streaming: bool
    stop_sequences: list[str] | None
    timeout: float | None
    max_retries: int
    default_headers: dict[str, str] | None
    http_client: httpx2.AsyncClient | None
    metadata: dict[str, str] | None
    service_tier: str | None
    prompt_caching: bool
    cache_diagnostics: bool
    extra_body: dict[str, Any] | None
    speed: Speed | None


class _DeprecatedConfigOverrides(AnthropicConfigOverrides, total=False):
    temperature: float | None
    top_p: float | None
    top_k: int | None


@dataclass(slots=True)
class AnthropicConfig(ModelConfig):
    model: ModelParam | str
    max_tokens: int = 4096
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    """Deprecated: ignored with `anthropic` 2.x, whose API version rejects it. Removed in 1.4.0."""
    top_p: float | None = None
    """Deprecated: ignored with `anthropic` 2.x, whose API version rejects it. Removed in 1.4.0."""
    top_k: int | None = None
    """Deprecated: ignored with `anthropic` 2.x, whose API version rejects it. Removed in 1.4.0."""
    streaming: bool = False
    stop_sequences: list[str] | None = None
    timeout: float | None = None
    max_retries: int = 2
    default_headers: dict[str, str] | None = None
    http_client: httpx2.AsyncClient | None = None
    metadata: dict[str, str] | None = None
    service_tier: str | None = None
    prompt_caching: bool = True
    cache_diagnostics: bool = False
    extra_body: dict[str, Any] | None = None
    speed: Speed | None = None
    """Inference speed; some models require a fast-mode beta header for either value."""

    if TYPE_CHECKING:
        # The generated `__init__` carries the deprecated sampling parameters; these overloads
        # exist only so type checkers can flag them. `sampling_extra_body` warns at runtime.
        @overload
        def __init__(
            self,
            model: ModelParam | str,
            max_tokens: int = 4096,
            api_key: str | None = None,
            base_url: str | None = None,
            streaming: bool = False,
            stop_sequences: list[str] | None = None,
            timeout: float | None = None,
            max_retries: int = 2,
            default_headers: dict[str, str] | None = None,
            http_client: httpx2.AsyncClient | None = None,
            metadata: dict[str, str] | None = None,
            service_tier: str | None = None,
            prompt_caching: bool = True,
            cache_diagnostics: bool = False,
            extra_body: dict[str, Any] | None = None,
            speed: Speed | None = None,
        ) -> None: ...

        @overload
        @deprecated(SAMPLING_DEPRECATION)
        def __init__(
            self,
            model: ModelParam | str,
            max_tokens: int = 4096,
            api_key: str | None = None,
            base_url: str | None = None,
            temperature: float | None = None,
            top_p: float | None = None,
            top_k: int | None = None,
            streaming: bool = False,
            stop_sequences: list[str] | None = None,
            timeout: float | None = None,
            max_retries: int = 2,
            default_headers: dict[str, str] | None = None,
            http_client: httpx2.AsyncClient | None = None,
            metadata: dict[str, str] | None = None,
            service_tier: str | None = None,
            prompt_caching: bool = True,
            cache_diagnostics: bool = False,
            extra_body: dict[str, Any] | None = None,
            speed: Speed | None = None,
        ) -> None: ...

        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.ANTHROPIC

    @overload
    def copy(self, /, **overrides: Unpack[AnthropicConfigOverrides]) -> "AnthropicConfig": ...

    @overload
    @deprecated(SAMPLING_DEPRECATION)
    def copy(self, /, **overrides: Unpack[_DeprecatedConfigOverrides]) -> "AnthropicConfig": ...

    def copy(self, /, **overrides: Unpack[_DeprecatedConfigOverrides]) -> "AnthropicConfig":
        return replace(self, **overrides)

    def create(self) -> AnthropicClient:
        sampling = {name: value for name in SAMPLING_FIELDS if (value := getattr(self, name)) is not None}

        options = CreateOptions(
            model=self.model,
            max_tokens=self.max_tokens,
            stream=self.streaming,
        )

        if self.stop_sequences is not None:
            options["stop_sequences"] = self.stop_sequences
        if self.metadata is not None:
            options["metadata"] = self.metadata
        if self.service_tier is not None:
            options["service_tier"] = self.service_tier
        if self.speed is not None:
            options["speed"] = self.speed

        return AnthropicClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            http_client=self.http_client,
            create_options=options,
            prompt_caching=self.prompt_caching,
            cache_diagnostics=self.cache_diagnostics,
            extra_body=sampling_extra_body(sampling, self.extra_body),
        )

    def create_files_client(self) -> AnthropicFilesClient:
        return AnthropicFilesClient(self)
