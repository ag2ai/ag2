# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import Any, TypedDict

import httpx2
from anthropic import OpenTelemetryClientOption
from anthropic.types import ModelParam
from typing_extensions import Unpack

from ag2.config.config import ModelConfig, ModelProvider

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
    open_telemetry: OpenTelemetryClientOption


@dataclass(slots=True)
class AnthropicConfig(ModelConfig):
    model: ModelParam | str
    max_tokens: int = 4096
    api_key: str | None = None
    base_url: str | None = None
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
    open_telemetry: OpenTelemetryClientOption = None
    """The SDK's own OpenTelemetry tracing of API calls: `False` disables its spans, a dict configures them.

    `None` leaves the option unset, so the SDK reads the `ANTHROPIC_OPEN_TELEMETRY*` environment variables.
    """

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.ANTHROPIC

    def copy(self, /, **overrides: Unpack[AnthropicConfigOverrides]) -> "AnthropicConfig":
        return replace(self, **overrides)

    def create(self) -> AnthropicClient:
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
            extra_body=self.extra_body,
            open_telemetry=self.open_telemetry,
        )

    def create_files_client(self) -> AnthropicFilesClient:
        return AnthropicFilesClient(self)
