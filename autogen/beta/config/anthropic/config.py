# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import TypedDict

import httpx
from anthropic.types import ModelParam
from typing_extensions import Unpack

from autogen.beta.config.config import ModelConfig

from .anthropic_client import AnthropicClient, CreateOptions
from .files import AnthropicFilesClient


class AnthropicConfigOverrides(TypedDict, total=False):
    model: ModelParam | str
    api_key: str | None
    base_url: str | None
    max_tokens: int
    temperature: float | None
    top_p: float | None
    top_k: int | None
    streaming: bool
    stop_sequences: list[str] | None
    timeout: float | None
    max_retries: int
    default_headers: dict[str, str] | None
    http_client: httpx.AsyncClient | None
    metadata: dict[str, str] | None
    service_tier: str | None
    prompt_caching: bool


@dataclass(slots=True)
class AnthropicConfig(ModelConfig):
    model: ModelParam | str
    max_tokens: int = 4096
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    streaming: bool = False
    stop_sequences: list[str] | None = None
    timeout: float | None = None
    max_retries: int = 2
    default_headers: dict[str, str] | None = None
    http_client: httpx.AsyncClient | None = None
    metadata: dict[str, str] | None = None
    service_tier: str | None = None
    prompt_caching: bool = True

    def copy(self, /, **overrides: Unpack[AnthropicConfigOverrides]) -> "AnthropicConfig":
        return replace(self, **overrides)

    def create(self) -> AnthropicClient:
        options = CreateOptions(
            model=self.model,
            max_tokens=self.max_tokens,
            stream=self.streaming,
        )

        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.top_k is not None:
            options["top_k"] = self.top_k
        if self.stop_sequences is not None:
            options["stop_sequences"] = self.stop_sequences
        if self.metadata is not None:
            options["metadata"] = self.metadata
        if self.service_tier is not None:
            options["service_tier"] = self.service_tier

        return AnthropicClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            http_client=self.http_client,
            create_options=options,
            prompt_caching=self.prompt_caching,
        )

    def create_files_client(self) -> AnthropicFilesClient:
        return AnthropicFilesClient(self)
