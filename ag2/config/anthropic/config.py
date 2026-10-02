# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import Any, TypedDict

import httpx2
from anthropic.types import ModelParam
from typing_extensions import Unpack

from ag2.config.config import ModelConfig, ModelProvider

from .anthropic_client import AnthropicClient, CreateOptions
from .files import AnthropicFilesClient
from .mappers import SAMPLING_FIELDS, merge_sampling_into_extra_body


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
    http_client: httpx2.AsyncClient | None
    metadata: dict[str, str] | None
    service_tier: str | None
    prompt_caching: bool
    extra_body: dict[str, Any] | None


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
    http_client: httpx2.AsyncClient | None = None
    metadata: dict[str, str] | None = None
    service_tier: str | None = None
    prompt_caching: bool = True
    extra_body: dict[str, Any] | None = None

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

        return AnthropicClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            http_client=self.http_client,
            create_options=options,
            prompt_caching=self.prompt_caching,
            extra_body=self._request_extra_body(),
        )

    def create_files_client(self) -> AnthropicFilesClient:
        return AnthropicFilesClient(self)

    def _request_extra_body(self) -> dict[str, Any] | None:
        """Fold the sampling fields into the request's extra body.

        ``anthropic`` 1.x dropped ``temperature``/``top_p``/``top_k`` from the
        generated method signatures — passing one is a ``TypeError`` — but the API
        still reads them from the request body, so the fields stay settable and
        keep reaching the model.
        """
        sampling = {name: value for name in SAMPLING_FIELDS if (value := getattr(self, name)) is not None}
        return merge_sampling_into_extra_body(sampling, self.extra_body)
