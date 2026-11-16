# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import TypedDict

import google.auth
import httpx
import httpx2
from google.genai import types
from typing_extensions import Unpack

from ag2.config.config import ModelConfig, ModelProvider

from ._compat import warn_if_sdk_2x
from .files import GeminiFilesClient
from .gemini_client import CreateConfig, GeminiApiVersion, GeminiClient, ThinkingLevel, VertexAIApiVersion


class GeminiBaseConfigOverrides(TypedDict, total=False):
    model: str
    temperature: float | None
    top_p: float | None
    top_k: int | None
    max_output_tokens: int | None
    stop_sequences: list[str] | None
    streaming: bool
    presence_penalty: float | None
    frequency_penalty: float | None
    seed: int | None
    cached_content: str | None
    response_modalities: list[str] | None
    image_config: types.ImageConfig | None
    thinking_config: types.ThinkingConfig | None
    thinking_level: ThinkingLevel | None
    thinking_budget: int | None
    automatic_continuation: bool | None
    http_client: httpx.AsyncClient | httpx2.AsyncClient | None


class GeminiConfigOverrides(GeminiBaseConfigOverrides, total=False):
    api_key: str | None
    api_version: GeminiApiVersion | None


class VertexAIConfigOverrides(GeminiBaseConfigOverrides, total=False):
    credentials: google.auth.credentials.Credentials | str | None
    project: str | None
    location: str | None
    api_version: VertexAIApiVersion | None


@dataclass(slots=True)
class GeminiBaseConfig:
    model: str
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_output_tokens: int | None = None
    stop_sequences: list[str] | None = None
    streaming: bool = False
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    cached_content: str | None = None
    response_modalities: list[str] | None = None
    image_config: types.ImageConfig | None = None
    thinking_config: types.ThinkingConfig | None = None
    thinking_level: ThinkingLevel | None = None
    thinking_budget: int | None = None
    # A response cut at the per-request token limit ends with finish reason
    # `CONTINUATION`; google-genai >= 2.30 then resumes it with the returned token
    # until the model finishes, unless this is False. Each resumed request is
    # billed on its own, and the usage reported is their sum. When off (or on
    # google-genai 2.29), the response reports `finish_reason="continuation"`.
    automatic_continuation: bool | None = None
    # google-genai 3.x accepts an httpx2 or an httpx client; 2.x only an httpx one.
    http_client: httpx.AsyncClient | httpx2.AsyncClient | None = None

    def _build_create_config(self) -> CreateConfig:
        config = CreateConfig()

        if self.temperature is not None:
            config["temperature"] = self.temperature
        if self.top_p is not None:
            config["top_p"] = self.top_p
        if self.top_k is not None:
            config["top_k"] = self.top_k
        if self.max_output_tokens is not None:
            config["max_output_tokens"] = self.max_output_tokens
        if self.stop_sequences is not None:
            config["stop_sequences"] = self.stop_sequences
        if self.presence_penalty is not None:
            config["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            config["frequency_penalty"] = self.frequency_penalty
        if self.seed is not None:
            config["seed"] = self.seed
        if self.response_modalities is not None:
            config["response_modalities"] = self.response_modalities
        if self.image_config is not None:
            config["image_config"] = self.image_config
        if self.automatic_continuation is not None:
            config["automatic_continuation"] = self.automatic_continuation

        thinking = self._resolve_thinking_config()
        if thinking is not None:
            config["thinking_config"] = thinking

        return config

    def _resolve_thinking_config(self) -> types.ThinkingConfig | None:
        if self.thinking_config is not None:
            return self.thinking_config
        if self.thinking_level is None and self.thinking_budget is None:
            return None
        kwargs: dict[str, object] = {}
        if self.thinking_level is not None:
            kwargs["thinking_level"] = self.thinking_level
        if self.thinking_budget is not None:
            kwargs["thinking_budget"] = self.thinking_budget
        return types.ThinkingConfig(**kwargs)


@dataclass(slots=True)
class GeminiConfig(GeminiBaseConfig, ModelConfig):
    api_key: str | None = None
    # None keeps the SDK's default: `v1` from google-genai 3.x, `v1beta` before it.
    api_version: GeminiApiVersion | None = None

    def copy(self, /, **overrides: Unpack[GeminiConfigOverrides]) -> "GeminiConfig":
        return replace(self, **overrides)

    def create(self) -> GeminiClient:
        warn_if_sdk_2x()
        return GeminiClient(
            model=self.model,
            api_key=self.api_key,
            vertexai=False,
            http_client=self.http_client,
            api_version=self.api_version,
            streaming=self.streaming,
            create_config=self._build_create_config(),
            cached_content=self.cached_content,
        )

    def create_files_client(self) -> GeminiFilesClient:
        warn_if_sdk_2x()
        return GeminiFilesClient(self)

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.GEMINI


@dataclass(slots=True)
class VertexAIConfig(GeminiBaseConfig, ModelConfig):
    credentials: google.auth.credentials.Credentials | str | None = None
    project: str | None = None
    location: str | None = None
    # None keeps the SDK's default, `v1beta1`.
    api_version: VertexAIApiVersion | None = None

    def copy(self, /, **overrides: Unpack[VertexAIConfigOverrides]) -> "VertexAIConfig":
        return replace(self, **overrides)

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.VERTEXAI

    def create(self) -> GeminiClient:
        warn_if_sdk_2x()
        return GeminiClient(
            model=self.model,
            vertexai=True,
            credentials=self.credentials,
            project=self.project,
            location=self.location,
            http_client=self.http_client,
            api_version=self.api_version,
            streaming=self.streaming,
            create_config=self._build_create_config(),
            cached_content=self.cached_content,
        )
