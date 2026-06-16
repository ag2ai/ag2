# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import TypedDict

import google.auth
import httpx
from google.genai import types
from typing_extensions import Unpack

from autogen.beta.config.config import ModelConfig

from .files import GeminiFilesClient
from .gemini_client import CreateConfig, GeminiClient, ThinkingLevel


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
    thinking_config: types.ThinkingConfig | None
    thinking_level: ThinkingLevel | None
    thinking_budget: int | None
    http_client: httpx.AsyncClient | None


class GeminiConfigOverrides(GeminiBaseConfigOverrides, total=False):
    api_key: str | None


class VertexAIConfigOverrides(GeminiBaseConfigOverrides, total=False):
    credentials: google.auth.credentials.Credentials | str | None
    project: str | None
    location: str | None


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
    thinking_config: types.ThinkingConfig | None = None
    thinking_level: ThinkingLevel | None = None
    thinking_budget: int | None = None
    http_client: httpx.AsyncClient | None = None

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

    def copy(self, /, **overrides: Unpack[GeminiConfigOverrides]) -> "GeminiConfig":
        return replace(self, **overrides)

    def create(self) -> GeminiClient:
        return GeminiClient(
            model=self.model,
            api_key=self.api_key,
            vertexai=False,
            http_client=self.http_client,
            streaming=self.streaming,
            create_config=self._build_create_config(),
            cached_content=self.cached_content,
        )

    def create_files_client(self) -> GeminiFilesClient:
        return GeminiFilesClient(self)


@dataclass(slots=True)
class VertexAIConfig(GeminiBaseConfig, ModelConfig):
    credentials: google.auth.credentials.Credentials | str | None = None
    project: str | None = None
    location: str | None = None

    def copy(self, /, **overrides: Unpack[VertexAIConfigOverrides]) -> "VertexAIConfig":
        return replace(self, **overrides)

    def create(self) -> GeminiClient:
        return GeminiClient(
            model=self.model,
            vertexai=True,
            credentials=self.credentials,
            project=self.project,
            location=self.location,
            http_client=self.http_client,
            streaming=self.streaming,
            create_config=self._build_create_config(),
            cached_content=self.cached_content,
        )
