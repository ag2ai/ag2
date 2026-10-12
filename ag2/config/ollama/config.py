# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import TypedDict

import httpx
from typing_extensions import Unpack

from ag2.config.config import ModelConfig, ModelProvider

from .ollama_client import OLLAMA_DEFAULT_HOST, CreateOptions, OllamaClient


class OllamaConfigOverrides(TypedDict, total=False):
    model: str
    host: str
    api_key: str | None
    http_client: httpx.AsyncClient | None
    temperature: float | None
    top_p: float | None
    streaming: bool
    max_tokens: int | None
    stop: str | list[str] | None
    seed: int | None
    frequency_penalty: float | None
    presence_penalty: float | None


@dataclass(slots=True)
class OllamaConfig(ModelConfig):
    """Configuration for an Ollama model.

    `api_key` is sent as a bearer token (Ollama Cloud); without it the SDK reads `OLLAMA_API_KEY`
    unless `http_client` is given. `http_client` is a ready-made `httpx.AsyncClient`, e.g.
    `httpx.AsyncClient(proxy="http://proxy:8080")`; it is left unmodified and open, and `host` and
    `api_key` apply to each request on top of it.
    """

    model: str
    host: str = OLLAMA_DEFAULT_HOST
    api_key: str | None = None
    http_client: httpx.AsyncClient | None = None
    temperature: float | None = None
    top_p: float | None = None
    streaming: bool = False
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    def copy(self, /, **overrides: Unpack[OllamaConfigOverrides]) -> "OllamaConfig":
        return replace(self, **overrides)

    def create(self) -> OllamaClient:
        options = CreateOptions(
            temperature=self.temperature,
            top_p=self.top_p,
            num_predict=self.max_tokens,
            stop=self.stop,
            seed=self.seed,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )

        return OllamaClient(
            model=self.model,
            host=self.host,
            api_key=self.api_key,
            http_client=self.http_client,
            streaming=self.streaming,
            create_options=options,
        )

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.OLLAMA
