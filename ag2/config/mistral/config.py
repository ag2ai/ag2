# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import Any, Literal, TypedDict

import httpx
from typing_extensions import Unpack

from ag2.config.config import ModelConfig, ModelProvider

from .files import MistralFilesClient
from .mistral_client import MISTRAL_DEFAULT_SERVER_URL, CreateOptions, MistralClient, ReasoningEffort


class MistralConfigOverrides(TypedDict, total=False):
    model: str
    api_key: str | None
    server_url: str | None
    timeout_ms: int | None
    async_client: httpx.AsyncClient | None
    streaming: bool
    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    stop: str | list[str] | None
    random_seed: int | None
    presence_penalty: float | None
    frequency_penalty: float | None
    n: int | None
    parallel_tool_calls: bool | None
    tool_choice: str | dict[str, Any] | None
    reasoning_effort: ReasoningEffort | None
    prompt_mode: Literal["reasoning"] | None
    prompt_cache_key: str | None
    safe_prompt: bool | None
    metadata: dict[str, Any] | None
    http_headers: dict[str, str] | None


@dataclass(slots=True)
class MistralConfig(ModelConfig):
    """Configuration for Mistral's chat-completions API (``mistralai`` SDK).

    ``api_key`` falls back to the ``MISTRAL_API_KEY`` environment variable.
    """

    model: str
    api_key: str | None = None
    server_url: str | None = None
    timeout_ms: int | None = None
    async_client: httpx.AsyncClient | None = None
    streaming: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    random_seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    n: int | None = None
    parallel_tool_calls: bool | None = None
    tool_choice: str | dict[str, Any] | None = None
    reasoning_effort: ReasoningEffort | None = None
    prompt_mode: Literal["reasoning"] | None = None
    prompt_cache_key: str | None = None
    safe_prompt: bool | None = None
    metadata: dict[str, Any] | None = None
    http_headers: dict[str, str] | None = None

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.MISTRAL

    def copy(self, /, **overrides: Unpack[MistralConfigOverrides]) -> "MistralConfig":
        return replace(self, **overrides)

    def create(self) -> MistralClient:
        options = CreateOptions(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop,
            random_seed=self.random_seed,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            n=self.n,
            parallel_tool_calls=self.parallel_tool_calls,
            tool_choice=self.tool_choice,
            reasoning_effort=self.reasoning_effort,
            prompt_mode=self.prompt_mode,
            prompt_cache_key=self.prompt_cache_key,
            safe_prompt=self.safe_prompt,
            metadata=self.metadata,
            http_headers=self.http_headers,
        )

        return MistralClient(
            model=self.model,
            api_key=self.api_key,
            server_url=self.server_url,
            timeout_ms=self.timeout_ms,
            async_client=self.async_client,
            streaming=self.streaming,
            create_options=options,
        )

    def create_files_client(self) -> MistralFilesClient:
        return MistralFilesClient(self)


__all__ = ["MISTRAL_DEFAULT_SERVER_URL", "MistralConfig", "MistralConfigOverrides"]
