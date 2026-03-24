# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import Any, TypedDict

import httpx
from openai import DEFAULT_MAX_RETRIES, not_given, omit
from openai._types import Omit
from openai.types import ChatModel
from typing_extensions import Unpack

from autogen.beta.config.config import ModelConfig

from .openai_client import CreateOptions, OpenAIClient, ReasoningEffort
from .openai_responses_client import CreateOptions as ResponseCreateOptions
from .openai_responses_client import OpenAIResponsesClient


class OpenAIConfigOverrides(TypedDict, total=False):
    model: ChatModel | str
    api_key: str | None
    base_url: str | None
    temperature: float | None | Omit
    top_p: float | None | Omit
    streaming: bool
    max_tokens: int | None | Omit
    max_completion_tokens: int | None | Omit
    websocket_base_url: str | None
    organization: str | None
    project: str | None
    timeout: Any
    max_retries: int
    default_headers: dict[str, str] | None
    default_query: dict[str, object] | None
    http_client: httpx.AsyncClient | None
    frequency_penalty: float | None | Omit
    presence_penalty: float | None | Omit
    seed: int | None | Omit
    stop: str | list[str] | None | Omit
    n: int | None | Omit
    user: str | Omit
    logprobs: bool | None | Omit
    top_logprobs: int | None | Omit
    tool_choice: str | dict[str, Any] | Omit
    parallel_tool_calls: bool | Omit
    reasoning_effort: ReasoningEffort | None | Omit
    logit_bias: dict[str, int] | None | Omit
    metadata: dict[str, str] | None | Omit
    modalities: list[str] | None | Omit
    prediction: dict[str, Any] | None | Omit
    prompt_cache_key: str | Omit
    prompt_cache_retention: str | None | Omit
    safety_identifier: str | Omit
    service_tier: str | None | Omit
    store: bool | None | Omit
    verbosity: str | None | Omit
    web_search_options: dict[str, Any] | Omit


@dataclass(slots=True)
class OpenAIConfig(ModelConfig):
    model: ChatModel | str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None | Omit = omit
    top_p: float | None | Omit = omit
    streaming: bool = False
    max_tokens: int | None | Omit = omit
    max_completion_tokens: int | None | Omit = omit
    websocket_base_url: str | None = None
    organization: str | None = None
    project: str | None = None
    timeout: Any = not_given
    max_retries: int = DEFAULT_MAX_RETRIES
    default_headers: dict[str, str] | None = None
    default_query: dict[str, object] | None = None
    http_client: httpx.AsyncClient | None = None
    frequency_penalty: float | None | Omit = omit
    presence_penalty: float | None | Omit = omit
    seed: int | None | Omit = omit
    stop: str | list[str] | None | Omit = omit
    n: int | None | Omit = omit
    user: str | Omit = omit
    logprobs: bool | None | Omit = omit
    top_logprobs: int | None | Omit = omit
    tool_choice: str | dict[str, Any] | Omit = omit
    parallel_tool_calls: bool | Omit = omit
    reasoning_effort: ReasoningEffort | None | Omit = omit
    logit_bias: dict[str, int] | None | Omit = omit
    metadata: dict[str, str] | None | Omit = omit
    modalities: list[str] | None | Omit = omit
    prediction: dict[str, Any] | None | Omit = omit
    prompt_cache_key: str | Omit = omit
    prompt_cache_retention: str | None | Omit = omit
    safety_identifier: str | Omit = omit
    service_tier: str | None | Omit = omit
    store: bool | None | Omit = omit
    verbosity: str | None | Omit = omit
    web_search_options: dict[str, Any] | Omit = omit

    def copy(self, /, **overrides: Unpack[OpenAIConfigOverrides]) -> "OpenAIConfig":
        return replace(self, **overrides)

    def create(self) -> OpenAIClient:
        options = CreateOptions(
            model=self.model,
            stream=self.streaming,
            reasoning_effort=self.reasoning_effort,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            max_completion_tokens=self.max_completion_tokens,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            seed=self.seed,
            stop=self.stop,
            n=self.n,
            user=self.user,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
            tool_choice=self.tool_choice,
            parallel_tool_calls=self.parallel_tool_calls,
            logit_bias=self.logit_bias,
            metadata=self.metadata,
            modalities=self.modalities,
            prediction=self.prediction,
            prompt_cache_key=self.prompt_cache_key,
            prompt_cache_retention=self.prompt_cache_retention,
            safety_identifier=self.safety_identifier,
            service_tier=self.service_tier,
            store=self.store,
            verbosity=self.verbosity,
            web_search_options=self.web_search_options,
            stream_options={"include_usage": True} if self.streaming else omit,
        )

        return OpenAIClient(
            api_key=self.api_key,
            organization=self.organization,
            project=self.project,
            base_url=self.base_url,
            websocket_base_url=self.websocket_base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            default_query=self.default_query,
            http_client=self.http_client,
            create_options=options,
        )


class OpenAIResponsesConfigOverrides(TypedDict, total=False):
    model: ChatModel | str
    api_key: str | None
    base_url: str | None
    temperature: float | None
    top_p: float | None
    streaming: bool
    max_output_tokens: int | None
    max_tool_calls: int | None
    store: bool | None
    websocket_base_url: str | None
    organization: str | None
    project: str | None
    timeout: Any
    max_retries: int
    default_headers: dict[str, str] | None
    default_query: dict[str, object] | None
    http_client: httpx.AsyncClient | None
    parallel_tool_calls: bool
    top_logprobs: int | None
    metadata: dict[str, str] | None
    service_tier: str | None
    user: str
    truncation: str | None


@dataclass(slots=True)
class OpenAIResponsesConfig(ModelConfig):
    model: ChatModel | str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    streaming: bool = False
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    store: bool | None = True
    websocket_base_url: str | None = None
    organization: str | None = None
    project: str | None = None
    timeout: Any = not_given
    max_retries: int = DEFAULT_MAX_RETRIES
    default_headers: dict[str, str] | None = None
    default_query: dict[str, object] | None = None
    http_client: httpx.AsyncClient | None = None
    parallel_tool_calls: bool = True
    top_logprobs: int | None = None
    metadata: dict[str, str] | None = None
    service_tier: str | None = None
    user: str = ""
    truncation: str | None = None

    def copy(self, /, **overrides: Unpack[OpenAIResponsesConfigOverrides]) -> "OpenAIResponsesConfig":
        return replace(self, **overrides)

    def create(self) -> OpenAIResponsesClient:
        options = ResponseCreateOptions(
            model=self.model,
            stream=self.streaming,
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            max_tool_calls=self.max_tool_calls,
            store=self.store,
            parallel_tool_calls=self.parallel_tool_calls,
            top_logprobs=self.top_logprobs,
            metadata=self.metadata,
            service_tier=self.service_tier,
            truncation=self.truncation,
        )

        # Only include user if non-empty
        if self.user:
            options["user"] = self.user

        return OpenAIResponsesClient(
            api_key=self.api_key,
            organization=self.organization,
            project=self.project,
            base_url=self.base_url,
            websocket_base_url=self.websocket_base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            default_query=self.default_query,
            http_client=self.http_client,
            create_options=options,
        )
