# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass, replace
from typing import Any, TypedDict

import httpx
from openai import DEFAULT_MAX_RETRIES, Omit, not_given, omit
from openai.types import ChatModel
from typing_extensions import Unpack

from ag2.config.config import ModelConfig, ModelProvider

from .orcarouter_client import CreateOptions, OrcaRouterClient, ReasoningEffort

ORCAROUTER_DEFAULT_BASE_URL = "https://api.orcarouter.ai/v1"
ORCAROUTER_DEFAULT_MODEL = "orcarouter/auto"


class OrcaRouterConfigOverrides(TypedDict, total=False):
    model: ChatModel | str
    api_key: str | None
    base_url: str | None
    temperature: float | None | Omit
    top_p: float | None | Omit
    streaming: bool
    max_tokens: int | None | Omit
    max_completion_tokens: int | None | Omit
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
    safety_identifier: str | Omit
    service_tier: str | None | Omit
    store: bool | None | Omit
    verbosity: str | None | Omit
    web_search_options: dict[str, Any] | Omit
    extra_body: dict[str, Any] | None


@dataclass(slots=True)
class OrcaRouterConfig(ModelConfig):
    """Configuration for the OrcaRouter OpenAI-compatible gateway.

    OrcaRouter (``https://api.orcarouter.ai/v1``) exposes an OpenAI-compatible
    chat-completions API that routes across multiple upstream model providers.
    ``api_key`` falls back to the ``ORCAROUTER_API_KEY`` environment variable;
    ``base_url`` defaults to ``https://api.orcarouter.ai/v1``.
    """

    model: ChatModel | str = ORCAROUTER_DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None | Omit = omit
    top_p: float | None | Omit = omit
    streaming: bool = False
    max_tokens: int | None | Omit = omit
    max_completion_tokens: int | None | Omit = omit
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
    safety_identifier: str | Omit = omit
    service_tier: str | None | Omit = omit
    store: bool | None | Omit = omit
    verbosity: str | None | Omit = omit
    web_search_options: dict[str, Any] | Omit = omit
    extra_body: dict[str, Any] | None = None

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.ORCAROUTER

    def copy(self, /, **overrides: Unpack[OrcaRouterConfigOverrides]) -> "OrcaRouterConfig":
        return replace(self, **overrides)

    def create(self) -> OrcaRouterClient:
        api_key = self.api_key if self.api_key is not None else os.getenv("ORCAROUTER_API_KEY")
        base_url = self.base_url if self.base_url is not None else ORCAROUTER_DEFAULT_BASE_URL

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
            safety_identifier=self.safety_identifier,
            service_tier=self.service_tier,
            store=self.store,
            verbosity=self.verbosity,
            web_search_options=self.web_search_options,
            stream_options={"include_usage": True} if self.streaming else omit,
            extra_body=self.extra_body,
        )

        return OrcaRouterClient(
            api_key=api_key,
            organization=self.organization,
            project=self.project,
            base_url=base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            default_query=self.default_query,
            http_client=self.http_client,
            create_options=options,
        )
