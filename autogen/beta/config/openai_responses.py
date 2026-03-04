# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import Any, TypedDict, Unpack

import httpx
from openai import DEFAULT_MAX_RETRIES, not_given

from .config import ModelConfig
from .llms.openai_responses import CreateOptions, OpenAIResponsesClient


class OpenAIResponsesConfigOverrides(TypedDict, total=False):
    model: str
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
    model: str
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
        options = CreateOptions(
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
