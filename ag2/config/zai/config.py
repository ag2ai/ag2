# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import Any, TypedDict

import httpx
from typing_extensions import Unpack

from ag2.config.config import ModelConfig

from .files import ZAIFilesClient
from .zai_client import CreateOptions, ZAIClient


class ZAIConfigOverrides(TypedDict, total=False):
    model: str
    api_key: str | None
    base_url: str | None
    timeout: float | httpx.Timeout | None
    max_retries: int
    http_client: httpx.Client | None
    custom_headers: dict[str, str] | None
    disable_token_cache: bool
    source_channel: str | None
    streaming: bool
    max_tokens: int | None
    temperature: float | None
    top_p: float | None
    stop: str | list[str] | None
    seed: int | None
    tool_choice: str | dict[str, Any] | None
    request_id: str | None
    user_id: str | None
    do_sample: bool | None
    meta: dict[str, str] | None
    sensitive_word_check: Any | None
    extra: Any | None
    request_timeout: float | httpx.Timeout | None
    watermark_enabled: bool | None
    tool_stream: bool | None
    reasoning_effort: str | None
    thinking: bool | None
    extra_headers: dict[str, str] | None
    extra_body: dict[str, Any] | None


@dataclass(slots=True)
class ZAIConfig(ModelConfig):
    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: float | httpx.Timeout | None = None
    max_retries: int = 3
    http_client: httpx.Client | None = None
    custom_headers: dict[str, str] | None = None
    disable_token_cache: bool = True
    source_channel: str | None = None
    streaming: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    tool_choice: str | dict[str, Any] | None = None
    request_id: str | None = None
    user_id: str | None = None
    do_sample: bool | None = None
    meta: dict[str, str] | None = None
    sensitive_word_check: Any | None = None
    extra: Any | None = None
    request_timeout: float | httpx.Timeout | None = None
    watermark_enabled: bool | None = None
    tool_stream: bool | None = None
    reasoning_effort: str | None = None
    thinking: bool | None = None
    extra_headers: dict[str, str] | None = None
    extra_body: dict[str, Any] | None = None

    def copy(self, /, **overrides: Unpack[ZAIConfigOverrides]) -> "ZAIConfig":
        return replace(self, **overrides)

    def create_files_client(self) -> ZAIFilesClient:
        return ZAIFilesClient(self)

    def create(self) -> ZAIClient:
        options = CreateOptions(
            model=self.model,
            stream=self.streaming,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop,
            seed=self.seed,
            tool_choice=self.tool_choice,
            request_id=self.request_id,
            user_id=self.user_id,
            do_sample=self.do_sample,
            meta=self.meta,
            sensitive_word_check=self.sensitive_word_check,
            extra=self.extra,
            timeout=self.request_timeout,
            watermark_enabled=self.watermark_enabled,
            tool_stream=self.tool_stream,
            reasoning_effort=self.reasoning_effort,
            thinking={"type": "enabled" if self.thinking else "disabled"} if self.thinking is not None else None,
            extra_headers=self.extra_headers,
            extra_body=self.extra_body,
        )

        return ZAIClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client=self.http_client,
            custom_headers=self.custom_headers,
            disable_token_cache=self.disable_token_cache,
            source_channel=self.source_channel,
            create_options=options,
        )
