# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, TypedDict

from typing_extensions import Unpack

from ag2.config.config import ModelConfig

from .files import XAIFilesClient
from .xai_client import CreateOptions, IncludeOption, ReasoningEffort, XAIClient

XAI_DEFAULT_API_HOST = "api.x.ai"


class XAIConfigOverrides(TypedDict, total=False):
    model: str
    api_key: str | None
    api_host: str
    timeout: float | None
    metadata: tuple[tuple[str, str], ...] | None
    channel_options: list[tuple[str, Any]] | None
    streaming: bool
    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    frequency_penalty: float | None
    presence_penalty: float | None
    seed: int | None
    stop: Sequence[str] | None
    user: str | None
    logprobs: bool | None
    top_logprobs: int | None
    tool_choice: str | None
    parallel_tool_calls: bool | None
    reasoning_effort: ReasoningEffort | None
    store_messages: bool | None
    previous_response_id: str | None
    use_encrypted_content: bool | None
    max_turns: int | None
    include: Sequence[IncludeOption] | None
    conversation_id: str | None


@dataclass(slots=True)
class XAIConfig(ModelConfig):
    model: str
    api_key: str | None = None
    api_host: str = XAI_DEFAULT_API_HOST
    timeout: float | None = None
    metadata: tuple[tuple[str, str], ...] | None = None
    channel_options: list[tuple[str, Any]] | None = None
    streaming: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: Sequence[str] | None = None
    user: str | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    tool_choice: str | None = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: ReasoningEffort | None = None
    store_messages: bool | None = None
    previous_response_id: str | None = None
    use_encrypted_content: bool | None = None
    max_turns: int | None = None
    include: Sequence[IncludeOption] | None = None
    conversation_id: str | None = None

    def copy(self, /, **overrides: Unpack[XAIConfigOverrides]) -> "XAIConfig":
        return replace(self, **overrides)

    def create(self) -> XAIClient:
        options = CreateOptions(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            seed=self.seed,
            stop=self.stop,
            user=self.user,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
            tool_choice=self.tool_choice,
            parallel_tool_calls=self.parallel_tool_calls,
            reasoning_effort=self.reasoning_effort,
            store_messages=self.store_messages,
            previous_response_id=self.previous_response_id,
            use_encrypted_content=self.use_encrypted_content,
            max_turns=self.max_turns,
            include=self.include,
            conversation_id=self.conversation_id,
        )

        return XAIClient(
            api_key=self.api_key,
            api_host=self.api_host,
            timeout=self.timeout,
            metadata=self.metadata,
            channel_options=self.channel_options,
            streaming=self.streaming,
            create_options=options,
        )

    def create_files_client(self) -> XAIFilesClient:
        return XAIFilesClient(self)
