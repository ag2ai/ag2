# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from .config import ModelConfig
from .llms.openai import OpenAIClient, ReasoningEffort


@dataclass(slots=True)
class OpenAIConfig(ModelConfig):
    model: str
    api_key: str | None = None
    base_url: str | None = None
    streaming: bool = False
    reasoning_effort: ReasoningEffort | None = None

    def copy(self) -> "OpenAIConfig":
        return OpenAIConfig(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            streaming=self.streaming,
            reasoning_effort=self.reasoning_effort,
        )

    def create(self) -> OpenAIClient:
        return OpenAIClient(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            streaming=self.streaming,
            reasoning_effort=self.reasoning_effort,
        )
