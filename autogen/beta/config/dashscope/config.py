# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import TypedDict

from typing_extensions import Unpack

from autogen.beta.config.config import ModelConfig

from .dashscope_client import DASHSCOPE_INTL_BASE_URL, CreateOptions, DashScopeClient


class DashScopeConfigOverrides(TypedDict, total=False):
    model: str
    base_url: str
    api_key: str | None
    temperature: float | None
    top_p: float | None
    streaming: bool
    max_tokens: int | None
    stop: str | list[str] | None
    seed: int | None
    frequency_penalty: float | None
    presence_penalty: float | None


@dataclass(slots=True)
class DashScopeConfig(ModelConfig):
    model: str
    base_url: str = DASHSCOPE_INTL_BASE_URL
    api_key: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    streaming: bool = False
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    def copy(self, /, **overrides: Unpack[DashScopeConfigOverrides]) -> "DashScopeConfig":
        return replace(self, **overrides)

    def create(self) -> DashScopeClient:
        options = CreateOptions(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop,
            seed=self.seed,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )

        return DashScopeClient(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            streaming=self.streaming,
            create_options=options,
        )
