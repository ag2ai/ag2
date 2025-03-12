# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, Optional

from pydantic import AnyUrl, Field, ValidationInfo, field_validator

from ..llm_config import LLMConfigEntry, register_llm_config


@register_llm_config
class DeepSeekLLMConfigEntry(LLMConfigEntry):
    api_type: Literal["deepseek"] = "deepseek"
    base_url: AnyUrl = AnyUrl("https://api.deepseek.com/v1")
    temperature: float = Field(0.5, ge=0.0, le=1.0)
    max_tokens: int = Field(10000, ge=1)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("top_p", mode="before")
    @classmethod
    def check_top_p(cls, v: Any, info: ValidationInfo) -> Any:
        if v is not None and info.data.get("temperature") is not None:
            raise ValueError("temperature and top_p cannot be set at the same time.")
        return v

    def create_client(self) -> None:  # type: ignore [override]
        raise NotImplementedError("DeepSeekLLMConfigEntry.create_client is not implemented.")
