# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from autogen.beta.context import Context
from autogen.beta.events import BaseEvent, ModelResponse
from autogen.beta.response import ResponseProto
from autogen.beta.tools.schemas import ToolSchema


@runtime_checkable
class LLMClient(Protocol):
    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        *,
        tools: Iterable[ToolSchema],
        response_schema: ResponseProto | None,
    ) -> ModelResponse: ...
