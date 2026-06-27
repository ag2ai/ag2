# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from fast_depends.library.serializer import SerializerProto

from ag2.context import ConversationContext
from ag2.events import BaseEvent, ModelResponse
from ag2.response import ResponseProto
from ag2.tools.schemas import ToolSchema


@runtime_checkable
class LLMClient(Protocol):
    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: ConversationContext,
        *,
        tools: Iterable[ToolSchema],
        response_schema: ResponseProto | None,
        serializer: SerializerProto,
    ) -> ModelResponse: ...
