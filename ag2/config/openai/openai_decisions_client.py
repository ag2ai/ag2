# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import httpx2
from fast_depends.library.serializer import SerializerProto
from openai import DEFAULT_MAX_RETRIES, AsyncOpenAI, Omit, not_given, omit

from ag2.config.client import LLMClient
from ag2.context import ConversationContext
from ag2.events import BaseEvent, ModelMessage, ModelResponse
from ag2.response import ResponseProto
from ag2.tools.schemas import ToolSchema

from .decisions_mappers import (
    PROVIDER,
    answer_metadata,
    answer_to_content,
    convert_input,
    find_answer,
    normalize_usage,
    response_proto_to_question,
    tool_to_api,
)

__all__ = ["OpenAIDecisionsClient"]


class OpenAIDecisionsClient(LLMClient):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        timeout: Any = not_given,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: dict[str, str] | None = None,
        default_query: dict[str, object] | None = None,
        http_client: httpx2.AsyncClient | None = None,
        safety_identifier: str | None | Omit = omit,
        boolean_threshold: float = 0.5,
        descriptions: Mapping[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
        )
        self._model = model
        self._safety_identifier = safety_identifier
        self._boolean_threshold = boolean_threshold
        self._descriptions = descriptions
        self._extra_body = extra_body

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: "ConversationContext",
        *,
        tools: Iterable[ToolSchema],
        response_schema: ResponseProto[Any] | None,
        serializer: SerializerProto,
    ) -> ModelResponse:
        for t in tools:
            tool_to_api(t)

        question = response_proto_to_question(
            response_schema,
            instructions="\n".join(s for s in context.prompt if s) or None,
            descriptions=self._descriptions,
        )

        decision = await self._client.decisions.create(
            model=self._model,
            input=convert_input(messages, serializer),
            questions=[question],
            safety_identifier=self._safety_identifier,
            extra_body=self._extra_body,
        )

        answer = find_answer(decision)
        model_msg = ModelMessage(
            answer_to_content(response_schema, answer, boolean_threshold=self._boolean_threshold),
            metadata=answer_metadata(answer),
        )
        await context.send(model_msg)

        return ModelResponse(
            message=model_msg,
            usage=normalize_usage(decision.usage),
            model=decision.model,
            provider=PROVIDER,
        )
