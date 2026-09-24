# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import httpx2
from fast_depends.library.serializer import SerializerProto
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from ag2.config.client import LLMClient
from ag2.context import ConversationContext
from ag2.events import BaseEvent, ModelMessage, ModelResponse
from ag2.response import ResponseProto
from ag2.tools.schemas import ToolSchema

from .mappers import (
    ANSWER_KEY,
    PROVIDER,
    answer_metadata,
    answer_to_content,
    convert_state,
    normalize_usage,
    response_proto_to_question,
    tool_to_api,
)

__all__ = ["TypeSafeClient"]


class TypeSafeClient(LLMClient):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | httpx2.Timeout | None = None,
        retry: RetryPolicy | None = None,
        headers: Mapping[str, str] | None = None,
        http_client: httpx2.AsyncClient | None = None,
        boolean_threshold: float = 0.5,
        criteria: Mapping[str, str] | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._retry = retry
        self._headers = headers
        self._http_client = http_client
        self._boolean_threshold = boolean_threshold
        self._criteria = criteria
        self._client: AsyncTypeSafeClient | None = None

    def _get_client(self) -> AsyncTypeSafeClient:
        if self._client is None:
            self._client = AsyncTypeSafeClient(
                api_key=self._api_key,
                model=self._model,
                retry=self._retry,
                timeout=self._timeout,
                headers=self._headers,
                http_client=self._http_client,
                base_url=self._base_url,
            )
        return self._client

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
            criteria=self._criteria,
        )

        response = await self._get_client().system_one(
            state=convert_state(messages, serializer),
            questions={ANSWER_KEY: question},
        )

        answer = response.answers.get(ANSWER_KEY)
        if answer is None:
            raise ValueError(f"TypeSafe returned no answer for question {ANSWER_KEY!r}.")

        model_msg = ModelMessage(
            answer_to_content(response_schema, answer, boolean_threshold=self._boolean_threshold),
            metadata=answer_metadata(answer),
        )
        await context.send(model_msg)

        return ModelResponse(
            message=model_msg,
            usage=normalize_usage(response.usage),
            model=response.model,
            provider=PROVIDER,
        )
