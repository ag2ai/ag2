# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import httpx2
from fast_depends.library.serializer import SerializerProto
from google import genai

from ag2.config.client import LLMClient
from ag2.context import ConversationContext
from ag2.events import BaseEvent, ModelMessage, ModelResponse
from ag2.response import ResponseProto
from ag2.tools.schemas import ToolSchema

from ._compat import build_http_options
from .decisions_mappers import (
    PROVIDER,
    answers_to_reply,
    blocked_error,
    convert_input,
    normalize_usage,
    question_instructions,
    response_proto_to_questions,
    tool_to_api,
)

__all__ = ["GeminiDecisionsClient"]


class GeminiDecisionsClient(LLMClient):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_version: str = "v1beta",
        http_client: httpx2.AsyncClient | None = None,
        boolean_threshold: float = 0.5,
        descriptions: Mapping[str, str] | None = None,
        question_instructions: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        self._client = genai.Client(
            vertexai=False,
            api_key=api_key,
            http_options=build_http_options(http_client, None),
        )
        self._model = model
        self._api_version = api_version
        self._boolean_threshold = boolean_threshold
        self._descriptions = descriptions
        self._question_instructions = question_instructions
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

        questions = response_proto_to_questions(
            response_schema,
            instructions=question_instructions(self._question_instructions, "\n".join(s for s in context.prompt if s)),
            descriptions=self._descriptions,
        )

        if not (decision_input := convert_input(messages, serializer)):
            raise ValueError("A Gemini decision needs input: the conversation has nothing to send.")

        try:
            decision = await self._client.aio.decisions.create(
                model=self._model,
                input=decision_input,
                questions=questions,
                api_version=self._api_version,
                extra_body=self._extra_body,
            )
        except Exception as e:
            if (blocked := blocked_error(e)) is not None:
                raise blocked from e
            raise

        content, metadata = answers_to_reply(response_schema, decision, boolean_threshold=self._boolean_threshold)
        model_msg = ModelMessage(content, metadata=metadata)
        await context.send(model_msg)

        return ModelResponse(
            message=model_msg,
            usage=normalize_usage(decision.usage),
            model=self._model,
            provider=PROVIDER,
        )
