# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable, Mapping, Sequence

import httpx
from fast_depends.library.serializer import SerializerProto
from nlip_sdk.nlip import NLIP_Message

from ag2.config.client import LLMClient
from ag2.context import ConversationContext
from ag2.events import BaseEvent, ModelMessage, ModelRequest, ModelResponse, TextInput, ToolCallsEvent, Usage
from ag2.response import ResponseProto
from ag2.tools.final.function_tool import FunctionToolSchema
from ag2.tools.schemas import ToolSchema

from .errors import NlipConnectionError, NlipInputRequiredError, NlipServerError, NlipTimeoutError
from .mappers import build_request_message, parse_response_message

_PROVIDER = "nlip"


class NlipClient(LLMClient):
    """``LLMClient`` that delegates a turn to a remote NLIP endpoint.

    NLIP is a stateless protocol: there is no server-side task/session to
    resume, so every call posts the full AG2 conversation history (as a
    JSON submessage) alongside the latest user-directed text, and the
    response is taken as the complete reply for the turn — there is no
    streaming and no polling.
    """

    def __init__(
        self,
        *,
        url: str,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = 60.0,
        max_retries: int = 3,
        httpx_client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._headers = dict(headers) if headers else None
        self._timeout = timeout
        self._max_retries = max_retries
        self._httpx_client_factory = httpx_client_factory

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: ConversationContext,
        *,
        tools: Iterable[ToolSchema],
        response_schema: ResponseProto | None,
        serializer: SerializerProto,
    ) -> ModelResponse:
        if response_schema is not None:
            raise NotImplementedError("response_schema is not supported with NlipConfig")

        function_schemas = [t for t in tools if isinstance(t, FunctionToolSchema)]
        text = _latest_text(messages)
        past_events = messages[:-1] if messages else ()

        outgoing = build_request_message(
            text,
            history_events=past_events,
            context=dict(context.variables) or None,
            tool_schemas=function_schemas,
        )

        nlip_response = await self._post(outgoing)
        parsed = parse_response_message(nlip_response)

        if parsed.input_required:
            raise NlipInputRequiredError(parsed.input_required)

        if parsed.context_update:
            context.variables.update(parsed.context_update)

        message = ModelMessage(parsed.text) if parsed.text else None
        return ModelResponse(
            message=message,
            tool_calls=ToolCallsEvent(parsed.tool_calls),
            usage=Usage(),
            provider=_PROVIDER,
            finish_reason="completed",
        )

    async def _post(self, message: NLIP_Message) -> NLIP_Message:
        # Only timeouts and connection failures are retried — they're
        # transient. A non-2xx HTTP status is a definitive server response
        # (not a transport failure), so it's raised immediately rather than
        # retried.
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                client = self._make_httpx_client()
                async with client:
                    response = await client.post(f"{self._url}/nlip/", json=message.to_dict())
                    response.raise_for_status()
                return NLIP_Message.model_validate(response.json())
            except httpx.TimeoutException as exc:
                last_error = exc
            except httpx.ConnectError as exc:
                last_error = exc
            except httpx.HTTPStatusError as exc:
                raise NlipServerError(status_code=exc.response.status_code, body=exc.response.text) from exc

        assert last_error is not None
        if isinstance(last_error, httpx.TimeoutException):
            raise NlipTimeoutError(
                f"Request to {self._url} timed out after {self._max_retries} attempts"
            ) from last_error
        raise NlipConnectionError(f"Failed to connect to {self._url}") from last_error

    def _make_httpx_client(self) -> httpx.AsyncClient:
        if self._httpx_client_factory is not None:
            return self._httpx_client_factory()
        return httpx.AsyncClient(timeout=self._timeout, headers=self._headers)


def _latest_text(messages: Sequence[BaseEvent]) -> str:
    for ev in reversed(messages):
        if isinstance(ev, ModelRequest):
            for part in reversed(ev.parts):
                if isinstance(part, TextInput) and part.content:
                    return part.content
    return ""


__all__: tuple[str, ...] = ("NlipClient",)
