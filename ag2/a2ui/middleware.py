# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Per-turn A2UI middleware. Register with ``Middleware(cls, **kwargs)``. Not exported."""

import logging
from collections.abc import Sequence

from ag2.annotations import Context
from ag2.events import (
    BaseEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
)
from ag2.middleware.base import AgentTurn, BaseMiddleware, LLMCall

from .events import A2UIClientEvent, A2UIMessageEvent, A2UIValidationFailedEvent
from .incoming import A2UIIncomingParseResult
from .parser import A2UIParseResult, A2UIResponseParser

logger = logging.getLogger(__name__)


def _to_prose_message(original: ModelMessage | None, text: str) -> ModelMessage:
    """Rebuild a prose-only ``ModelMessage`` (A2UI content out-of-band), preserving metadata."""
    metadata = dict(original.metadata) if original is not None and original.metadata else {}
    return ModelMessage(text, metadata=metadata)


async def _publish_a2ui(
    parse_result: A2UIParseResult,
    response: ModelResponse,
    context: Context,
) -> None:
    """Emit one :class:`A2UIMessageEvent` per parsed message and strip the block from the response text.

    A no-op without an A2UI block. An unparsable block is still stripped, so raw JSON never reaches the prose.
    """
    if not parse_result.has_a2ui:
        return
    for op in parse_result.operations:
        await context.send(A2UIMessageEvent(op))
    response.message = _to_prose_message(response.message, parse_result.text)


class A2UIExtractionMiddleware(BaseMiddleware):
    """Publish the response's A2UI block out-of-band without validating it (``validate_responses=False``)."""

    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        *,
        parser: A2UIResponseParser,
    ) -> None:
        super().__init__(event, context)
        self._parser = parser

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        response = await call_next(events, context)
        # A2UI lives in `content`, tool calls in `tool_calls`: publish the UI and leave tool calls to the executor.
        text = response.content
        if not text:
            return response
        await _publish_a2ui(self._parser.parse(text), response, context)
        return response


class A2UIInboundMiddleware(BaseMiddleware):
    """Emit one :class:`A2UIClientEvent` per incoming interaction at the start of the turn."""

    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        *,
        interactions: Sequence[A2UIIncomingParseResult],
    ) -> None:
        super().__init__(event, context)
        self._interactions = interactions

    async def on_turn(self, call_next: AgentTurn, event: BaseEvent, context: Context) -> ModelResponse:
        for interaction in self._interactions:
            await context.send(A2UIClientEvent(interaction))
        return await call_next(event, context)


class A2UIValidationMiddleware(BaseMiddleware):
    """Validate the response's A2UI block, retrying with feedback up to ``max_retries`` times.

    A valid block is published out-of-band. When retries run out, the block is dropped, the reply degrades to
    prose, and an :class:`A2UIValidationFailedEvent` is emitted.
    """

    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        *,
        parser: A2UIResponseParser,
        max_retries: int,
    ) -> None:
        super().__init__(event, context)
        self._parser = parser
        self._max_retries = max_retries

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        current_events: list[BaseEvent] = list(events)
        attempt = 0
        while True:
            attempt += 1
            response = await call_next(current_events, context)
            text = response.content
            if not text:
                return response

            parse_result, validation_errors = self._validate(text)
            if validation_errors is None:
                await _publish_a2ui(parse_result, response, context)
                return response

            # Tool-call turns are not retried, since that fights the tool executor.
            has_tool_calls = bool(response.tool_calls and response.tool_calls.calls)
            if has_tool_calls or attempt > self._max_retries:
                if not has_tool_calls:
                    logger.warning("A2UI validation failed after %d attempt(s). Returning text-only response.", attempt)
                await context.send(A2UIValidationFailedEvent(validation_errors, attempt))
                response.message = _to_prose_message(response.message, parse_result.text)
                return response

            logger.info("A2UI validation failed (attempt %d/%d). Retrying.", attempt, self._max_retries)
            logger.debug("Validation errors: %s", validation_errors)
            feedback = self._parser.format_validation_error(validation_errors)
            # Provider mappers render only ModelResponse/ModelRequest events, so the bad turn must be wrapped.
            current_events = current_events + [
                ModelResponse(ModelMessage(text)),
                ModelRequest([TextInput(feedback)]),
            ]

    def _validate(self, response_text: str) -> "tuple[A2UIParseResult, list[str] | None]":
        """Parse and validate an A2UI response; ``errors=None`` means valid (or no A2UI content)."""
        parse_result = self._parser.parse(response_text)
        if not parse_result.has_a2ui:
            return parse_result, None
        if parse_result.parse_error:
            return parse_result, [parse_result.parse_error]
        validation_result = self._parser.validate(parse_result.operations)
        if validation_result.is_valid:
            return parse_result, None
        return parse_result, validation_result.errors
