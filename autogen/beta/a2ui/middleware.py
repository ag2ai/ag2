# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence

from autogen.beta.annotations import Context
from autogen.beta.events import (
    BaseEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
)
from autogen.beta.middleware.base import BaseMiddleware, LLMCall, MiddlewareFactory

from .events import A2UIMessageEvent
from .parser import A2UIParseResult, A2UIResponseParser, A2UIValidationResult

logger = logging.getLogger(__name__)


def _to_prose_message(original: ModelMessage | None, text: str) -> ModelMessage:
    """Rebuild a prose-only ``ModelMessage`` while preserving original metadata.

    A2UI content is carried out-of-band as :class:`A2UIMessageEvent`s, so the
    durable ``ModelMessage`` keeps only the conversational text. This is also
    the graceful-degradation path. Rebuilding the message naively would drop
    any ``metadata`` the provider attached (e.g. tracing/usage hints), so it
    is carried over here.
    """
    metadata = dict(original.metadata) if original is not None and original.metadata else {}
    return ModelMessage(text, metadata=metadata)


class A2UIValidationMiddleware(MiddlewareFactory):
    """Factory that builds an A2UI validation middleware per turn.

    Wraps ``on_llm_call`` so the LLM's response text is parsed for A2UI JSON
    and validated against the catalog schema. On validation failure, appends
    the bad response plus a corrective user message to the events list and
    retries the call. After ``max_retries + 1`` total attempts, the middleware
    returns the last response with the A2UI JSON stripped from its content
    (graceful degradation to text-only).
    """

    def __init__(self, parser: A2UIResponseParser, max_retries: int = 1) -> None:
        self._parser = parser
        self._max_retries = max_retries

    def __call__(self, event: BaseEvent, context: Context) -> BaseMiddleware:
        return _A2UIValidationMiddleware(
            event,
            context,
            parser=self._parser,
            max_retries=self._max_retries,
        )


class _A2UIValidationMiddleware(BaseMiddleware):
    """The per-turn instance used by :class:`A2UIValidationMiddleware`."""

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
        last_parse_result: A2UIParseResult | None = None
        response: ModelResponse | None = None

        for attempt in range(self._max_retries + 1):
            response = await call_next(current_events, context)

            # Tool calls / empty content — middleware is not concerned.
            if response.tool_calls and response.tool_calls.calls:
                return response
            text = response.content
            if not text:
                return response

            parse_result, validation_errors = self._validate(text)
            if validation_errors is None:
                # Valid (or no A2UI at all). When A2UI content is present, emit
                # one A2UIMessageEvent per message onto the stream and keep the
                # durable response prose-only — transports consume the events.
                if parse_result.has_a2ui:
                    for op in parse_result.operations:
                        await context.send(A2UIMessageEvent(op))
                    response.message = _to_prose_message(response.message, parse_result.text)
                return response

            last_parse_result = parse_result

            if attempt >= self._max_retries:
                logger.warning(
                    "A2UI validation failed after %d attempt(s). Returning text-only response.",
                    attempt + 1,
                )
                response.message = _to_prose_message(response.message, parse_result.text)
                return response

            logger.info(
                "A2UI validation failed (attempt %d/%d). Retrying.",
                attempt + 1,
                self._max_retries,
            )
            logger.debug("Validation errors: %s", validation_errors)

            feedback = self._parser.format_validation_error(
                parse_result,
                A2UIValidationResult(is_valid=False, errors=validation_errors),
            )
            current_events = current_events + [
                ModelMessage(text),
                ModelRequest([TextInput(feedback)]),
            ]

        # ``response`` is guaranteed to be set because the loop runs at least once.
        # This branch only executes if the loop completes without an early return,
        # which currently cannot happen (the final attempt always returns above).
        assert response is not None
        if last_parse_result is not None:
            response.message = _to_prose_message(response.message, last_parse_result.text)
        return response

    def _validate(self, response_text: str) -> "tuple[A2UIParseResult, list[str] | None]":
        """Parse and validate an A2UI response.

        Returns ``(parse_result, errors)`` — ``errors=None`` means valid (or
        no A2UI content at all). ``errors=[..]`` means the response had A2UI
        content that failed validation or JSON parsing.
        """
        parse_result = self._parser.parse(response_text)
        if not parse_result.has_a2ui:
            return parse_result, None
        if parse_result.parse_error:
            return parse_result, [parse_result.parse_error]
        validation_result = self._parser.validate(parse_result.operations)
        if validation_result.is_valid:
            return parse_result, None
        return parse_result, validation_result.errors
