# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

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
from ag2.middleware.base import AgentTurn, BaseMiddleware, LLMCall, MiddlewareFactory

from .events import A2UIClientEvent, A2UIMessageEvent, A2UIValidationFailedEvent
from .incoming import A2UIIncomingParseResult
from .parser import A2UIParseResult, A2UIResponseParser, A2UIValidationResult

logger = logging.getLogger(__name__)


def _to_prose_message(original: ModelMessage | None, text: str) -> ModelMessage:
    """Rebuild a prose-only ``ModelMessage`` (A2UI content out-of-band), preserving metadata."""
    metadata = dict(original.metadata) if original is not None and original.metadata else {}
    return ModelMessage(text, metadata=metadata)


async def _publish_a2ui(parse_result: A2UIParseResult, response: ModelResponse, context: Context) -> None:
    """Publish a parsed A2UI block and strip it from the durable response.

    Emits one :class:`A2UIMessageEvent` per parsed message onto the stream and
    rewrites ``response.message`` to prose only, so the A2UI block travels
    out-of-band of the text channel (the protocol carries prose and UI messages
    separately). A no-op when the response has no A2UI block. When a block is
    present but unparsable, there are no operations to emit, yet the block is
    still stripped from the prose so raw JSON never leaks into the text.

    Shared by :class:`A2UIExtractionMiddleware` (always on) and
    :class:`A2UIValidationMiddleware` (after a response validates).
    """
    if not parse_result.has_a2ui:
        return
    for op in parse_result.operations:
        await context.send(A2UIMessageEvent(op))
    response.message = _to_prose_message(response.message, parse_result.text)


class A2UIExtractionMiddleware(MiddlewareFactory):
    """Factory that builds the always-on A2UI extraction middleware per turn.

    Wraps ``on_llm_call`` to parse the model's response for an A2UI block,
    publish it as out-of-band :class:`A2UIMessageEvent`s, and strip it from the
    durable text — *without* any schema validation. This is the seam that makes
    A2UI work; validation (:class:`A2UIValidationMiddleware`) is a separate,
    optional layer that performs the same publication after it validates.

    Used when ``validate_responses=False``: the model's UI is trusted as-is and
    the client validates/degrades, but the wire still stays spec-compliant
    (clean prose + structured UI messages, never raw JSON in the text).
    """

    def __init__(self, parser: A2UIResponseParser) -> None:
        self._parser = parser

    def __call__(self, event: BaseEvent, context: Context) -> BaseMiddleware:
        return _A2UIExtractionMiddleware(event, context, parser=self._parser)


class _A2UIExtractionMiddleware(BaseMiddleware):
    """The per-turn instance used by :class:`A2UIExtractionMiddleware`."""

    def __init__(self, event: BaseEvent, context: Context, *, parser: A2UIResponseParser) -> None:
        super().__init__(event, context)
        self._parser = parser

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        response = await call_next(events, context)
        # A2UI lives in `content`, tool calls in `tool_calls` — orthogonal. Publish
        # any UI from content (a no-op if none) and leave tool_calls for the executor.
        text = response.content
        if not text:
            return response
        await _publish_a2ui(self._parser.parse(text), response, context)
        return response


class A2UIInboundMiddleware(MiddlewareFactory):
    """Factory that emits one :class:`A2UIClientEvent` per incoming interaction.

    Built per-request with the client→server interactions parsed from the
    request (button clicks, v1.0 ``functionResponse``s, client errors). The
    per-turn instance emits them on the stream at the start of the turn — inside
    the turn, where the agent's observers are subscribed — so server-side code
    can react to client interactions, mirroring how :class:`A2UIMessageEvent`
    surfaces server→client messages. The envelopes are still rewritten into the
    turn's prompt separately; this is an observability seam, not the prompt path.
    """

    def __init__(self, interactions: Sequence[A2UIIncomingParseResult]) -> None:
        self._interactions = tuple(interactions)

    def __call__(self, event: BaseEvent, context: Context) -> BaseMiddleware:
        return _A2UIInboundMiddleware(event, context, self._interactions)


class _A2UIInboundMiddleware(BaseMiddleware):
    """The per-turn instance used by :class:`A2UIInboundMiddleware`."""

    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        interactions: Sequence[A2UIIncomingParseResult],
    ) -> None:
        super().__init__(event, context)
        self._interactions = interactions

    async def on_turn(self, call_next: AgentTurn, event: BaseEvent, context: Context) -> ModelResponse:
        for interaction in self._interactions:
            await context.send(A2UIClientEvent(interaction))
        return await call_next(event, context)


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
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
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

        for attempt in range(self._max_retries + 1):
            response = await call_next(current_events, context)

            # Tool-call turns aren't retried (that fights the executor): extract the
            # UI once — publish if valid, else degrade — and return with tool_calls.
            if response.tool_calls and response.tool_calls.calls:
                text = response.content
                if text:
                    parse_result, validation_errors = self._validate(text)
                    if validation_errors is None:
                        await _publish_a2ui(parse_result, response, context)
                    else:
                        await context.send(A2UIValidationFailedEvent(validation_errors, attempt + 1))
                        response.message = _to_prose_message(response.message, parse_result.text)
                return response
            text = response.content
            if not text:
                return response

            parse_result, validation_errors = self._validate(text)
            if validation_errors is None:
                # Valid (or no A2UI at all). Publish the validated block out-of-band
                # and keep the durable response prose-only — transports consume the
                # events. Same publication path as the extraction middleware.
                await _publish_a2ui(parse_result, response, context)
                return response

            if attempt >= self._max_retries:
                logger.warning(
                    "A2UI validation failed after %d attempt(s). Returning text-only response.",
                    attempt + 1,
                )
                # Emit a typed observability event before degrading. The A2UI
                # spec has no server→client error frame, so the wire stays
                # prose-only (graceful degradation); this event lets observers
                # tell a failed UI apart from an intentional text reply.
                await context.send(A2UIValidationFailedEvent(validation_errors, attempt + 1))
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
            # Wrap the bad turn in a ModelResponse: provider mappers only render
            # ModelResponse/ModelRequest/ToolResults events, so a bare
            # ModelMessage would be silently dropped and the LLM would never see
            # what it got wrong — defeating the corrective retry.
            current_events = current_events + [
                ModelResponse(ModelMessage(text)),
                ModelRequest([TextInput(feedback)]),
            ]

        # Unreachable: the loop runs at least once (``max_retries >= 0``) and its
        # final attempt (``attempt >= self._max_retries``) always returns.
        raise AssertionError("A2UI validation loop exited without returning a response")

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
