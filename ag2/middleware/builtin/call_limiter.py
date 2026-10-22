# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from ag2.annotations import Context
from ag2.events import (
    BaseEvent,
    BuiltinToolCallEvent,
    CallLimitReached,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
)
from ag2.middleware.base import BaseMiddleware, LLMCall, MiddlewareFactory, ToolExecution, ToolResultType
from ag2.middleware.describe import MiddlewareDescription

logger = logging.getLogger(__name__)

#: ``ModelMessage.metadata`` key of a reply the limiter shaped: ``{"reason": ..., "ending": ...}``.
CALL_LIMIT_KEY = "ag2:call_limit"
#: ``ModelResponse.finish_reason`` of a reply no model call wrote.
CALL_LIMIT_FINISH_REASON = "call_limit"
#: ``error.type`` of the result a refused tool call gets.
CALL_LIMIT_ERROR_TYPE = "call_limit_exceeded"

DEFAULT_SYNTHESIS_PROMPT = (
    "You have used your call budget. Answer now from the information you already have, "
    "and state clearly what you could not verify. Start no new work with tools: such calls "
    "are refused from here on. Write your final answer as text."
)
TEXT_ONLY_PROMPT = (
    "Tools are no longer available and your last reply only called them. Reply with your final "
    "answer as text: what you found, and what you could not verify."
)

#: Model calls an exhausted turn gets to write its answer: the final call, and one more
#: for text when its reply was only refused tool calls or exempt calls that have now run.
_FINAL_CALLS = 2

OnExhausted: TypeAlias = Literal["synthesise", "stop"]
Ending: TypeAlias = Literal["empty", "last_text"]
LimitResult: TypeAlias = str | dict[str, Any] | Callable[[ToolCallEvent], str | dict[str, Any]]


@dataclass(slots=True)
class CallCounter:
    """What a ``CallLimiter`` has spent and refused.

    A fresh counter per turn is the default. Pass one to ``CallLimiter(counter=...)``
    to share it, for example across every agent of a delegation through
    ``context.dependencies``. ``max_seconds`` runs from ``started``, which defaults
    to the counter's creation; set it to line the clock up with a host's own.
    """

    clock: Callable[[], float] = time.monotonic
    started: float | None = None

    tool_calls: int = 0
    llm_calls: int = 0
    per_tool: dict[str, int] = field(default_factory=dict)

    #: Tool calls refused without running.
    denied: int = 0
    denied_per_tool: dict[str, int] = field(default_factory=dict)
    #: Model calls the limiter ended the turn on instead of dispatching.
    llm_denied: int = 0

    def __post_init__(self) -> None:
        if self.started is None:
            self.started = self.clock()

    def elapsed(self) -> float:
        return self.clock() - (self.started or 0.0)


CounterSource: TypeAlias = CallCounter | Callable[[Context], CallCounter]


@dataclass(frozen=True, slots=True)
class _Limits:
    max_tool_calls: int | None
    max_tool_calls_per_tool: Mapping[str, int]
    max_llm_calls: int | None
    max_seconds: float | None
    exempt: frozenset[str]
    on_exhausted: OnExhausted
    ending: Ending
    synthesis_prompt: str
    show_remaining: bool
    tool_limit_result: LimitResult | None
    exhausted_result: LimitResult | None


class _LimitPrompt(ModelRequest):
    """The prompt of a limiter's final call. Marks that call, so a ``CallLimiter``
    further in lets it through uncounted instead of refusing it a second time."""


class CallLimiter(MiddlewareFactory):
    """Cap the tool and model calls of a turn, then have the model answer.

    A capped tool call is refused rather than run: the model gets a structured
    ``call_limit_exceeded`` result and the tool schemas never change, so the
    provider's prompt cache stays intact. A ``max_tool_calls_per_tool`` cap
    refuses that tool only and the turn goes on. A used-up total, model-call cap
    or ``max_seconds`` exhausts the turn: with ``on_exhausted="synthesise"`` the
    model gets a final call asking it to answer from what it has, and at most one
    more for text. After those, or at once under ``"stop"``, the turn ends with no
    model call.

    A reply no model call wrote has ``finish_reason="call_limit"``, and every
    ending sends :class:`~ag2.events.CallLimitReached`. Tools in ``exempt`` are
    never counted or refused.

    Counts are per agent and per turn unless ``counter`` shares a
    :class:`CallCounter`.
    """

    def __init__(
        self,
        *,
        max_tool_calls: int | None = None,
        max_tool_calls_per_tool: Mapping[str, int] | None = None,
        max_llm_calls: int | None = None,
        max_seconds: float | None = None,
        exempt: Iterable[str] = (),
        on_exhausted: OnExhausted = "synthesise",
        ending: Ending = "empty",
        synthesis_prompt: str = DEFAULT_SYNTHESIS_PROMPT,
        show_remaining: bool = False,
        tool_limit_result: LimitResult | None = None,
        exhausted_result: LimitResult | None = None,
        counter: CounterSource | None = None,
    ) -> None:
        per_tool = dict(max_tool_calls_per_tool or {})
        for name, cap in (("max_tool_calls", max_tool_calls), ("max_llm_calls", max_llm_calls), *per_tool.items()):
            if cap is not None and cap < 0:
                raise ValueError(f"{name} must not be negative")
        if max_seconds is not None and max_seconds < 0:
            raise ValueError("max_seconds must not be negative")
        if on_exhausted not in ("synthesise", "stop"):
            raise ValueError("on_exhausted must be 'synthesise' or 'stop'")
        if ending not in ("empty", "last_text"):
            raise ValueError("ending must be 'empty' or 'last_text'")

        self._limits = _Limits(
            max_tool_calls=max_tool_calls,
            max_tool_calls_per_tool=per_tool,
            max_llm_calls=max_llm_calls,
            max_seconds=max_seconds,
            exempt=frozenset(exempt),
            on_exhausted=on_exhausted,
            ending=ending,
            synthesis_prompt=synthesis_prompt,
            show_remaining=show_remaining,
            tool_limit_result=tool_limit_result,
            exhausted_result=exhausted_result,
        )
        self._counter = counter

    def describe(self) -> MiddlewareDescription:
        limits = self._limits
        return MiddlewareDescription(
            kind=type(self).__qualname__,
            config={
                "max_tool_calls": limits.max_tool_calls,
                "max_tool_calls_per_tool": dict(limits.max_tool_calls_per_tool),
                "max_llm_calls": limits.max_llm_calls,
                "max_seconds": limits.max_seconds,
                "exempt": sorted(limits.exempt),
                "on_exhausted": limits.on_exhausted,
                "ending": limits.ending,
                "show_remaining": limits.show_remaining,
                "shared_counter": self._counter is not None,
            },
        )

    def __call__(self, event: BaseEvent, context: Context) -> BaseMiddleware:
        source = self._counter
        if source is None:
            counter = CallCounter()
        elif isinstance(source, CallCounter):
            counter = source
        else:
            counter = source(context)
        return _CallLimiter(event, context, self._limits, counter)


class _CallLimiter(BaseMiddleware):
    def __init__(self, event: BaseEvent, context: Context, limits: _Limits, counter: CallCounter) -> None:
        super().__init__(event, context)
        self._limits = limits
        self._counter = counter
        #: Why this turn is exhausted; ``None`` while it is not.
        self._exhausted: str | None = None
        #: Model calls left to write the answer once the turn is exhausted.
        self._final_calls = _FINAL_CALLS

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        name = event.name
        if name in self._limits.exempt:
            return await call_next(event, context)

        # Totals first: a call at both a total and its own cap exhausts the turn.
        reason = self._exhausted or self._tool_total_hit()
        if reason is not None:
            self._exhausted = reason
            self._deny(name)
            return ToolResultEvent.from_call(event, self._exhausted_result(event, reason))

        cap = self._limits.max_tool_calls_per_tool.get(name)
        if cap is not None and self._counter.per_tool.get(name, 0) >= cap:
            self._deny(name)
            return ToolResultEvent.from_call(event, self._tool_limit_result(event))

        self._counter.tool_calls += 1
        self._counter.per_tool[name] = self._counter.per_tool.get(name, 0) + 1
        return await call_next(event, context)

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        if events and isinstance(events[-1], _LimitPrompt):
            # An outer limiter's final call: it already counted and shaped it.
            return await call_next(events, context)

        if self._exhausted is None:
            self._exhausted = self._llm_hit()
        if self._exhausted is not None:
            if self._limits.on_exhausted == "synthesise" and self._final_calls:
                return await self._synthesise(call_next, events, context)
            self._counter.llm_denied += 1
            return await self._end(events, context)

        self._counter.llm_calls += 1
        if self._limits.show_remaining and (note := self._remaining_note()) is not None:
            # This call only: the note is never part of the history.
            events = [*events, ModelRequest.ensure_request([note])]
        return await call_next(events, context)

    def _tool_total_hit(self) -> str | None:
        limits, counter = self._limits, self._counter
        if limits.max_seconds is not None and counter.elapsed() >= limits.max_seconds:
            return "max_seconds"
        if limits.max_tool_calls is not None and counter.tool_calls >= limits.max_tool_calls:
            return "max_tool_calls"
        return None

    def _llm_hit(self) -> str | None:
        limits, counter = self._limits, self._counter
        if limits.max_seconds is not None and counter.elapsed() >= limits.max_seconds:
            return "max_seconds"
        if limits.max_llm_calls is not None and counter.llm_calls >= limits.max_llm_calls:
            return "max_llm_calls"
        return None

    def _deny(self, name: str) -> None:
        self._counter.denied += 1
        self._counter.denied_per_tool[name] = self._counter.denied_per_tool.get(name, 0) + 1

    async def _synthesise(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        """A final call, answered as text. At most two are made per turn.

        Exempt calls in the reply stand: they are how a turn hands off or records
        its result, and once they have run the model gets the second final call to
        answer. Other tool calls are dropped and the reply's text kept. A reply of
        nothing but dropped calls is asked once more for text.
        """
        assert self._exhausted is not None
        prompt: list[BaseEvent] = [*events, _LimitPrompt.ensure_request([self._synthesis_prompt()])]
        while self._final_calls:
            self._final_calls -= 1
            self._counter.llm_calls += 1
            try:
                response = await call_next(prompt, context)
            except Exception as e:
                # The caller answers from what it has; a failed final call does not fail the turn.
                return await self._end(events, context, error=e)

            calls = response.tool_calls.calls
            kept = [c for c in calls if isinstance(c, BuiltinToolCallEvent) or c.name in self._limits.exempt]
            text = response.message.content.strip() if response.message else ""
            if kept or text or not calls:
                if len(kept) != len(calls):
                    logger.warning(
                        "CallLimiter dropped tool calls %s from the final reply (%s)",
                        [c.name for c in calls if all(c is not k for k in kept)],
                        self._exhausted,
                    )
                    response.tool_calls = ToolCallsEvent(kept)
                return await self._answered(response, context)
            prompt = [*prompt, _LimitPrompt.ensure_request([TEXT_ONLY_PROMPT])]

        return await self._end(events, context)

    def _synthesis_prompt(self) -> str:
        prompt = self._limits.synthesis_prompt
        if self._limits.exempt:
            # "Start no new work with tools" otherwise reads as forbidding the call
            # that records or hands off the result.
            prompt += f" These tools are not refused: {', '.join(sorted(self._limits.exempt))}."
        return prompt

    async def _answered(self, response: ModelResponse, context: Context) -> ModelResponse:
        assert self._exhausted is not None
        if response.message is not None:
            response.message.metadata[CALL_LIMIT_KEY] = {"reason": self._exhausted, "ending": "synthesis"}
        if not response.tool_calls:
            # Exempt calls still to run mean the turn has not ended yet.
            await context.send(CallLimitReached(reason=self._exhausted, ending="synthesis"))
        return response

    async def _end(
        self,
        events: Sequence[BaseEvent],
        context: Context,
        error: Exception | None = None,
    ) -> ModelResponse:
        """End the turn without a model call."""
        assert self._exhausted is not None
        text = _last_text(events) if self._limits.ending == "last_text" else None
        ending = "last_text" if text else "empty"
        await context.send(
            CallLimitReached(
                reason=self._exhausted,
                ending=ending,
                error_type=type(error).__name__ if error is not None else None,
                error=str(error) if error is not None else None,
            )
        )
        # An empty ending carries no message: an empty text block replayed in a
        # later turn is rejected by some providers.
        message = (
            ModelMessage(text, metadata={CALL_LIMIT_KEY: {"reason": self._exhausted, "ending": ending}})
            if text
            else None
        )
        return ModelResponse(message, finish_reason=CALL_LIMIT_FINISH_REASON)

    def _exhausted_result(self, event: ToolCallEvent, reason: str) -> str | dict[str, Any]:
        if self._limits.exhausted_result is not None:
            return _resolve(self._limits.exhausted_result, event)
        exempt = sorted(self._limits.exempt)
        still = f"; {', '.join(exempt)} remain available" if exempt else ""
        return _error(
            f"call limit reached ({reason}): no further tool calls{still}",
            ["answer now from the information you already have"],
        )

    def _tool_limit_result(self, event: ToolCallEvent) -> str | dict[str, Any]:
        if self._limits.tool_limit_result is not None:
            return _resolve(self._limits.tool_limit_result, event)
        return _error(
            f"call limit reached for {event.name} (max_tool_calls_per_tool): do not call it again",
            ["continue with the information already gathered", "other tools remain available"],
        )

    def _remaining_note(self) -> str | None:
        """What is left after the call the note rides on."""
        limits, counter = self._limits, self._counter
        parts: list[str] = []
        if limits.max_seconds is not None:
            parts.append(f"{max(limits.max_seconds - counter.elapsed(), 0.0):.0f} s of time")
        if limits.max_tool_calls is not None:
            parts.append(f"{max(limits.max_tool_calls - counter.tool_calls, 0)} of {limits.max_tool_calls} tool calls")
        if limits.max_llm_calls is not None:
            parts.append(f"{max(limits.max_llm_calls - counter.llm_calls, 0)} of {limits.max_llm_calls} model calls")
        for name, cap in sorted(limits.max_tool_calls_per_tool.items()):
            parts.append(f"{name}: {max(cap - counter.per_tool.get(name, 0), 0)} of {cap} calls")
        if not parts:
            return None
        return f"[call limit] Remaining: {'; '.join(parts)}. Plan to answer before these run out."


def _resolve(result: LimitResult, event: ToolCallEvent) -> str | dict[str, Any]:
    return result(event) if callable(result) else result


def _error(message: str, recovery: list[str]) -> dict[str, Any]:
    return {
        "error": {
            "type": CALL_LIMIT_ERROR_TYPE,
            "message": message,
            "recovery": recovery,
            "retryable": False,
        }
    }


def _last_text(events: Sequence[BaseEvent]) -> str | None:
    """The model's last non-empty text in this turn, back to the request that began it."""
    for event in reversed(events):
        if isinstance(event, ModelRequest):
            return None
        if isinstance(event, ModelResponse) and event.message and event.message.content.strip():
            return event.message.content
    return None
