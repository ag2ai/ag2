# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias
from uuid import UUID, uuid4

from ag2.annotations import Context
from ag2.events import (
    BaseEvent,
    BuiltinToolCallEvent,
    CallLimitReached,
    DrainedModelRequest,
    Field,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResult,
    ToolResultEvent,
    UsageEvent,
)
from ag2.middleware.base import AgentTurn, BaseMiddleware, LLMCall, MiddlewareFactory, ToolExecution, ToolResultType
from ag2.middleware.describe import MiddlewareDescription

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
    """The additional instruction carried by one final model call."""


class _RemainingNote(DrainedModelRequest):
    """Persistent budget context, without starting a new user turn or model call."""

    note_id: UUID = Field(default_factory=uuid4)


class CallLimiter(MiddlewareFactory):
    """Cap the tool and model calls of a turn, then have the model answer.

    A capped tool call is refused rather than run: the model gets a structured
    ``call_limit_exceeded`` result and the tool schemas stay unchanged for
    prompt caching. A ``max_tool_calls_per_tool`` cap
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
            complete=(
                limits.synthesis_prompt == DEFAULT_SYNTHESIS_PROMPT
                and limits.tool_limit_result is None
                and limits.exhausted_result is None
            ),
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
    _requires_turn_scope = True

    def __init__(self, event: BaseEvent, context: Context, limits: _Limits, counter: CallCounter) -> None:
        super().__init__(event, context)
        self._limits = limits
        self._counter = counter
        #: Why this turn is exhausted; ``None`` while it is not.
        self._exhausted: str | None = None
        #: Model calls left to write the answer once the turn is exhausted.
        self._final_calls = _FINAL_CALLS
        self._group: _CallGroup | None = None

    async def on_turn(self, call_next: AgentTurn, event: BaseEvent, context: Context) -> ModelResponse:
        previous = context.dependencies.get(_CallGroup)
        owner = previous is None or not previous.collecting or previous.stream_id != context.stream.id
        if owner:
            group = _CallGroup(context)
        else:
            assert previous is not None
            group = previous
        group.limiters.append(self)
        self._group = group
        if owner:
            context.dependencies[_CallGroup] = group
        try:
            response = await call_next(event, context)
            return await group.finish(response, context) if owner else response
        finally:
            if owner:
                group.collecting = False
                if previous is None:
                    context.dependencies.pop(_CallGroup, None)
                else:
                    context.dependencies[_CallGroup] = previous

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        group = self._call_group(context)
        if group.limiters[0] is not self:
            return await call_next(event, context)
        return await group.execute_tool(call_next, event, context)

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        group = self._call_group(context)
        if group.limiters[0] is not self:
            return await call_next(events, context)
        return await group.call_model(call_next, events, context)

    def _call_group(self, context: Context) -> "_CallGroup":
        if self._group is None:
            self._group = _CallGroup(context)
            self._group.limiters.append(self)
        self._group.collecting = False
        return self._group

    def _tool_total_hit(self) -> str | None:
        limits, counter = self._limits, self._counter
        if limits.max_seconds is not None and counter.elapsed() >= limits.max_seconds:
            return "max_seconds"
        if limits.max_tool_calls is not None and counter.tool_calls >= limits.max_tool_calls:
            return "max_tool_calls"
        return None

    def _llm_hit(self) -> str | None:
        limits, counter = self._limits, self._counter
        if reason := self._tool_total_hit():
            return reason
        if limits.max_llm_calls is not None and counter.llm_calls >= limits.max_llm_calls:
            return "max_llm_calls"
        return None

    def _synthesis_prompt(self) -> str:
        prompt = self._limits.synthesis_prompt
        if self._limits.exempt:
            # "Start no new work with tools" otherwise reads as forbidding the call
            # that records or hands off the result.
            prompt += f" These tools are not refused: {', '.join(sorted(self._limits.exempt))}."
        return prompt

    def _end(self, events: Sequence[BaseEvent]) -> ModelResponse:
        """End the turn without a model call."""
        assert self._exhausted is not None
        text = _last_text(events) if self._limits.ending == "last_text" else None
        ending = "last_text" if text else "empty"
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


class _CallGroup:
    """Coordinate the limiters of one turn before reserving or dispatching calls."""

    def __init__(self, context: Context) -> None:
        self.stream_id = context.stream.id
        self.limiters: list[_CallLimiter] = []
        self.collecting = True
        self.ending: CallLimitReached | None = None
        self.pending_notes: list[_RemainingNote] = []
        self.final_results: list[ToolResult] = []

    async def execute_tool(self, call_next: ToolExecution, event: ToolCallEvent, context: Context) -> ToolResultType:
        participating = [limiter for limiter in self.limiters if event.name not in limiter._limits.exempt]
        refusal: str | dict[str, Any] | None = None
        for limiter in participating:
            reason = limiter._exhausted or limiter._tool_total_hit()
            if reason is not None:
                limiter._exhausted = reason
                refusal = limiter._exhausted_result(event, reason)
                break
            cap = limiter._limits.max_tool_calls_per_tool.get(event.name)
            if cap is not None and limiter._counter.per_tool.get(event.name, 0) >= cap:
                refusal = limiter._tool_limit_result(event)
                break

        counters = _counters(participating)
        if refusal is not None:
            for counter in counters:
                counter.denied += 1
                counter.denied_per_tool[event.name] = counter.denied_per_tool.get(event.name, 0) + 1
            self._complete_final_tools()
            return ToolResultEvent.from_call(event, refusal)

        # Reserve only once every limiter has accepted the call. Reserving before
        # awaiting execution also bounds parallel siblings using a shared counter.
        for counter in counters:
            counter.tool_calls += 1
            counter.per_tool[event.name] = counter.per_tool.get(event.name, 0) + 1
        result = await call_next(event, context)
        if isinstance(result, ToolResultEvent) and result.result.final:
            self.final_results.append(result.result)
            self._complete_final_tools()
        return result

    def _complete_final_tools(self) -> None:
        if not self.final_results:
            return
        if self.ending is None:
            for limiter in self.limiters:
                if limiter._exhausted is not None:
                    self.ending = CallLimitReached(reason=limiter._exhausted, ending="final_tool")
                    break
        if self.ending is not None:
            # A final tool may settle before a parallel sibling is refused.
            # Update its result before the executor publishes the final reply.
            for result in self.final_results:
                result.metadata[CALL_LIMIT_KEY] = {"reason": self.ending.reason, "ending": self.ending.ending}

    async def call_model(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        # Storage may hand us its live list; snapshot it before publishing notes.
        prompt = list(events)
        # Retries reuse the caller's original snapshot. Carry input appended by
        # failed attempts even when storage does not expose a live history list.
        known = {event.note_id for event in prompt if isinstance(event, _RemainingNote)}
        prompt.extend(note for note in self.pending_notes if note.note_id not in known)
        retrying: _CallLimiter | None = None
        while True:
            exhausted = self._first_exhausted()
            if exhausted is None:
                for counter in _counters(self.limiters):
                    counter.llm_calls += 1
                for limiter in self.limiters:
                    if limiter._limits.show_remaining and (text := limiter._remaining_note()) is not None:
                        note = _RemainingNote([TextInput(text)])
                        # Persist this append-only input without scheduling a second
                        # model call. Replaying it preserves the next cached prefix.
                        await context.send(note)
                        prompt.append(note)
                        self.pending_notes.append(note)
                response = await call_next(prompt, context)
                self.pending_notes.clear()
                return response

            index, limiter = exhausted
            if limiter._limits.on_exhausted == "stop" or not limiter._final_calls:
                limiter._counter.llm_denied += 1
                return self._end(limiter, prompt)

            limiter._final_calls -= 1
            for counter in _counters(self.limiters[: index + 1]):
                counter.llm_calls += 1
            final_prompt = [*prompt, _LimitPrompt.ensure_request([limiter._synthesis_prompt()])]
            if retrying is limiter:
                final_prompt.append(_LimitPrompt.ensure_request([TEXT_ONLY_PROMPT]))
            try:
                response = await call_next(final_prompt, context)
            except Exception as error:
                return self._end(limiter, prompt, error)

            calls = response.tool_calls.calls
            kept = [
                call for call in calls if isinstance(call, BuiltinToolCallEvent) or call.name in limiter._limits.exempt
            ]
            text = response.message.content.strip() if response.message else ""
            if kept or text or not calls:
                if len(kept) != len(calls):
                    logging.getLogger(__name__).warning(
                        "CallLimiter dropped tool calls %s from the final reply (%s)",
                        [call.name for call in calls if all(call is not other for other in kept)],
                        limiter._exhausted,
                    )
                    response.tool_calls = ToolCallsEvent(kept)
                assert limiter._exhausted is not None
                self.ending = CallLimitReached(reason=limiter._exhausted, ending="synthesis")
                if response.message is not None:
                    response.message.metadata[CALL_LIMIT_KEY] = {"reason": limiter._exhausted, "ending": "synthesis"}
                self.pending_notes.clear()
                return response
            # The agent emits usage only for the response returned by this hook.
            # Discarding a tool-only answer must not discard its billable work.
            if response.usage:
                await context.send(
                    UsageEvent(
                        response.usage,
                        model=response.model,
                        provider=response.provider,
                        finish_reason=response.finish_reason,
                    )
                )
            # Recheck outer limits before each retry, including its own final-call
            # allowance. An inner synthesis cannot bypass a run-wide model cap.
            retrying = limiter

    def _first_exhausted(self) -> tuple[int, _CallLimiter] | None:
        for index, limiter in enumerate(self.limiters):
            if limiter._exhausted is None:
                limiter._exhausted = limiter._llm_hit()
            if limiter._exhausted is not None:
                return index, limiter
        return None

    def _end(self, limiter: _CallLimiter, events: Sequence[BaseEvent], error: Exception | None = None) -> ModelResponse:
        response = limiter._end(events)
        assert limiter._exhausted is not None
        self.ending = CallLimitReached(
            reason=limiter._exhausted,
            ending="last_text" if response.message is not None else "empty",
            error_type=type(error).__name__ if error is not None else None,
            error=str(error) if error is not None else None,
        )
        return response

    async def finish(self, response: ModelResponse, context: Context) -> ModelResponse:
        if self.ending is not None:
            if response.message is not None:
                response.message.metadata[CALL_LIMIT_KEY] = {"reason": self.ending.reason, "ending": self.ending.ending}
            await context.send(self.ending)
        return response


def _counters(limiters: Sequence[_CallLimiter]) -> list[CallCounter]:
    """One reservation per counter, even when several policies share it."""
    counters: list[CallCounter] = []
    seen: set[int] = set()
    for limiter in limiters:
        if id(limiter._counter) not in seen:
            counters.append(limiter._counter)
            seen.add(id(limiter._counter))
    return counters


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
        if isinstance(event, ModelRequest) and not isinstance(event, DrainedModelRequest):
            return None
        if isinstance(event, ModelResponse) and event.message and event.message.content.strip():
            return event.message.content
    return None
