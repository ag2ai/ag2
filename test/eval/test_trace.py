# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Public-API tests for :class:`Trace` and :class:`TokenUsage`."""

from ag2.eval import Trace
from ag2.eval.trace import TokenUsage
from ag2.events import (
    BaseEvent,
    ModelMessage,
    ModelResponse,
    ToolCallEvent,
    Usage,
)


def _trace(
    *events: BaseEvent,
    exception: BaseException | None = None,
    duration_ms: int = 0,
) -> Trace:
    return Trace(
        events=list(events),
        exception=exception,
        duration_ms=duration_ms,
    )


class TestEventsOf:
    def test_filters_by_type(self) -> None:
        call = ToolCallEvent(name="get_weather", arguments="{}")
        trace = _trace(call, ModelMessage("hi"))

        assert trace.events_of(ToolCallEvent) == (call,)

    def test_filters_by_name(self) -> None:
        call_a = ToolCallEvent(name="get_weather", arguments="{}")
        call_b = ToolCallEvent(name="get_news", arguments="{}")
        trace = _trace(call_a, call_b)

        assert trace.events_of(ToolCallEvent, name="get_weather") == (call_a,)

    def test_preserves_event_order(self) -> None:
        first = ToolCallEvent(name="x", arguments="{}")
        second = ToolCallEvent(name="x", arguments="{}")
        trace = _trace(first, ModelMessage("between"), second)

        assert trace.events_of(ToolCallEvent) == (first, second)

    def test_returns_empty_tuple_when_none_match(self) -> None:
        trace = _trace(ModelMessage("just text"))

        assert trace.events_of(ToolCallEvent) == ()

    def test_unknown_name_returns_empty(self) -> None:
        trace = _trace(ToolCallEvent(name="get_weather", arguments="{}"))

        assert trace.events_of(ToolCallEvent, name="get_news") == ()


class TestTokens:
    def test_sums_across_model_responses(self) -> None:
        first = ModelResponse(usage=Usage(prompt_tokens=10, completion_tokens=20))
        second = ModelResponse(usage=Usage(prompt_tokens=5, completion_tokens=8))
        trace = _trace(first, second)

        assert trace.tokens == TokenUsage(input=15, output=28)

    def test_includes_cache_token_counts(self) -> None:
        response = ModelResponse(
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=5,
                cache_creation_input_tokens=3,
                cache_read_input_tokens=7,
            )
        )
        trace = _trace(response)

        assert trace.tokens == TokenUsage(input=10, output=5, cache_creation=3, cache_read=7)

    def test_zero_when_no_model_responses(self) -> None:
        trace = _trace(ToolCallEvent(name="x", arguments="{}"))

        assert trace.tokens == TokenUsage()

    def test_handles_missing_usage_fields(self) -> None:
        response = ModelResponse(usage=Usage())
        trace = _trace(response)

        assert trace.tokens == TokenUsage()


class TestTokenUsage:
    def test_total_is_input_plus_output(self) -> None:
        usage = TokenUsage(input=10, output=20, cache_creation=5, cache_read=2)

        assert usage.total == 30

    def test_default_is_all_zero(self) -> None:
        assert TokenUsage() == TokenUsage(input=0, output=0, cache_creation=0, cache_read=0)


class TestProperties:
    def test_events_is_a_tuple(self) -> None:
        trace = _trace(ModelMessage("hi"))

        assert isinstance(trace.events, tuple)

    def test_duration_ms_round_trips(self) -> None:
        trace = _trace(duration_ms=1234)

        assert trace.duration_ms == 1234

    def test_exception_defaults_to_none(self) -> None:
        trace = _trace()

        assert trace.exception is None

    def test_exception_captured_when_set(self) -> None:
        err = RuntimeError("boom")
        trace = _trace(exception=err)

        assert trace.exception is err
