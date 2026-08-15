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
    UsageEvent,
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
    def test_sums_across_model_calls(self) -> None:
        first = UsageEvent(Usage(prompt_tokens=10, completion_tokens=20))
        second = UsageEvent(Usage(prompt_tokens=5, completion_tokens=8))
        trace = _trace(first, second)

        assert trace.tokens == TokenUsage(input=15, output=28)

    def test_includes_cache_token_counts(self) -> None:
        event = UsageEvent(
            Usage(
                prompt_tokens=10,
                completion_tokens=5,
                cache_creation_input_tokens=3,
                cache_read_input_tokens=7,
            )
        )
        trace = _trace(event)

        assert trace.tokens == TokenUsage(input=10, output=5, cache_creation=3, cache_read=7)

    def test_counts_delegated_spend(self) -> None:
        """A delegating agent does most of its spending in workers.

        Only the ``"subtask"`` rollup reaches the parent, and it is not a
        ``ModelResponse`` — so this was the largest share of what eval missed.
        """
        trace = _trace(
            UsageEvent(Usage(prompt_tokens=100, completion_tokens=10)),
            UsageEvent(Usage(prompt_tokens=900, completion_tokens=90), kind="subtask", label="worker"),
        )

        assert trace.tokens == TokenUsage(input=1000, output=100)

    def test_a_failed_run_still_reports_what_it_spent(self) -> None:
        """Tokens spent before the failure were still billed.

        The accounting event is emitted as the tokens are spent, so it is on the
        trace whether or not the run went on to produce an answer. Reporting a
        crashed task as free would understate exactly the runs worth
        investigating.
        """
        trace = _trace(
            UsageEvent(Usage(prompt_tokens=100, completion_tokens=10)),
            UsageEvent(Usage(prompt_tokens=900, completion_tokens=90), kind="subtask", label="worker"),
            exception=RuntimeError("tool blew up"),
        )

        assert trace.tokens == TokenUsage(input=1000, output=100)

    def test_counts_maintenance_work(self) -> None:
        """Compaction and memory aggregation make real, billable calls."""
        trace = _trace(
            UsageEvent(Usage(prompt_tokens=40, completion_tokens=4), kind="compaction"),
            UsageEvent(Usage(prompt_tokens=20, completion_tokens=2), kind="aggregation"),
        )

        assert trace.tokens == TokenUsage(input=60, output=6)

    def test_counts_a_model_call_once_when_the_response_accompanies_it(self) -> None:
        """The main loop emits both; reading both sources would double every call."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)
        trace = _trace(UsageEvent(usage), ModelResponse(usage=usage))

        assert trace.tokens == TokenUsage(input=10, output=20)

    def test_model_response_alone_reports_zero(self) -> None:
        """Documents the deliberate change of source.

        ``UsageEvent`` is the framework's accounting record; a trace carrying
        only a response event carries no accounting.
        """
        trace = _trace(ModelResponse(usage=Usage(prompt_tokens=10, completion_tokens=20)))

        assert trace.tokens == TokenUsage()

    def test_zero_when_no_model_calls(self) -> None:
        trace = _trace(ToolCallEvent(name="x", arguments="{}"))

        assert trace.tokens == TokenUsage()

    def test_empty_trace_reports_zero(self) -> None:
        assert _trace().tokens == TokenUsage()

    def test_handles_missing_usage_fields(self) -> None:
        trace = _trace(UsageEvent(Usage()))

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
