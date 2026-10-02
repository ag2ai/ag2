# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Token usage as an AG-UI client sees it on the run-terminating events.

Every assertion is made on the events a client actually receives from ``dispatch``,
parsed back into the protocol's own models — nothing reaches for the mapping helper.
A run spanning several provider/model pairs is constructible because the test client
returns each supplied ``ModelResponse`` verbatim, and the agent emits that response's
own ``model``, ``provider`` and ``usage`` onto the stream.
"""

from typing import Any

import pytest
from ag_ui.core import RunErrorEvent, RunFinishedEvent, TokenUsage, UserMessage

from ag2 import Agent, KnowledgeConfig
from ag2.ag_ui import AGUIStream
from ag2.aggregate import AggregateTrigger, ConversationSummaryAggregate
from ag2.compact import CompactTrigger, SummarizeCompact
from ag2.events import ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent, Usage
from ag2.knowledge import MemoryKnowledgeStore
from ag2.testing import TestConfig
from ag2.tools import tool
from test._helpers import lookup
from test.ag_ui.harness import dispatch_events, dispatch_run, events_of_failing_run, exploding_agent, run_input

pytestmark = pytest.mark.asyncio


async def _frames(agent: Agent) -> list[dict[str, Any]]:
    """The SSE frames one run yields, decoded but not parsed."""
    return await dispatch_run(AGUIStream(agent), run_input(UserMessage(id="msg_1", content="go")))


async def _finished(agent: Agent) -> RunFinishedEvent:
    """The terminating event of a completed run.

    The run must end on it: a run that ended any other way fails here, not in the assertion.
    """
    *_, last = await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="msg_1", content="go")))
    assert isinstance(last, RunFinishedEvent)
    return last


async def _run_error(agent: Agent) -> RunErrorEvent:
    """The terminating event of a failing run."""
    *_, last = await events_of_failing_run(agent, run_input(UserMessage(id="msg_1", content="go")))
    assert isinstance(last, RunErrorEvent)
    return last


class TestCompletedRun:
    async def test_reports_input_output_and_total(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    ModelMessage("done"),
                    usage=Usage(prompt_tokens=120, completion_tokens=48, total_tokens=168),
                    model="claude-sonnet-4",
                    provider="anthropic",
                ),
            ),
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(
                provider="anthropic",
                model="claude-sonnet-4",
                input_tokens=120,
                output_tokens=48,
                total_tokens=168,
            )
        ]

    async def test_sums_every_model_call_not_just_the_last(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(calls=[ToolCallEvent(name="lookup", arguments="{}")]),
                    usage=Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
                    model="gpt-5",
                    provider="openai",
                ),
                ModelResponse(
                    ModelMessage("it is 42"),
                    usage=Usage(prompt_tokens=40, completion_tokens=4, total_tokens=44),
                    model="gpt-5",
                    provider="openai",
                ),
            ),
            tools=[lookup],
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(provider="openai", model="gpt-5", input_tokens=140, output_tokens=14, total_tokens=154)
        ]

    async def test_the_total_is_input_plus_output_whatever_the_provider_reported(self) -> None:
        """AG-UI's total is the sum of its two totals, so it is computed rather than copied.

        Copying would put a figure on the wire smaller than the input and output beside
        it — 140 in, 14 out, 110 altogether — which is not a total of anything.
        """

        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(calls=[ToolCallEvent(name="lookup", arguments="{}")]),
                    usage=Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
                    model="gpt-5",
                    provider="openai",
                ),
                ModelResponse(
                    ModelMessage("it is 42"),
                    usage=Usage(prompt_tokens=40, completion_tokens=4),
                    model="gpt-5",
                    provider="openai",
                ),
            ),
            tools=[lookup],
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(provider="openai", model="gpt-5", input_tokens=140, output_tokens=14, total_tokens=154)
        ]

    async def test_the_total_of_a_pair_matches_its_grouped_input_and_output_when_a_call_reports_no_output(self) -> None:
        """A call with no output count has no total of its own, but the pair's still adds up."""
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(calls=[ToolCallEvent(name="lookup", arguments="{}")]),
                    usage=Usage(prompt_tokens=10, completion_tokens=5),
                    model="m",
                    provider="openai",
                ),
                ModelResponse(ModelMessage("it is 42"), usage=Usage(prompt_tokens=7), model="m", provider="openai"),
            ),
            tools=[lookup],
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(provider="openai", model="m", input_tokens=17, output_tokens=5, total_tokens=22)
        ]

    async def test_additive_counts_are_summed_across_calls_in_one_pair(self) -> None:
        """A provider omits ``thinking_tokens`` on a call that did no reasoning, so within
        one provider/model pair an absent additive count means zero and summing it is the
        measurement.
        """

        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(calls=[ToolCallEvent(name="lookup", arguments="{}")]),
                    usage=Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110, thinking_tokens=64),
                    model="gpt-5",
                    provider="openai",
                ),
                ModelResponse(
                    ModelMessage("it is 42"),
                    usage=Usage(prompt_tokens=40, completion_tokens=4, total_tokens=44),
                    model="gpt-5",
                    provider="openai",
                ),
            ),
            tools=[lookup],
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(
                provider="openai",
                model="gpt-5",
                input_tokens=140,
                output_tokens=14,
                total_tokens=154,
                reasoning_tokens=64,
            )
        ]

    @pytest.mark.parametrize("rejected", [float("nan"), float("inf"), -5.0])
    async def test_omits_a_count_the_wire_type_would_reject(self, rejected: float) -> None:
        """``Usage`` counts are floats, so a provider mapper can hand over a value the
        protocol's non-negative integer field would refuse. It is omitted rather than
        sent, which is also what keeps the mapping from raising on the failure path in
        place of the run's own cause — and with it the total it would have been part of.
        """
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    ModelMessage("done"),
                    usage=Usage(prompt_tokens=rejected, completion_tokens=4, total_tokens=14),
                    model="claude-sonnet-4",
                    provider="anthropic",
                ),
            ),
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(provider="anthropic", model="claude-sonnet-4", output_tokens=4)
        ]

    async def test_reported_spend_agrees_with_the_run_s_own_usage_report(self) -> None:
        """The figure a client sees is the figure ``AgentReply.usage()`` reports.

        Both read the same event log, so this pins that the mapping neither drops a record
        nor counts one twice — including the delegated sub-agent's rollup, which is the
        record the per-model and per-provider maps would have lost.
        """

        def build() -> Agent:
            worker = Agent(
                "worker",
                config=TestConfig(
                    ModelResponse(
                        ModelMessage("researched"),
                        usage=Usage(prompt_tokens=200, completion_tokens=60, total_tokens=260),
                        model="claude-haiku-4",
                        provider="anthropic",
                    ),
                ),
            )
            return Agent(
                "test_agent",
                config=TestConfig(
                    ModelResponse(
                        tool_calls=ToolCallsEvent(
                            calls=[ToolCallEvent(name="task_worker", arguments='{"objective": "go"}')]
                        ),
                        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                        model="gpt-5",
                        provider="openai",
                    ),
                    ModelResponse(
                        ModelMessage("summarised"),
                        usage=Usage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
                        model="gpt-5",
                        provider="openai",
                    ),
                ),
                tools=[worker.as_tool(description="Delegate research to the worker.")],
            )

        entries = (await _finished(build())).usage
        report = await (await build().ask("go")).usage()

        assert entries is not None
        assert sum(entry.input_tokens or 0 for entry in entries) == report.total.prompt_tokens
        assert sum(entry.output_tokens or 0 for entry in entries) == report.total.completion_tokens
        assert sum(entry.total_tokens or 0 for entry in entries) == report.total.total_tokens

    async def test_one_entry_per_pair_in_order_of_first_appearance(self) -> None:
        @tool
        def handoff() -> str:
            """Hand off to the other model."""
            return "ok"

        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(calls=[ToolCallEvent(name="handoff", arguments="{}")]),
                    usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
                    model="claude-sonnet-4",
                    provider="anthropic",
                ),
                ModelResponse(
                    ModelMessage("done"),
                    usage=Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
                    model="gpt-5",
                    provider="openai",
                ),
            ),
            tools=[handoff],
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(
                provider="anthropic", model="claude-sonnet-4", input_tokens=10, output_tokens=2, total_tokens=12
            ),
            TokenUsage(provider="openai", model="gpt-5", input_tokens=7, output_tokens=3, total_tokens=10),
        ]

    async def test_reports_reasoning_and_cached_input_when_supplied(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    ModelMessage("done"),
                    usage=Usage(
                        prompt_tokens=100,
                        completion_tokens=30,
                        total_tokens=130,
                        thinking_tokens=18,
                        cache_read_input_tokens=64,
                    ),
                    model="gpt-5",
                    provider="openai",
                ),
            ),
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(
                provider="openai",
                model="gpt-5",
                input_tokens=100,
                output_tokens=30,
                total_tokens=130,
                reasoning_tokens=18,
                cached_input_tokens=64,
            )
        ]

    async def test_omits_fields_the_provider_did_not_report(self) -> None:
        """Absence, never a zero standing in for an unmeasured value.

        The total is the one figure derived, because the protocol defines it as the
        sum of the two beside it.
        """
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    ModelMessage("done"),
                    usage=Usage(prompt_tokens=10, completion_tokens=4),
                    model="claude-sonnet-4",
                    provider="anthropic",
                ),
            ),
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(provider="anthropic", model="claude-sonnet-4", input_tokens=10, output_tokens=4, total_tokens=14)
        ]

    async def test_unreported_fields_are_absent_on_the_wire_not_null(self) -> None:
        """The one claim the parsed model cannot carry.

        Parsing puts an omitted field back as ``None``, so absence and an explicit
        ``null`` are indistinguishable once decoded — this asserts on the raw frame.
        """
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    ModelMessage("done"),
                    usage=Usage(prompt_tokens=10, completion_tokens=4),
                    model="claude-sonnet-4",
                    provider="anthropic",
                ),
            ),
        )
        frames = await _frames(agent)

        [entry] = frames[-1]["usage"]
        assert entry == {
            "provider": "anthropic",
            "model": "claude-sonnet-4",
            "inputTokens": 10,
            "outputTokens": 4,
            "totalTokens": 14,
        }

    async def test_cache_writes_are_reported_apart_from_cache_reads(self) -> None:
        """Priced differently, so reported in a field of their own — and inside the input."""
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    ModelMessage("done"),
                    usage=Usage(
                        prompt_tokens=10,
                        completion_tokens=4,
                        cache_creation_input_tokens=512,
                        cache_read_input_tokens=8,
                    ),
                    model="claude-sonnet-4",
                    provider="anthropic",
                ),
            ),
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(
                provider="anthropic",
                model="claude-sonnet-4",
                input_tokens=530,
                output_tokens=4,
                total_tokens=534,
                cached_input_tokens=8,
                cache_write_input_tokens=512,
            )
        ]

    async def test_a_run_that_spent_nothing_omits_usage(self) -> None:
        agent = Agent("test_agent", config=TestConfig("hello"))

        assert (await _finished(agent)).usage is None

    async def test_a_single_pair_delegation_keeps_its_labels(self) -> None:
        """A sub-agent that used one configuration is attributed to it.

        The spend arrives as one rollup, still — that invariant is unchanged — but the
        rollup now carries the pair behind it, so per-model attribution survives
        delegation instead of collapsing into an unlabelled row.
        """
        worker = Agent(
            "worker",
            config=TestConfig(
                ModelResponse(
                    ModelMessage("researched"),
                    usage=Usage(prompt_tokens=200, completion_tokens=60, total_tokens=260),
                    model="claude-haiku-4",
                    provider="anthropic",
                ),
            ),
        )
        parent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(
                        calls=[ToolCallEvent(name="task_worker", arguments='{"objective": "go"}')]
                    ),
                    usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    model="gpt-5",
                    provider="openai",
                ),
                ModelResponse(
                    ModelMessage("summarised"),
                    usage=Usage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
                    model="gpt-5",
                    provider="openai",
                ),
            ),
            tools=[worker.as_tool(description="Delegate research to the worker.")],
        )

        assert (await _finished(parent)).usage == [
            TokenUsage(provider="openai", model="gpt-5", input_tokens=30, output_tokens=13, total_tokens=43),
            TokenUsage(
                provider="anthropic",
                model="claude-haiku-4",
                input_tokens=200,
                output_tokens=60,
                total_tokens=260,
            ),
        ]

    async def test_a_delegation_s_total_is_its_input_plus_output(self) -> None:
        """A rollup's total is computed like any other, whatever its calls reported."""
        worker = Agent(
            "worker",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(calls=[ToolCallEvent(name="lookup", arguments="{}")]),
                    usage=Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
                    model="claude-haiku-4",
                    provider="anthropic",
                ),
                ModelResponse(
                    ModelMessage("researched"),
                    usage=Usage(prompt_tokens=40, completion_tokens=4),
                    model="claude-haiku-4",
                    provider="anthropic",
                ),
            ),
            tools=[lookup],
        )
        parent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(
                        calls=[ToolCallEvent(name="task_worker", arguments='{"objective": "go"}')]
                    ),
                    usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    model="gpt-5",
                    provider="openai",
                ),
                ModelResponse(
                    ModelMessage("summarised"),
                    usage=Usage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
                    model="gpt-5",
                    provider="openai",
                ),
            ),
            tools=[worker.as_tool(description="Delegate research to the worker.")],
        )

        assert (await _finished(parent)).usage == [
            TokenUsage(provider="openai", model="gpt-5", input_tokens=30, output_tokens=13, total_tokens=43),
            TokenUsage(
                provider="anthropic",
                model="claude-haiku-4",
                input_tokens=140,
                output_tokens=14,
                total_tokens=154,
            ),
        ]

    async def test_a_mixed_pair_delegation_reports_each_call_under_its_own_pair(self) -> None:
        """A sub-agent that spanned two configurations is reported per configuration.

        Its rollup carries no single honest label, so each call it sums is reported under
        the pair that served it — each counted in its own provider's accounting.
        """

        worker = Agent(
            "worker",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(calls=[ToolCallEvent(name="lookup", arguments="{}")]),
                    usage=Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                    model="claude-haiku-4",
                    provider="anthropic",
                ),
                ModelResponse(
                    ModelMessage("researched"),
                    usage=Usage(prompt_tokens=100, completion_tokens=40, total_tokens=140),
                    model="gpt-5-mini",
                    provider="openai",
                ),
            ),
            tools=[lookup],
        )
        parent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(
                        calls=[ToolCallEvent(name="task_worker", arguments='{"objective": "go"}')]
                    ),
                    usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    model="gpt-5",
                    provider="openai",
                ),
                ModelResponse(
                    ModelMessage("summarised"),
                    usage=Usage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
                    model="gpt-5",
                    provider="openai",
                ),
            ),
            tools=[worker.as_tool(description="Delegate research to the worker.")],
        )

        assert (await _finished(parent)).usage == [
            TokenUsage(provider="openai", model="gpt-5", input_tokens=30, output_tokens=13, total_tokens=43),
            TokenUsage(
                provider="anthropic", model="claude-haiku-4", input_tokens=100, output_tokens=20, total_tokens=120
            ),
            TokenUsage(provider="openai", model="gpt-5-mini", input_tokens=100, output_tokens=40, total_tokens=140),
        ]


def _answering(usage: Usage, *, provider: str, model: str = "m") -> Agent:
    return Agent(
        "test_agent",
        config=TestConfig(ModelResponse(ModelMessage("done"), usage=usage, model=model, provider=provider)),
    )


class TestProtocolAccounting:
    """AG-UI's input and output are totals, and its cache and reasoning counts parts of them.

    Corrected per provider where usage leaves for the wire; ``Usage`` itself keeps the
    provider's own figures.
    """

    async def test_a_missing_cache_count_adds_nothing(self) -> None:
        agent = _answering(
            Usage(prompt_tokens=10, completion_tokens=2, cache_read_input_tokens=4), provider="anthropic"
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(
                provider="anthropic",
                model="m",
                input_tokens=14,
                output_tokens=2,
                total_tokens=16,
                cached_input_tokens=4,
            )
        ]

    async def test_a_missing_prompt_count_leaves_input_absent_whatever_was_cached(self) -> None:
        """Cache counts alone are not the input: they are only part of it."""
        agent = _answering(Usage(completion_tokens=2, cache_read_input_tokens=4), provider="anthropic")

        assert (await _finished(agent)).usage == [
            TokenUsage(provider="anthropic", model="m", output_tokens=2, cached_input_tokens=4)
        ]

    async def test_a_missing_completion_count_leaves_output_absent_whatever_was_reasoned(self) -> None:
        agent = _answering(Usage(prompt_tokens=5, thinking_tokens=30), provider="google")

        assert (await _finished(agent)).usage == [
            TokenUsage(provider="google", model="m", input_tokens=5, reasoning_tokens=30)
        ]

    async def test_bedrock_cache_is_added_to_its_input(self) -> None:
        """Converse's `inputTokens` leaves out what was read from and written to the cache."""
        usage = Usage(
            prompt_tokens=100, completion_tokens=30, cache_read_input_tokens=64, cache_creation_input_tokens=8
        )

        assert (await _finished(_answering(usage, provider="bedrock"))).usage == [
            TokenUsage(
                provider="bedrock",
                model="m",
                input_tokens=172,
                output_tokens=30,
                total_tokens=202,
                cached_input_tokens=64,
                cache_write_input_tokens=8,
            )
        ]

    @pytest.mark.parametrize("provider", ["openai", "xai"])
    async def test_other_providers_are_reported_as_they_count(self, provider: str) -> None:
        """OpenAI counts cache and reasoning inside its totals already. xAI's reasoning
        is not known to sit outside them, so it is not added in.
        """
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=30,
            thinking_tokens=18,
            cache_read_input_tokens=64,
            cache_creation_input_tokens=8,
        )

        assert (await _finished(_answering(usage, provider=provider))).usage == [
            TokenUsage(
                provider=provider,
                model="m",
                input_tokens=100,
                output_tokens=30,
                total_tokens=130,
                reasoning_tokens=18,
                cached_input_tokens=64,
                cache_write_input_tokens=8,
            )
        ]

    async def test_ag2_s_own_usage_figures_stay_as_the_provider_reported_them(self) -> None:
        agent = _answering(
            Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12, cache_read_input_tokens=100),
            provider="anthropic",
        )

        reply = await agent.ask("go")

        assert (await reply.usage()).total == Usage(
            prompt_tokens=10, completion_tokens=2, total_tokens=12, cache_read_input_tokens=100
        )

    async def test_a_delegation_across_two_models_is_corrected_call_by_call(self) -> None:
        """Its rollup names neither model, so it is counted from the calls it sums."""
        cached = Usage(prompt_tokens=10, completion_tokens=2, cache_read_input_tokens=100)
        worker = Agent(
            "worker",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(calls=[ToolCallEvent(name="lookup", arguments="{}")]),
                    usage=cached,
                    model="claude-haiku-4",
                    provider="anthropic",
                ),
                ModelResponse(ModelMessage("researched"), usage=cached, model="claude-sonnet-4", provider="anthropic"),
            ),
            tools=[lookup],
        )
        parent = Agent(
            "test_agent",
            config=TestConfig(ToolCallEvent(name="task_worker", arguments='{"objective": "go"}'), "summarised"),
            tools=[worker.as_tool(description="Delegate research to the worker.")],
        )

        usage = (await _finished(parent)).usage

        assert usage == [
            TokenUsage(
                provider="anthropic",
                model=model,
                input_tokens=110,
                output_tokens=2,
                total_tokens=112,
                cached_input_tokens=100,
            )
            for model in ("claude-haiku-4", "claude-sonnet-4")
        ]


class TestInternalMaintenanceSpend:
    """Compaction and memory aggregation cost tokens too, and a client must see them.

    Both run on the agent's own machinery rather than in the turn's model loop, and
    both would be invisible to a client that only saw ``"model_call"`` records — so
    each is asserted here at the seam, on the frame the client receives, and not
    only on ``UsageReport`` inside the process.
    """

    async def test_a_compacted_run_reports_what_the_summarization_cost(self) -> None:
        # Compaction rewrites history, which is where this transport reads spend
        # from, so this pins the whole path: the summarization call's record is
        # emitted while the rewrite is in flight and must still reach the wire.
        summarizer = TestConfig(
            ModelResponse(
                ModelMessage("summary"),
                usage=Usage(prompt_tokens=500, completion_tokens=25, total_tokens=525),
                model="stub-summarizer",
                provider="stub",
            )
        )
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(calls=[ToolCallEvent(name="lookup", arguments="{}")]),
                    usage=Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
                    model="stub-main",
                    provider="stub",
                ),
                ModelResponse(
                    ModelMessage("done"),
                    usage=Usage(prompt_tokens=40, completion_tokens=4, total_tokens=44),
                    model="stub-main",
                    provider="stub",
                ),
            ),
            tools=[lookup],
            knowledge=KnowledgeConfig(
                store=MemoryKnowledgeStore(),
                compact=SummarizeCompact(target=6, config=summarizer),
                compact_trigger=CompactTrigger(max_events=2),
                expose_tool=False,
                write_event_log=False,
            ),
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(provider="stub", model="stub-main", input_tokens=140, output_tokens=14, total_tokens=154),
            TokenUsage(provider="stub", model="stub-summarizer", input_tokens=500, output_tokens=25, total_tokens=525),
        ]

    async def test_an_aggregated_run_reports_what_the_aggregation_cost(self) -> None:
        # The aggregation call runs on a throwaway stream of its own, so its
        # record reaches history only because ``_emit_aggregation_usage`` sends
        # it onto the real one. That hop is what this asserts.
        aggregator = TestConfig(
            ModelResponse(
                ModelMessage("conversation summary"),
                usage=Usage(prompt_tokens=80, completion_tokens=12, total_tokens=92),
                model="stub-aggregator",
                provider="stub",
            )
        )
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelResponse(
                    ModelMessage("done"),
                    usage=Usage(prompt_tokens=30, completion_tokens=6, total_tokens=36),
                    model="stub-main",
                    provider="stub",
                ),
            ),
            knowledge=KnowledgeConfig(
                store=MemoryKnowledgeStore(),
                aggregate=ConversationSummaryAggregate(config=aggregator),
                aggregate_trigger=AggregateTrigger(on_end=True),
                expose_tool=False,
                write_event_log=False,
            ),
        )

        assert (await _finished(agent)).usage == [
            TokenUsage(provider="stub", model="stub-main", input_tokens=30, output_tokens=6, total_tokens=36),
            TokenUsage(provider="stub", model="stub-aggregator", input_tokens=80, output_tokens=12, total_tokens=92),
        ]


class TestFailedRun:
    async def test_reports_usage_spent_before_the_failure(self) -> None:
        agent = exploding_agent(Usage(prompt_tokens=250, completion_tokens=50, total_tokens=300))

        assert (await _run_error(agent)).usage == [
            TokenUsage(
                provider="anthropic",
                model="claude-sonnet-4",
                input_tokens=250,
                output_tokens=50,
                total_tokens=300,
            )
        ]

    async def test_a_failure_before_any_spend_omits_usage(self) -> None:
        assert (await _run_error(exploding_agent())).usage is None

    async def test_usage_reporting_does_not_replace_the_run_s_own_failure(self) -> None:
        """Pinned with usage on the event, since that is the mapping that could fail in its place."""
        agent = exploding_agent(Usage(prompt_tokens=250, completion_tokens=50, total_tokens=300))

        assert "downstream is down" in (await _run_error(agent)).message
