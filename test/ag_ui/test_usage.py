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

from .utils import (
    collect_events,
    create_run_input,
    exploding_agent,
    frames_of_failing_run,
    leaf_exceptions,
)

pytestmark = pytest.mark.asyncio


async def _frames(agent: Agent) -> list[dict[str, Any]]:
    """The SSE frames one run yields, decoded but not parsed."""
    return await collect_events(AGUIStream(agent), create_run_input(UserMessage(id="msg_1", content="go")))


async def _finished(agent: Agent) -> RunFinishedEvent:
    """The terminating event of a completed run.

    Taken as the last frame and parsed by the class the implementation sends. The class
    rejects a frame of any other type, so this also pins that the run really did end on
    ``RUN_FINISHED`` — no separate search by type is needed.
    """
    return RunFinishedEvent.model_validate((await _frames(agent))[-1])


async def _run_error(agent: Agent) -> RunErrorEvent:
    """The terminating event of a failing run, emitted before ``dispatch`` re-raises."""
    run_input = create_run_input(UserMessage(id="msg_1", content="go"))
    return RunErrorEvent.model_validate((await frames_of_failing_run(agent, run_input))[-1])


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

    async def test_a_total_a_call_did_not_report_is_absent_not_partial(self) -> None:
        """An unreported total is unknown, not zero, so the pair reports none at all.

        Summing would put a figure on the wire smaller than the input and output beside
        it — 140 in, 14 out, 110 altogether — which is not a total of anything. It is left
        absent rather than derived from input and output, and the additive counts, whose
        absence really does mean zero, are still reported in full.
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
            TokenUsage(provider="openai", model="gpt-5", input_tokens=140, output_tokens=14)
        ]

    async def test_additive_counts_are_summed_across_calls_in_one_pair(self) -> None:
        """The deliberate asymmetry with the total above.

        A provider omits ``thinking_tokens`` on a call that did no reasoning, so within
        one provider/model pair an absent additive count means zero and summing it is the
        measurement. Only the total gets the all-or-nothing treatment.
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
        place of the run's own cause.
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
            TokenUsage(provider="anthropic", model="claude-sonnet-4", output_tokens=4, total_tokens=14)
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
        """Absence, never a zero or a derived figure standing in for an unmeasured value.

        The whole-object comparison is what pins it: a total derived from 10 + 4 would
        show up here as ``total_tokens=14``.
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
            TokenUsage(provider="anthropic", model="claude-sonnet-4", input_tokens=10, output_tokens=4)
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
        }

    async def test_cache_write_tokens_appear_nowhere(self) -> None:
        """Cache-write is not cached-input, and is not folded into input either."""
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
                input_tokens=10,
                output_tokens=4,
                cached_input_tokens=8,
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

    async def test_a_delegation_omits_a_total_no_call_of_it_fully_reported(self) -> None:
        """A rollup does not put a total on the wire that its calls did not measure."""
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
                total_tokens=None,
            ),
        ]

    async def test_a_mixed_pair_delegation_reports_unlabelled_rather_than_mislabelled(self) -> None:
        """A sub-agent that spanned two configurations has no honest label.

        Naming either one — or the parent's — would show an attribution that is not true
        and that a client cannot detect. Absence says "real spend, unattributable", which
        it can render. The tokens are still counted in full.
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
            TokenUsage(provider=None, model=None, input_tokens=200, output_tokens=60, total_tokens=260),
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
        """Covered for the bare failure path in ``test_run_error.py``; pinned again here
        with usage on the event, since that is the mapping that could raise in its place."""
        agent = exploding_agent(Usage(prompt_tokens=250, completion_tokens=50, total_tokens=300))
        run_input = create_run_input(UserMessage(id="msg_1", content="go"))

        with pytest.raises(Exception) as exc_info:
            await collect_events(AGUIStream(agent), run_input)

        assert [type(e) for e in leaf_exceptions(exc_info.value)] == [RuntimeError]
