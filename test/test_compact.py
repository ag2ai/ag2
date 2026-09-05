# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for CompactStrategy, CompactTrigger, and built-in strategies."""

import pytest

from ag2 import Agent, Context
from ag2.agent import KnowledgeConfig
from ag2.compact import CompactTrigger, CompactionSummary, SummarizeCompact, TailWindowCompact
from ag2.events import (
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    CompactionCompleted,
    CompactionFailed,
    CompactionStarted,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResult,
    ToolResultEvent,
    ToolResultsEvent,
    Usage,
    UsageEvent,
)
from ag2.events.base import is_conversational
from ag2.knowledge import MemoryKnowledgeStore
from ag2.stream import MemoryStream
from ag2.testing import TestConfig, TrackingConfig
from test._helpers import DurableReasoning, lookup


class TestTailWindowCompact:
    @pytest.mark.asyncio
    async def test_no_op_below_target(self) -> None:
        compact = TailWindowCompact(target=10)
        events = [ModelRequest([TextInput(f"msg-{i}")]) for i in range(5)]
        ctx = Context(stream=MemoryStream())
        result = await compact.compact(events, ctx, None)
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_truncates_above_target(self) -> None:
        compact = TailWindowCompact(target=3)
        events = [ModelRequest([TextInput(f"msg-{i}")]) for i in range(10)]
        ctx = Context(stream=MemoryStream())
        result = await compact.compact(events, ctx, None)
        assert len(result) == 3
        assert result[0].parts[0].content == "msg-7"

    @pytest.mark.asyncio
    async def test_persists_dropped_to_store(self) -> None:
        store = MemoryKnowledgeStore()
        compact = TailWindowCompact(target=3)
        events = [ModelRequest([TextInput(f"msg-{i}")]) for i in range(10)]
        stream = MemoryStream()
        ctx = Context(stream=stream)
        result = await compact.compact(events, ctx, store)
        assert len(result) == 3

        # Check that dropped events were persisted
        entries = await store.list("/log/")
        dropped = [e for e in entries if "dropped" in e]
        assert len(dropped) == 1

    @pytest.mark.asyncio
    async def test_no_persist_without_store(self) -> None:
        compact = TailWindowCompact(target=3)
        events = [ModelRequest([TextInput(f"msg-{i}")]) for i in range(10)]
        ctx = Context(stream=MemoryStream())
        result = await compact.compact(events, ctx, None)
        assert len(result) == 3


def _cycle(cid: str, name: str = "t") -> tuple[ModelResponse, ToolResultsEvent]:
    """A (ModelResponse tool call, ToolResultsEvent) pair linked by id."""
    call = ToolCallEvent(id=cid, name=name, arguments="{}")
    mr = ModelResponse(tool_calls=ToolCallsEvent(calls=[call]))
    res = ToolResultsEvent(results=[ToolResultEvent(parent_id=cid, name=name, result=ToolResult("ok"))])
    return mr, res


@pytest.mark.asyncio
class TestTailWindowToolCycleBoundary:
    """A tool-call/result cycle must never be split across the compaction
    boundary — a retained orphan result would crash strict providers."""

    async def test_split_cycle_compacts_whole(self) -> None:
        mr, res = _cycle("c1")
        events = [
            ModelRequest([TextInput("u0")]),
            mr,
            res,
            ModelRequest([TextInput("u1")]),
            ModelResponse(ModelMessage("done")),
        ]
        # target=3 would cut between the call and its result
        result = await TailWindowCompact(target=3).compact(events, Context(stream=MemoryStream()), None)
        assert result == [events[3], events[4]]

    async def test_clean_cycle_boundary_kept(self) -> None:
        mr, res = _cycle("c1")
        events = [ModelRequest([TextInput("u0")]), mr, res]
        result = await TailWindowCompact(target=2).compact(events, Context(stream=MemoryStream()), None)
        assert result == [mr, res]

    async def test_split_second_of_chained_cycles(self) -> None:
        mr1, res1 = _cycle("c1")
        mr2, res2 = _cycle("c2")
        events = [mr1, res1, mr2, res2, ModelResponse(ModelMessage("end"))]
        # target=2 would cut between the second call and its result
        result = await TailWindowCompact(target=2).compact(events, Context(stream=MemoryStream()), None)
        assert result == [events[4]]

    async def test_split_cycle_persisted_whole(self) -> None:
        store = MemoryKnowledgeStore()
        mr, res = _cycle("c1")
        events = [
            ModelRequest([TextInput("u0")]),
            mr,
            res,
            ModelRequest([TextInput("u1")]),
            ModelResponse(ModelMessage("done")),
        ]
        result = await TailWindowCompact(target=3).compact(events, Context(stream=MemoryStream()), store)
        assert mr not in result and res not in result
        entries = await store.list("/log/")
        assert [e for e in entries if "dropped" in e]


@pytest.mark.asyncio
class TestTailWindowBuiltinToolBoundary:
    """A builtin (server-side) call satisfies its own result and needs a reasoning
    anchor. Reading only ``ModelResponse.tool_calls`` sees neither."""

    async def test_intact_builtin_cycle_does_not_block_compaction(self) -> None:
        # The retained window ends with a builtin result whose call is right there.
        # A boundary check that cannot see builtin calls calls this unsafe forever,
        # runs the cut off the end, and silently gives up on compacting at all.
        events = [
            ModelRequest([TextInput("u0")]),
            ModelRequest([TextInput("u1")]),
            DurableReasoning("plan"),
            BuiltinToolCallEvent(id="ws_1", name="web_search", arguments="{}"),
            BuiltinToolResultEvent(parent_id="ws_1", name="web_search", result=ToolResult("ok")),
        ]

        result = await TailWindowCompact(target=3).compact(events, Context(stream=MemoryStream()), None)

        assert result == events[2:]

    async def test_split_builtin_group_compacts_whole(self) -> None:
        events = [
            ModelRequest([TextInput("u0")]),
            DurableReasoning("plan"),
            BuiltinToolCallEvent(id="ws_1", name="web_search", arguments="{}"),
            BuiltinToolResultEvent(parent_id="ws_1", name="web_search", result=ToolResult("ok")),
            ModelRequest([TextInput("u1")]),
        ]

        result = await TailWindowCompact(target=3).compact(events, Context(stream=MemoryStream()), None)

        assert result == [events[-1]]


@pytest.mark.asyncio
class TestTelemetryNotConversational:
    """UsageEvent is persisted telemetry, not conversation — it must not consume
    the retention window, leak into the summary, or trigger compaction."""

    async def test_usage_events_do_not_consume_window(self) -> None:
        events: list = []
        for i in range(3):
            events.append(ModelRequest([TextInput(f"u{i}")]))
            events.append(UsageEvent(Usage(total_tokens=10)))
        result = await TailWindowCompact(target=2).compact(events, Context(stream=MemoryStream()), None)

        conv = [e for e in result if isinstance(e, ModelRequest)]
        assert [e.parts[0].content for e in conv] == ["u1", "u2"]
        # Retained telemetry rides along so UsageReport keeps the window's usage
        assert any(isinstance(e, UsageEvent) for e in result)

    async def test_telemetry_alone_is_no_op(self) -> None:
        events: list = [ModelRequest([TextInput("only")])]
        events += [UsageEvent(Usage(total_tokens=1)) for _ in range(10)]
        result = await TailWindowCompact(target=3).compact(events, Context(stream=MemoryStream()), None)
        assert result == events

    async def test_summarizer_prompt_excludes_telemetry(self) -> None:
        tracking = TrackingConfig(TestConfig(ModelResponse(ModelMessage("summary"))))
        events: list = [
            ModelRequest([TextInput("keep-this-text")]),
            UsageEvent(Usage(total_tokens=313373)),
            ModelResponse(ModelMessage("and-this-text")),
            ModelRequest([TextInput("recent")]),
        ]
        await SummarizeCompact(target=1, config=tracking).compact(events, Context(stream=MemoryStream()), None)

        prompt = tracking.mock.call_args.args[0].parts[0].content
        assert "keep-this-text" in prompt and "and-this-text" in prompt
        assert "313373" not in prompt

    async def test_usage_events_do_not_advance_trigger(self) -> None:
        # One turn = ModelRequest + ModelResponse (2 conversational) + UsageEvent.
        # max_events=2 must NOT fire: counting the UsageEvent would push it to 3.
        stream = MemoryStream()
        completions: list[CompactionCompleted] = []
        stream.where(CompactionCompleted).subscribe(lambda e: completions.append(e))

        agent = Agent(
            "compactor",
            config=TestConfig(ModelResponse(ModelMessage("a"), usage=Usage(total_tokens=5))),
            knowledge=KnowledgeConfig(
                store=MemoryKnowledgeStore(),
                compact=TailWindowCompact(target=2),
                compact_trigger=CompactTrigger(max_events=2),
            ),
        )
        await agent.ask("once", stream=stream)
        assert completions == []


@pytest.mark.asyncio
class TestTelemetrySurvivesCompaction:
    """Compaction rewrites history; the run's token records must outlive it.

    ``UsageEvent`` is the only thing ``UsageReport`` reads (`ADR 0014`), so a
    record the rewrite drops is spend the run can never report again — on
    ``AgentReply.usage()`` and on every transport built over it.
    """

    @staticmethod
    def _agent() -> Agent:
        """A tool loop of 110 then 44 tokens, with a 525-token summarization call.

        The tool call is what makes this a two-call turn, so the first call's
        record is already outside the window compaction retains by the time the
        turn ends. Stub usage throughout, so the arithmetic is exact.
        """
        summarizer = TestConfig(
            ModelResponse(
                ModelMessage("summary"),
                usage=Usage(prompt_tokens=500, completion_tokens=25, total_tokens=525),
                model="stub-summarizer",
                provider="stub",
            )
        )
        return Agent(
            "compactor",
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

    async def test_spend_from_before_the_retained_window_survives(self) -> None:
        stream = MemoryStream()
        emitted: list[UsageEvent] = []
        stream.where(UsageEvent).subscribe(lambda e: emitted.append(e))

        report = await (await self._agent().ask("go", stream=stream)).usage()

        # Every token the run emitted is still accounted for after the rewrite.
        assert sum(e.usage.total_tokens or 0 for e in emitted) == 679
        assert report.total.total_tokens == 679

    async def test_the_summarization_call_s_own_spend_survives_and_trails(self) -> None:
        # This record is not merely outside the retained window: it is emitted
        # *while* the strategy runs, so it is absent from the snapshot the
        # strategy returns and cannot be recovered from it. It ran last, so it
        # is recorded last — consumers grouping by first appearance stay stable.
        stream = MemoryStream()

        report = await (await self._agent().ask("go", stream=stream)).usage()

        assert [(r.kind, r.usage.total_tokens) for r in report.records] == [
            ("model_call", 110),
            ("model_call", 44),
            ("compaction", 525),
        ]

    async def test_carried_records_do_not_advance_the_compaction_trigger(self) -> None:
        # The records ride along in history but are not conversational (`ADR
        # 0010`), so they must not count toward max_events — otherwise every
        # compaction would bring the next one closer.
        stream = MemoryStream()
        completions: list[CompactionCompleted] = []
        stream.where(CompactionCompleted).subscribe(lambda e: completions.append(e))

        await self._agent().ask("go", stream=stream)

        history = list(await stream.history.get_events())
        assert len(completions) == 1
        assert len([e for e in history if isinstance(e, UsageEvent)]) == 3
        # The retained window is the summary plus target conversational events;
        # the three carried records are not among them.
        assert len([e for e in history if is_conversational(e)]) <= 7

    async def test_a_second_compaction_carries_the_first_s_records_once(self) -> None:
        # Each compaction rebuilds the kept records from the history it is
        # replacing, so records already carried through one compaction are
        # carried through the next — once each. Double counting here would
        # inflate every subsequent report on a long conversation, and dropping
        # would silently re-introduce the bug this class exists for.
        agent = self._agent()
        stream = MemoryStream()
        completions: list[CompactionCompleted] = []
        stream.where(CompactionCompleted).subscribe(lambda e: completions.append(e))

        await agent.ask("go", stream=stream)
        report = await (await agent.ask("again", stream=stream)).usage()

        assert len(completions) == 2
        # Two turns of 110 + 44, and one summarization call per compaction.
        assert [(r.kind, r.usage.total_tokens) for r in report.records] == [
            ("model_call", 110),
            ("model_call", 44),
            ("compaction", 525),
            ("model_call", 110),
            ("model_call", 44),
            ("compaction", 525),
        ]
        assert report.total.total_tokens == 1358


class TestCompactionSummary:
    def test_is_base_event(self) -> None:
        summary = CompactionSummary(summary="Earlier work...", event_count=50)
        assert summary.summary == "Earlier work..."
        assert summary.event_count == 50


class TestCompactTrigger:
    def test_defaults(self) -> None:
        trigger = CompactTrigger()
        assert trigger.max_events == 0
        assert trigger.max_tokens == 0
        assert trigger.chars_per_token == 4

    def test_custom_values(self) -> None:
        trigger = CompactTrigger(max_events=100, max_tokens=50000)
        assert trigger.max_events == 100
        assert trigger.max_tokens == 50000

    def test_custom_chars_per_token(self) -> None:
        trigger = CompactTrigger(max_tokens=100, chars_per_token=2)
        assert trigger.chars_per_token == 2


class TestCompactionWiredOnAgent:
    """End-to-end: an Agent configured with compaction emits CompactionCompleted
    on the stream and shrinks history once the trigger threshold is crossed."""

    @pytest.mark.asyncio
    async def test_fires_when_threshold_crossed(self) -> None:
        store = MemoryKnowledgeStore()
        stream = MemoryStream()
        completions: list[CompactionCompleted] = []
        stream.where(CompactionCompleted).subscribe(lambda e: completions.append(e))

        agent = Agent(
            "compactor",
            config=TestConfig(
                ModelResponse(ModelMessage("a")),
                ModelResponse(ModelMessage("b")),
                ModelResponse(ModelMessage("c")),
                ModelResponse(ModelMessage("d")),
            ),
            knowledge=KnowledgeConfig(
                store=store,
                compact=TailWindowCompact(target=2),
                compact_trigger=CompactTrigger(max_events=3),
            ),
        )

        # Four turns on the same stream — history grows past max_events=3
        reply = await agent.ask("turn-1", stream=stream)
        for q in ("turn-2", "turn-3", "turn-4"):
            reply = await reply.ask(q)

        assert len(completions) >= 1
        assert completions[0].agent == "compactor"
        assert completions[0].strategy == "TailWindowCompact"
        assert completions[0].events_after <= 2

    @pytest.mark.asyncio
    async def test_does_not_fire_below_threshold(self) -> None:
        store = MemoryKnowledgeStore()
        stream = MemoryStream()
        completions: list[CompactionCompleted] = []
        stream.where(CompactionCompleted).subscribe(lambda e: completions.append(e))

        agent = Agent(
            "compactor",
            config=TestConfig(ModelResponse(ModelMessage("hi"))),
            knowledge=KnowledgeConfig(
                store=store,
                compact=TailWindowCompact(target=2),
                compact_trigger=CompactTrigger(max_events=100),
            ),
        )
        await agent.ask("once", stream=stream)

        assert completions == []

    @pytest.mark.asyncio
    async def test_max_tokens_fires_on_large_content(self) -> None:
        # A single large turn (~2500 est. tokens) must cross max_tokens. The old
        # truncated str(event) estimate capped it near zero and never fired.
        store = MemoryKnowledgeStore()
        stream = MemoryStream()
        completions: list[CompactionCompleted] = []
        stream.where(CompactionCompleted).subscribe(lambda e: completions.append(e))

        agent = Agent(
            "compactor",
            config=TestConfig(ModelResponse(ModelMessage("ok"))),
            knowledge=KnowledgeConfig(
                store=store,
                compact=TailWindowCompact(target=1),
                compact_trigger=CompactTrigger(max_tokens=1000),
            ),
        )
        await agent.ask("x" * 10_000, stream=stream)

        assert len(completions) >= 1


class _RaisingCompact:
    """CompactStrategy that always raises — for failure-path tests."""

    last_usage: dict = {}

    async def compact(self, events, context, store) -> list:
        raise RuntimeError("compact boom")


@pytest.mark.asyncio
class TestCompactionLifecycleEvents:
    """Started + Failed events must reach the stream so failures are
    observable without configuring Python logging."""

    async def test_started_event_fires_before_strategy_runs(self) -> None:
        store = MemoryKnowledgeStore()
        stream = MemoryStream()
        started: list[CompactionStarted] = []
        stream.where(CompactionStarted).subscribe(lambda e: started.append(e))

        agent = Agent(
            "compactor",
            config=TestConfig(
                ModelResponse(ModelMessage("a")),
                ModelResponse(ModelMessage("b")),
                ModelResponse(ModelMessage("c")),
                ModelResponse(ModelMessage("d")),
            ),
            knowledge=KnowledgeConfig(
                store=store,
                compact=TailWindowCompact(target=2),
                compact_trigger=CompactTrigger(max_events=3),
            ),
        )
        reply = await agent.ask("turn-1", stream=stream)
        for q in ("turn-2", "turn-3", "turn-4"):
            reply = await reply.ask(q)

        assert started
        assert started[0].agent == "compactor"
        assert started[0].strategy == "TailWindowCompact"

    async def test_failed_event_fires_when_strategy_raises(self) -> None:
        store = MemoryKnowledgeStore()
        stream = MemoryStream()
        failures: list[CompactionFailed] = []
        completions: list[CompactionCompleted] = []
        stream.where(CompactionFailed).subscribe(lambda e: failures.append(e))
        stream.where(CompactionCompleted).subscribe(lambda e: completions.append(e))

        agent = Agent(
            "broken-compactor",
            config=TestConfig(
                ModelResponse(ModelMessage("a")),
                ModelResponse(ModelMessage("b")),
            ),
            knowledge=KnowledgeConfig(
                store=store,
                compact=_RaisingCompact(),
                compact_trigger=CompactTrigger(max_events=1),
            ),
        )
        # The turn itself succeeds — only compaction failed.
        await agent.ask("turn-1", stream=stream)

        assert len(failures) == 1
        assert failures[0].agent == "broken-compactor"
        assert failures[0].strategy == "_RaisingCompact"
        assert failures[0].error_type == "RuntimeError"
        assert "compact boom" in failures[0].error
        assert completions == []
