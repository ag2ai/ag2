# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the harness-layer bug sweep.

Each test locks in the fix for a specific bug caught during the Actor +
harness audit:

1. ``_HaltCheckMiddleware`` no longer leaks a ``HaltEvent`` subscription
   onto the user's stream — the subscription is scoped to one turn and
   unsubscribed in a ``finally``.
2. ``Actor`` emits ``ObserverStarted`` after observers register and
   ``ObserverCompleted`` before they unregister, so observers subscribed
   to their own lifecycle events actually receive them.
3. Concurrent asks on the same ``Actor`` bootstrap the knowledge store at
   most once (``asyncio.Lock`` + ``_bootstrap_done`` flag).
4. ``AlertPolicy`` deduplicates on ``(source, severity, message)``
   content rather than ``id(event)``, surviving history replay and
   compaction.
5. ``AggregateTrigger(on_end=True)`` fires through
   ``_AggregationMiddleware`` (one unified code path) and only once per
   ask even if other trigger conditions also match.
6. The auto-injected ``run_subtask`` tool delegates to
   ``tools.subagents.run_task.run_task`` and therefore emits the unified
   ``TaskStarted`` / ``TaskCompleted`` event taxonomy.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import ExitStack
from typing import Any

import pytest
from typing_extensions import Self

from autogen.beta import Actor, BaseObserver, KnowledgeConfig, TaskConfig
from autogen.beta.actor import (
    _AggregationMiddleware,
    _HaltCheckMiddleware,
)
from autogen.beta.aggregate import AggregateTrigger
from autogen.beta.annotations import Context
from autogen.beta.config import LLMClient, ModelConfig
from autogen.beta.context import ConversationContext as ContextType
from autogen.beta.events import (
    BaseEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TaskCompleted,
    TaskStarted,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
)
from autogen.beta.events.alert import HaltEvent, ObserverAlert, Severity
from autogen.beta.events.conditions import TypeCondition
from autogen.beta.events.lifecycle import (
    AggregationCompleted,
    ObserverCompleted,
    ObserverStarted,
)
from autogen.beta.knowledge import MemoryKnowledgeStore, StoreBootstrap
from autogen.beta.policies import AlertPolicy
from autogen.beta.stream import MemoryStream
from autogen.beta.watch import EventWatch


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _CannedClient(LLMClient):
    """LLM client that returns a deterministic sequence of ModelResponses."""

    def __init__(self, *responses: ModelResponse | ToolCallEvent | str) -> None:
        self._responses = list(responses)
        self._i = 0

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: ContextType,
        **kwargs: Any,
    ) -> ModelResponse:
        resp = self._responses[self._i] if self._i < len(self._responses) else "done"
        self._i += 1
        if isinstance(resp, str):
            return ModelResponse(message=ModelMessage(content=resp))
        if isinstance(resp, ToolCallEvent):
            return ModelResponse(tool_calls=ToolCallsEvent(calls=[resp]))
        return resp


class _CannedConfig(ModelConfig):
    __test__ = False

    def __init__(self, *responses: ModelResponse | ToolCallEvent | str) -> None:
        self._responses = responses

    def copy(self) -> Self:
        return self

    def create(self) -> _CannedClient:
        return _CannedClient(*self._responses)


class _CountingBootstrap(StoreBootstrap):
    """Bootstrap that counts invocations so we can catch double-bootstrap."""

    def __init__(self) -> None:
        self.calls = 0

    async def bootstrap(self, store, actor_name: str) -> None:
        self.calls += 1
        await store.write("/SKILL.md", f"bootstrapped by {actor_name}")


class _FatalObserver(BaseObserver):
    """Emits a single FATAL ObserverAlert on the first matching event."""

    def __init__(self) -> None:
        super().__init__("fatal-obs", watch=EventWatch(ModelResponse))
        self._fired = False

    async def process(self, events, ctx) -> ObserverAlert | None:
        if self._fired:
            return None
        self._fired = True
        return ObserverAlert(
            source=self.name,
            severity=Severity.FATAL,
            message="stop",
        )


# ---------------------------------------------------------------------------
# Bug 3: _HaltCheckMiddleware subscription cleanup
# ---------------------------------------------------------------------------


class TestHaltCheckMiddlewareSubscriptionCleanup:
    @pytest.mark.asyncio
    async def test_on_turn_unsubscribes_halt_listener(self) -> None:
        """After the turn, the HaltEvent subscription is gone from the stream."""
        stream = MemoryStream()
        ctx = ContextType(stream=stream)
        initial_event = ModelRequest([TextInput("start")])

        # Baseline sub count (stream.subscribe is used internally for history save)
        baseline = len(stream._subscribers)

        mw = _HaltCheckMiddleware(initial_event, ctx)

        async def call_next(event, context):
            # Mid-turn: exactly one additional subscriber (the HaltEvent listener)
            assert len(stream._subscribers) == baseline + 1
            return ModelResponse(message=ModelMessage(content="ok"))

        await mw.on_turn(call_next, initial_event, ctx)

        # Post-turn: subscription is gone
        assert len(stream._subscribers) == baseline

    @pytest.mark.asyncio
    async def test_on_turn_unsubscribes_even_on_exception(self) -> None:
        """If the turn raises, the subscription is still cleaned up."""
        stream = MemoryStream()
        ctx = ContextType(stream=stream)
        initial_event = ModelRequest([TextInput("start")])
        baseline = len(stream._subscribers)

        mw = _HaltCheckMiddleware(initial_event, ctx)

        async def exploding_call_next(event, context):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await mw.on_turn(exploding_call_next, initial_event, ctx)

        assert len(stream._subscribers) == baseline

    @pytest.mark.asyncio
    async def test_halt_event_emitted_during_turn_short_circuits(self) -> None:
        """Halt fired inside the turn causes on_llm_call to return HALTED."""
        stream = MemoryStream()
        ctx = ContextType(stream=stream)
        initial_event = ModelRequest([TextInput("start")])

        mw = _HaltCheckMiddleware(initial_event, ctx)

        async def inner_llm_call(events, context):
            return ModelResponse(message=ModelMessage(content="should not reach LLM"))

        async def call_next(event, context):
            # Emit a HaltEvent while the subscription is armed
            await context.send(HaltEvent(reason="policy blew up", source="fatal-obs"))
            # Subsequent LLM call should be short-circuited
            return await mw.on_llm_call(inner_llm_call, [], context)

        result = await mw.on_turn(call_next, initial_event, ctx)
        assert isinstance(result, ModelResponse)
        assert result.content is not None
        assert "HALTED" in result.content
        assert "policy blew up" in result.content


# ---------------------------------------------------------------------------
# Bug 4: ObserverStarted / Completed emitted while observers are registered
# ---------------------------------------------------------------------------


class _SelfAwareObserver(BaseObserver):
    """Observer that watches ObserverStarted & ObserverCompleted on itself."""

    def __init__(self, name: str = "self-aware") -> None:
        super().__init__(name, watch=EventWatch(ObserverStarted | ObserverCompleted))
        self.started_seen: list[str] = []
        self.completed_seen: list[str] = []

    async def process(self, events, ctx) -> None:
        for event in events:
            if isinstance(event, ObserverStarted):
                self.started_seen.append(event.name)
            elif isinstance(event, ObserverCompleted):
                self.completed_seen.append(event.name)
        return None


class TestObserverLifecycleSelfVisibility:
    @pytest.mark.asyncio
    async def test_observer_sees_own_started_and_completed(self) -> None:
        """An observer subscribed to its own lifecycle event receives it."""
        config = _CannedConfig("hello")
        obs = _SelfAwareObserver()

        agent = Actor("with-obs", config=config, observers=[obs])
        await agent.ask("hi")

        assert obs.started_seen == ["self-aware"]
        assert obs.completed_seen == ["self-aware"]

    @pytest.mark.asyncio
    async def test_started_emitted_after_register(self) -> None:
        """External subscriber sees ObserverStarted AFTER the stream is wired up.

        Before the fix, the Started event was fired before the observer was
        registered on the stream, which meant the observer itself could not
        react to its own start. After the fix the emission is inside the
        ExitStack that holds the observer registration.
        """
        config = _CannedConfig("hi")
        stream = MemoryStream()
        started: list[ObserverStarted] = []
        stream.where(ObserverStarted).subscribe(lambda e: started.append(e))

        obs = _SelfAwareObserver(name="alpha")
        agent = Actor("lifecycle", config=config, observers=[obs])
        await agent.ask("go", stream=stream)

        # External listener still sees Started (and only one, matching obs.name)
        assert len(started) == 1
        assert started[0].name == "alpha"


# ---------------------------------------------------------------------------
# Bug 5: Bootstrap TOCTOU race
# ---------------------------------------------------------------------------


class TestBootstrapRace:
    @pytest.mark.asyncio
    async def test_concurrent_asks_bootstrap_once(self) -> None:
        """Two concurrent asks on the same actor run bootstrap exactly once."""
        store = MemoryKnowledgeStore()
        counter = _CountingBootstrap()

        def build_agent() -> Actor:
            return Actor(
                "booter",
                config=_CannedConfig("ok"),
                knowledge=KnowledgeConfig(store=store, bootstrap=counter),
            )

        actor = build_agent()
        await asyncio.gather(
            actor.ask("hi", stream=MemoryStream()),
            actor.ask("hi", stream=MemoryStream()),
            actor.ask("hi", stream=MemoryStream()),
        )

        assert counter.calls == 1
        # And the sentinel exists
        assert await store.exists("/.initialized")

    @pytest.mark.asyncio
    async def test_second_actor_same_store_respects_sentinel(self) -> None:
        """A brand-new Actor on an already-initialized store doesn't re-bootstrap."""
        store = MemoryKnowledgeStore()
        await store.write("/.initialized", "prior-actor")

        counter = _CountingBootstrap()
        actor = Actor(
            "later",
            config=_CannedConfig("ok"),
            knowledge=KnowledgeConfig(store=store, bootstrap=counter),
        )
        await actor.ask("hi")

        assert counter.calls == 0


# ---------------------------------------------------------------------------
# Bug 6: AlertPolicy content-based dedup
# ---------------------------------------------------------------------------


class TestAlertPolicyContentDedup:
    @pytest.mark.asyncio
    async def test_identical_content_from_distinct_events_dedups(self) -> None:
        """Two ObserverAlert events with identical content are delivered once."""
        policy = AlertPolicy()
        ctx = ContextType(stream=MemoryStream())

        a = ObserverAlert(source="mon", severity=Severity.WARNING, message="over budget")
        b = ObserverAlert(source="mon", severity=Severity.WARNING, message="over budget")
        assert a is not b  # distinct Python objects

        # First call delivers
        prompts1, _ = await policy.apply([], [a], ctx)
        assert any("over budget" in p for p in prompts1)

        # Second call, same content key — deduped
        prompts2, _ = await policy.apply([], [b], ctx)
        assert all("over budget" not in p for p in prompts2)

    @pytest.mark.asyncio
    async def test_different_message_not_deduped(self) -> None:
        policy = AlertPolicy()
        ctx = ContextType(stream=MemoryStream())

        prompts1, _ = await policy.apply(
            [],
            [ObserverAlert(source="mon", severity=Severity.WARNING, message="one")],
            ctx,
        )
        prompts2, _ = await policy.apply(
            [],
            [ObserverAlert(source="mon", severity=Severity.WARNING, message="two")],
            ctx,
        )
        assert any("one" in p for p in prompts1)
        assert any("two" in p for p in prompts2)

    @pytest.mark.asyncio
    async def test_dedup_survives_history_replace(self) -> None:
        """Rewriting the history with a fresh copy of an alert still dedupes.

        Compaction constructs new event objects, so id() dedup would
        re-deliver the alert. Content dedup does not.
        """
        policy = AlertPolicy()
        ctx = ContextType(stream=MemoryStream())

        original = ObserverAlert(source="mon", severity=Severity.WARNING, message="watch out")
        await policy.apply([], [original], ctx)

        replaced = ObserverAlert(source="mon", severity=Severity.WARNING, message="watch out")
        prompts, _ = await policy.apply([], [replaced], ctx)
        assert all("watch out" not in p for p in prompts)


# ---------------------------------------------------------------------------
# Bug 11: on_end aggregation via unified middleware path
# ---------------------------------------------------------------------------


class _TrackedAggregate:
    def __init__(self) -> None:
        self.calls = 0
        self.last_usage: dict = {}

    async def aggregate(self, events, context, store) -> None:
        self.calls += 1
        await store.write(f"/memory/runs/{self.calls}.md", "rolled up")


class TestAggregationOnEndViaMiddleware:
    @pytest.mark.asyncio
    async def test_on_end_fires_exactly_once_per_ask(self) -> None:
        """A bare on_end=True trigger fires once per ask via the middleware."""
        store = MemoryKnowledgeStore()
        strategy = _TrackedAggregate()
        events_seen: list[AggregationCompleted] = []

        stream = MemoryStream()
        stream.where(AggregationCompleted).subscribe(lambda e: events_seen.append(e))

        agent = Actor(
            "roller",
            config=_CannedConfig("done"),
            knowledge=KnowledgeConfig(
                store=store,
                aggregate=strategy,
                aggregate_trigger=AggregateTrigger(on_end=True),
            ),
        )
        await agent.ask("go", stream=stream)

        assert strategy.calls == 1
        assert len(events_seen) == 1
        assert events_seen[0].actor == "roller"

    @pytest.mark.asyncio
    async def test_on_end_runs_even_when_turn_raises(self) -> None:
        """on_end aggregation still fires if the turn errored.

        The middleware uses try/finally so post-turn aggregation is
        resilient to failures in the underlying call chain.
        """
        store = MemoryKnowledgeStore()
        strategy = _TrackedAggregate()
        stream = MemoryStream()
        ctx = ContextType(stream=stream)
        initial_event = ModelRequest([TextInput("start")])

        mw = _AggregationMiddleware(
            initial_event,
            ctx,
            actor_name="test",
            strategy=strategy,
            store=store,
            trigger=AggregateTrigger(on_end=True),
        )

        async def exploding_call_next(event, context):
            raise RuntimeError("LLM down")

        with pytest.raises(RuntimeError):
            await mw.on_turn(exploding_call_next, initial_event, ctx)

        assert strategy.calls == 1

    @pytest.mark.asyncio
    async def test_on_end_does_not_double_fire_with_other_triggers(self) -> None:
        """With on_end=True AND every_n_turns=1, aggregation still only fires once."""
        store = MemoryKnowledgeStore()
        strategy = _TrackedAggregate()

        agent = Actor(
            "once",
            config=_CannedConfig("ok"),
            knowledge=KnowledgeConfig(
                store=store,
                aggregate=strategy,
                aggregate_trigger=AggregateTrigger(every_n_turns=1, on_end=True),
            ),
        )
        await agent.ask("go")

        # Both conditions match, but the middleware still calls aggregate() once
        assert strategy.calls == 1


# ---------------------------------------------------------------------------
# Bug 1+2: Unified task event emission via run_task()
# ---------------------------------------------------------------------------


class TestTaskEventsUnified:
    @pytest.mark.asyncio
    async def test_run_subtask_emits_new_taxonomy(self) -> None:
        """The auto-injected run_subtask tool emits TaskStarted/TaskCompleted.

        Before the fix, it emitted TaskRequest/TaskResult from
        ``lifecycle.py``. After the fix it delegates to ``run_task()`` which
        emits from ``task_events.py``, unifying the event trees.
        """
        starts: list[TaskStarted] = []
        completions: list[TaskCompleted] = []

        stream = MemoryStream()
        stream.where(TaskStarted).subscribe(lambda e: starts.append(e))
        stream.where(TaskCompleted).subscribe(lambda e: completions.append(e))

        parent = Actor(
            "parent",
            config=_CannedConfig("from parent"),
            tasks=TaskConfig(config=_CannedConfig("from child")),
        )

        # Call the auto-injected run_subtask directly via _spawn_subtask.
        # (Doing it via ask() requires the LLM to actually invoke the tool,
        # which is tested in the smoke suite.)
        ctx = ContextType(stream=stream)
        result = await parent._spawn_subtask("what is 2+2", ctx)

        assert result == "from child"
        assert len(starts) == 1
        assert len(completions) == 1
        assert starts[0].agent_name == completions[0].agent_name
        assert starts[0].task_id == completions[0].task_id
        assert starts[0].objective == "what is 2+2"

    @pytest.mark.asyncio
    async def test_run_subtask_completions_carry_usage(self) -> None:
        """TaskCompleted has a usage dict populated from the child reply."""
        completions: list[TaskCompleted] = []
        stream = MemoryStream()
        stream.where(TaskCompleted).subscribe(lambda e: completions.append(e))

        from autogen.beta.events.types import Usage

        child_response = ModelResponse(
            message=ModelMessage(content="42"),
            usage=Usage(prompt_tokens=7, completion_tokens=2, total_tokens=9),
        )

        parent = Actor(
            "parent",
            config=_CannedConfig("ignored"),
            tasks=TaskConfig(config=_CannedConfig(child_response)),
        )
        ctx = ContextType(stream=stream)
        await parent._spawn_subtask("answer 40+2", ctx)

        assert len(completions) == 1
        assert completions[0].usage == {
            "prompt_tokens": 7,
            "completion_tokens": 2,
            "total_tokens": 9,
        }
