# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from ag2.annotations import Context
from ag2.events import (
    AggregationCompleted,
    AggregationFailed,
    AggregationStarted,
    BaseEvent,
    CompactionCompleted,
    CompactionFailed,
    CompactionStarted,
    ModelRequest,
    ModelResponse,
    ToolCallsEvent,
    ToolResultsEvent,
    UsageEvent,
    estimated_tokens,
    is_conversational,
)
from ag2.middleware.base import BaseMiddleware
from ag2.usage import collect_usage_events, usage_event_from
from ag2.utils import AGENT_CONTEXT_DEPENDENCY_KEY

from .base import KnowledgeStore

if TYPE_CHECKING:
    from ag2.aggregate import AggregateStrategy, AggregateTrigger
    from ag2.compact import CompactStrategy, CompactTrigger


class _CompactionMiddleware(BaseMiddleware):
    """Triggers compaction after agent turns when thresholds are exceeded."""

    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        *,
        actor_name: str,
        strategy: "CompactStrategy",
        store: KnowledgeStore | None,
        trigger: "CompactTrigger",
    ) -> None:
        super().__init__(event, context)
        self._actor_name = actor_name
        self._strategy = strategy
        self._store = store
        self._trigger = trigger
        self._last_compact_event_count = 0

    async def on_turn(
        self,
        call_next: Callable[..., Any],
        event: BaseEvent,
        context: Context,
    ) -> Any:
        result = await call_next(event, context)

        events = list(await context.stream.history.get_events())
        # Count only conversational events — transient artifacts and persisted
        # telemetry (UsageEvent) must not drive compaction, even if persist_all.
        conversation_events = [e for e in events if is_conversational(e)]
        event_count = len(conversation_events)

        # Prevent double compaction — skip if count hasn't grown since last
        if event_count <= self._last_compact_event_count:
            return result

        should_compact = False
        if self._trigger.max_events > 0 and event_count > self._trigger.max_events:
            should_compact = True
        if self._trigger.max_tokens > 0:
            # Sized over every event: telemetry weighs nothing, but retained
            # status is resent on each call and must count toward the budget.
            estimated = sum(estimated_tokens(e, self._trigger.chars_per_token) for e in events)
            if estimated > self._trigger.max_tokens:
                should_compact = True

        if should_compact:
            strategy_name = type(self._strategy).__name__
            await context.send(
                CompactionStarted(
                    agent=self._actor_name,
                    strategy=strategy_name,
                    event_count=len(events),
                )
            )
            # Collected live rather than re-read from history afterwards: the
            # summarization call's own record is sent *while* the strategy runs,
            # so it is in neither ``events`` nor what the strategy returns. A
            # subscriber sees it as it is sent, which needs no assumption about
            # how history grew in the meantime and no second full read of it.
            spent_during: list[UsageEvent] = []
            try:
                with context.stream.where(UsageEvent).sub_scope(collect_usage_events(spent_during)):
                    compacted = await self._strategy.compact(events, context, self._store)
            except Exception as exc:
                logging.getLogger(__name__).exception("Compaction failed for %s", self._actor_name)
                with suppress(Exception):
                    await context.send(
                        CompactionFailed(
                            agent=self._actor_name,
                            strategy=strategy_name,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                    )
                return result

            await context.stream.history.replace(
                _with_usage_events(before=events, compacted=compacted, spent_during=spent_during)
            )
            self._last_compact_event_count = len([e for e in compacted if is_conversational(e)])

            usage = getattr(self._strategy, "last_usage", {})
            await context.send(
                CompactionCompleted(
                    agent=self._actor_name,
                    strategy=strategy_name,
                    events_before=len(events),
                    events_after=len(compacted),
                    llm_calls=1 if usage else 0,
                    usage=usage,
                )
            )

        return result


def _with_usage_events(
    *,
    before: list[BaseEvent],
    compacted: list[BaseEvent],
    spent_during: list[UsageEvent],
) -> list[BaseEvent]:
    """``compacted``, with every token record of the run put back around it.

    Compaction replaces stream history with what its strategy retained, and
    ``UsageEvent`` is the only thing ``UsageReport`` reads (`ADR 0014`), so
    without this a run that compacted under-reports what it cost — on
    ``AgentReply.usage()`` and on every transport that reports spend. Two
    records are lost, for two different reasons: those outside the retained
    window, which the strategy drops along with the conversation, and the
    summarization call's own, which is emitted *while* the strategy runs and is
    therefore absent from the snapshot the strategy returns. ``spent_during``
    carries the second kind, collected from the stream as they were sent.

    The records are rebuilt from ``before`` rather than from ``compacted``, so a
    strategy that retained a record and one that dropped it read the same: each
    record of the run appears exactly once, whatever the strategy did with it.
    The one case this does not cover is a strategy that *synthesises* a new
    ``UsageEvent`` into its returned list without sending it — that record is
    dropped here because it was never published as a source accounting record.
    A strategy that ``context.send``s its spend, as the built-in ones do, is
    accounted correctly. Source ``event_id`` values survive serialization, so
    retained or replayed representations of a publication count once.

    Order is telemetry from before the compaction, then the retained
    conversation, then what was spent during it. That is not the chronological
    order of the run — records from inside the retained window are hoisted ahead
    of the conversation they were interleaved with — but the token records keep
    their order relative to each other, which is what consumers that group by
    first appearance need (AG-UI's per-``(provider, model)`` list among them).

    None of this feeds the model or moves the compaction trigger: ``UsageEvent``
    is not conversational (`ADR 0010`).

    Note that the kept records accumulate: every compaction carries forward
    every record the stream has seen, so a long-lived stream keeps one event per
    model call for its whole life and each compaction rewrites all of them. That
    is the cost of `ADR 0014`'s decision to keep spend in the event log; bounding
    it belongs with that decision, not here.
    """
    return [
        *(e for e in before if usage_event_from(e) is not None),
        *(e for e in compacted if usage_event_from(e) is None),
        *spent_during,
    ]


class _CompactionMiddlewareFactory:
    """Factory for _CompactionMiddleware."""

    def __init__(
        self,
        strategy: "CompactStrategy",
        trigger: "CompactTrigger",
        legacy_store: KnowledgeStore | None = None,
    ) -> None:
        self._strategy = strategy
        self._trigger = trigger
        self._legacy_store = legacy_store

    def __call__(self, event: BaseEvent, context: Context) -> _CompactionMiddleware:
        return _CompactionMiddleware(
            event,
            context,
            actor_name=context.dependencies[AGENT_CONTEXT_DEPENDENCY_KEY].name,
            strategy=self._strategy,
            store=_effective_store(self._legacy_store, context),
            trigger=self._trigger,
        )


class _AggregationMiddleware(BaseMiddleware):
    """Triggers aggregation after agent turns when thresholds are exceeded.

    Counts are derived from stream history — this middleware holds no
    state of its own. That matters because ``_execute`` builds a fresh
    middleware instance on every ``ask()``; any per-instance counter
    would reset between turns and make ``every_n_turns=N`` for ``N>1``
    effectively dead.

    ``every_n_turns`` counts :class:`ModelRequest` events in history
    (one per user ask). ``every_n_events`` fires when the total history
    count crosses a multiple of the threshold during the current turn,
    which handles non-uniform growth (e.g. tool-heavy turns). ``on_end``
    fires unconditionally once the turn completes.
    """

    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        *,
        actor_name: str,
        strategy: "AggregateStrategy",
        store: KnowledgeStore,
        trigger: "AggregateTrigger",
    ) -> None:
        super().__init__(event, context)
        self._actor_name = actor_name
        self._strategy = strategy
        self._store = store
        self._trigger = trigger

    async def on_turn(
        self,
        call_next: Callable[..., Any],
        event: BaseEvent,
        context: Context,
    ) -> Any:
        events_before = list(await context.stream.history.get_events())

        try:
            result = await call_next(event, context)
        finally:
            events_after = list(await context.stream.history.get_events())
            count_after = len(events_after)

            should_aggregate = False

            if self._trigger.on_end:
                should_aggregate = True

            if self._trigger.every_n_turns > 0:
                turn_count = sum(1 for e in events_after if isinstance(e, ModelRequest))
                if turn_count > 0 and turn_count % self._trigger.every_n_turns == 0:
                    should_aggregate = True

            if self._trigger.every_n_events > 0:
                threshold = self._trigger.every_n_events
                # Count conversational/work events only, not telemetry (e.g. UsageEvent).
                _countable = (ModelRequest, ModelResponse, ToolCallsEvent, ToolResultsEvent)
                n_before = sum(1 for e in events_before if isinstance(e, _countable))
                n_after = sum(1 for e in events_after if isinstance(e, _countable))
                if n_after // threshold > n_before // threshold:
                    should_aggregate = True

            if should_aggregate:
                strategy_name = type(self._strategy).__name__
                with suppress(Exception):
                    await context.send(
                        AggregationStarted(
                            agent=self._actor_name,
                            strategy=strategy_name,
                            event_count=count_after,
                        )
                    )
                try:
                    await self._strategy.aggregate(events_after, context, self._store)
                    usage = getattr(self._strategy, "last_usage", {})
                    await context.send(
                        AggregationCompleted(
                            agent=self._actor_name,
                            strategy=strategy_name,
                            event_count=count_after,
                            llm_calls=1 if usage else 0,
                            usage=usage,
                        )
                    )
                except Exception as exc:
                    logging.getLogger(__name__).exception("Aggregation failed for %s", self._actor_name)
                    with suppress(Exception):
                        await context.send(
                            AggregationFailed(
                                agent=self._actor_name,
                                strategy=strategy_name,
                                error_type=type(exc).__name__,
                                error=str(exc),
                            )
                        )

        return result


class _AggregationMiddlewareFactory:
    """Factory for _AggregationMiddleware."""

    def __init__(
        self,
        strategy: "AggregateStrategy",
        trigger: "AggregateTrigger",
        legacy_store: KnowledgeStore | None = None,
    ) -> None:
        self._strategy = strategy
        self._trigger = trigger
        self._legacy_store = legacy_store

    def __call__(self, event: BaseEvent, context: Context) -> _AggregationMiddleware:
        return _AggregationMiddleware(
            event,
            context,
            actor_name=context.dependencies[AGENT_CONTEXT_DEPENDENCY_KEY].name,
            strategy=self._strategy,
            store=_effective_store(self._legacy_store, context),
            trigger=self._trigger,
        )


def _effective_store(bound: KnowledgeStore | None, context: Context) -> KnowledgeStore:
    """The legacy-bound store, or the invocation's effective ``KnowledgeStore`` dependency."""
    if bound is not None:
        return bound
    store: KnowledgeStore = context.dependencies[KnowledgeStore]
    return store
