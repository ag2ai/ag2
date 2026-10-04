# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Rollouts: resuming an agent from a boundary and letting it run.

Protocol for one rollout:

1. restore a **fresh** environment to the state at the cut
   (:meth:`Environment.restore`), with no model in the loop;
2. show the agent exactly one context — the raw history (PRE) or the compacted
   one (POST) — through :meth:`ag2.Agent.resume`;
3. let it run free until it has executed ``horizon`` tool calls or finishes,
   and record every real result.

Nothing is teacher-forced: what the recorded run did after the cut is never
shown to the agent. Both arms of a boundary restore the same prefix, so a
difference between them is attributable to the context alone. PRE does not
depend on the strategy, so one set of PRE rollouts serves every strategy
compared at a boundary.
"""

import asyncio
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ag2 import Agent
from ag2.annotations import Context
from ag2.compact import CompactStrategy
from ag2.eval import Trace
from ag2.events import BaseEvent, ModelMessage, ModelResponse, ToolCallEvent
from ag2.middleware import BaseMiddleware
from ag2.middleware.base import LLMCall, ToolExecution, ToolResultType
from ag2.stream import MemoryStream

from .actions import Action, actions_from_events
from .boundary import Boundary, CompactedContext, compact_context, make_boundary, select_cuts
from .burden import score_boundary
from .environment import Environment
from .report import Arm, BoundaryResult, Rollout, StrategyReport, VerificationReport

__all__ = ("ActionBudget", "CompactionVerifier", "Recording", "run_rollout")

logger = logging.getLogger(__name__)

BUDGET_REACHED = "[compaction verifier] action budget reached"


class ActionBudget:
    """Middleware factory that ends a turn once ``limit`` tool calls have executed.

    After the ``limit``-th execution the model is not called again: the turn
    ends with a synthetic final message instead, the same way AG2's halt check
    ends a turn. Calls the model issued in one response all execute, so a
    rollout can overshoot ``limit`` by the width of its last response; scoring
    reads only the first ``limit`` actions.
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self._limit = limit

    def __call__(self, event: BaseEvent, context: Context) -> BaseMiddleware:
        return _ActionBudgetMiddleware(event, context, limit=self._limit)


class _ActionBudgetMiddleware(BaseMiddleware):
    def __init__(self, event: BaseEvent, context: Context, *, limit: int) -> None:
        super().__init__(event, context)
        self._limit = limit
        self._executed = 0

    async def on_tool_execution(
        self,
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        result = await call_next(event, context)
        self._executed += 1
        return result

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        if self._executed >= self._limit:
            return ModelResponse(ModelMessage(content=BUDGET_REACHED))
        return await call_next(events, context)


async def run_rollout(
    agent: Agent[Any],
    context: Sequence[BaseEvent],
    environment: Environment,
    prefix: Sequence[Action],
    *,
    horizon: int,
    arm: Arm,
    sample: int = 0,
) -> Rollout:
    """Resume ``agent`` from ``context`` in a restored environment and record what it does.

    A failure while the agent runs (a provider error, say) is recorded on the
    rollout rather than raised, so one bad sample does not discard a boundary;
    such a rollout is not scored. A failure to restore the environment is
    raised: every rollout of that boundary would start from the wrong state.
    """
    tools = await environment.restore(prefix)
    stream = MemoryStream()
    emitted = _EventLog()
    # Only what this run emits: the seeded history is written to storage, not
    # sent, so the agent's new calls are read here whatever ids they reuse.
    stream.subscribe(emitted.add, sync_to_thread=False)
    try:
        await agent.resume(*context, stream=stream, tools=tools, middleware=[ActionBudget(horizon)])
        actions = tuple(actions_from_events(emitted.events))
    except Exception as exc:
        logger.warning("%s rollout %d failed: %s: %s", arm, sample, type(exc).__name__, exc)
        return Rollout(arm=arm, sample=sample, actions=(), ending="error", error=f"{type(exc).__name__}: {exc}")
    return Rollout(arm=arm, sample=sample, actions=actions, ending="budget" if len(actions) >= horizon else "answer")


class _EventLog:
    __slots__ = ("events",)

    def __init__(self) -> None:
        self.events: list[BaseEvent] = []

    async def add(self, event: BaseEvent) -> None:
        self.events.append(event)


@dataclass(frozen=True, slots=True)
class Recording:
    """A recorded run and the environment it ran against.

    Attributes:
        events: The run's history — an event sequence, or an
            :class:`ag2.eval.Trace` from an eval run.
        environment: Rebuilds this run's environment at a cut; see
            :class:`Environment`. Runs of different tasks usually start from
            different states, so each recording carries its own.
        cuts: Explicit cut indices. When ``None``, cuts are chosen by
            :func:`select_cuts`.
    """

    events: Sequence[BaseEvent] | Trace
    environment: Environment
    cuts: Sequence[int] | None = None


class CompactionVerifier:
    """Measures whether compacting an agent's history makes its next actions worse.

    For every boundary of every recorded trajectory, the agent is resumed
    ``samples`` times from the raw history (PRE) and ``samples`` times from each
    strategy's compaction of it (POST), in an environment restored to the same
    state each time. A strategy's score at a boundary is POST minus PRE burden
    over the first ``horizon`` actions (see :mod:`.burden`), so the agent's own
    run-to-run variance is differenced out rather than assumed away.

    Args:
        agent: The agent under test, built without the environment's tools —
            each rollout passes tools bound to its own restored environment.
            Sampling variance only averages out if its model samples with a
            non-zero temperature.
        horizon: Actions per rollout, and the ``k`` scores are read at.
        samples: Rollouts per arm per boundary.
        concurrency: Rollouts and compactions allowed in flight at once.
    """

    def __init__(
        self,
        agent: Agent[Any],
        *,
        horizon: int = 5,
        samples: int = 3,
        concurrency: int = 4,
    ) -> None:
        if horizon < 1 or samples < 1 or concurrency < 1:
            raise ValueError("horizon, samples and concurrency must each be at least 1")
        self.agent = agent
        self.horizon = horizon
        self.samples = samples
        self.concurrency = concurrency

    async def verify(
        self,
        recordings: Iterable[Recording],
        strategies: Mapping[str, CompactStrategy],
        *,
        every: int = 5,
        min_prefix: int = 3,
    ) -> VerificationReport:
        """Score every strategy at every boundary of ``recordings``.

        Args:
            recordings: Recorded runs, each with its environment.
            strategies: The strategies to compare, by display name. They are
                scored against the same PRE rollouts.
            every: Actions between consecutive chosen cuts, for recordings
                without explicit ``cuts``.
            min_prefix: Actions that must precede the first chosen cut.
        """
        if not strategies:
            raise ValueError("give at least one strategy to verify")
        planned: list[tuple[Boundary, Environment]] = []
        for t, recording in enumerate(recordings):
            events = recording.events.events if isinstance(recording.events, Trace) else tuple(recording.events)
            cuts = recording.cuts
            if cuts is None:
                cuts = select_cuts(events, every=every, min_prefix=min_prefix)
            planned.extend((make_boundary(events, cut, trajectory=t), recording.environment) for cut in cuts)
        logger.info("verifying %d strategies at %d boundaries", len(strategies), len(planned))

        gate = asyncio.Semaphore(self.concurrency)
        results = await asyncio.gather(*(self._verify_boundary(b, env, strategies, gate) for b, env in planned))

        by_strategy: dict[str, list[BoundaryResult]] = {name: [] for name in strategies}
        failed_pre = 0
        for boundary_results, pre_failures in results:
            failed_pre += pre_failures
            for result in boundary_results:
                by_strategy[result.strategy].append(result)
        return VerificationReport(
            horizon=self.horizon,
            samples=self.samples,
            strategies={name: StrategyReport.build(name, rs, self.horizon) for name, rs in by_strategy.items()},
            failed_pre_rollouts=failed_pre,
        )

    async def _verify_boundary(
        self,
        boundary: Boundary,
        environment: Environment,
        strategies: Mapping[str, CompactStrategy],
        gate: asyncio.Semaphore,
    ) -> tuple[list[BoundaryResult], int]:
        names = list(strategies)
        compacted = await asyncio.gather(*(self._gated_compact(strategies[n], boundary, gate) for n in names))
        pre_task = asyncio.gather(
            *(self._gated_rollout(boundary.context, boundary, environment, "pre", i, gate) for i in range(self.samples))
        )
        post_tasks = [
            asyncio.gather(
                *(self._gated_rollout(c.events, boundary, environment, "post", i, gate) for i in range(self.samples))
            )
            for c in compacted
        ]
        pre, *posts = await asyncio.gather(pre_task, *post_tasks)
        scorable_pre = [r.actions for r in pre if r.ending != "error"]
        results = []
        for name, context, post in zip(names, compacted, posts, strict=True):
            scorable_post = [r.actions for r in post if r.ending != "error"]
            deltas = (
                score_boundary(scorable_pre, scorable_post, boundary.history, range(1, self.horizon + 1))
                if scorable_pre and scorable_post
                else ()
            )
            results.append(
                BoundaryResult(
                    trajectory=boundary.trajectory,
                    cut=boundary.cut,
                    strategy=name,
                    deltas=deltas,
                    pre_size=boundary.size,
                    post_size=context.size,
                    pre=tuple(pre),
                    post=tuple(post),
                    compaction_tokens=_spent(context),
                    unchanged=context.events == boundary.context,
                )
            )
        logger.info("boundary %d@%d done", boundary.trajectory, boundary.cut)
        return results, sum(1 for r in pre if r.ending == "error")

    async def _gated_compact(
        self, strategy: CompactStrategy, boundary: Boundary, gate: asyncio.Semaphore
    ) -> CompactedContext:
        async with gate:
            return await compact_context(strategy, boundary)

    async def _gated_rollout(
        self,
        context: Sequence[BaseEvent],
        boundary: Boundary,
        environment: Environment,
        arm: Arm,
        sample: int,
        gate: asyncio.Semaphore,
    ) -> Rollout:
        async with gate:
            return await run_rollout(
                self.agent, context, environment, boundary.prefix, horizon=self.horizon, arm=arm, sample=sample
            )


def _spent(context: CompactedContext) -> int:
    total = 0
    for event in context.usage:
        usage = event.usage
        total += int(usage.prompt_tokens or 0) + int(usage.completion_tokens or 0)
    return total
