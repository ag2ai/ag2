# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Boundaries: the points in a recorded trajectory where compaction is tested.

A boundary cuts a recorded event sequence right after a
:class:`~ag2.events.ToolResultsEvent` — the agent has just seen tool output and
is about to decide its next action, which is exactly where
:meth:`ag2.Agent.resume` re-enters the loop. Each boundary defines two contexts
for the **same** execution state:

* **PRE**, the control: the raw history up to the cut;
* **POST**, the treatment: what a :class:`~ag2.compact.CompactStrategy` returns
  for that history.

Both keep the cut's ``ToolResultsEvent`` as their last event, so both resume
from the same trigger, and both are scored against the calls the environment
actually executed before the cut.

AG2 fires compaction between turns (``_CompactionMiddleware`` runs after a turn
completes), so a long single-turn tool loop is never compacted mid-run. The
verifier does not depend on where a strategy happens to fire in production: it
asks what the strategy would cost *if* history were compacted at each cut.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from ag2.annotations import Context
from ag2.compact import CompactStrategy
from ag2.context import ConversationContext
from ag2.events import BaseEvent, ToolResultsEvent, UsageEvent, estimated_tokens, is_conversational
from ag2.stream import MemoryStream

from .actions import Action, actions_from_events, actions_with_positions
from .burden import history_signatures

__all__ = (
    "Boundary",
    "CompactedContext",
    "ContextSize",
    "compact_context",
    "make_boundary",
    "resumable_cuts",
    "select_cuts",
)


@dataclass(frozen=True, slots=True)
class ContextSize:
    """Size of a context as the model would see it."""

    events: int
    """Conversational events (telemetry excluded)."""
    tokens: int
    """Estimated tokens, from :func:`ag2.events.estimated_tokens`."""

    @classmethod
    def of(cls, events: Sequence[BaseEvent]) -> "ContextSize":
        return cls(
            events=sum(1 for e in events if is_conversational(e)),
            tokens=sum(estimated_tokens(e) for e in events),
        )


@dataclass(frozen=True, slots=True)
class Boundary:
    """One cut of one recorded trajectory.

    Attributes:
        trajectory: Index of the source trajectory in the caller's input.
        cut: ``events[:cut]`` is the history at the boundary.
        context: The raw history — the PRE context.
        prefix: The calls executed before the cut, replayed to restore the
            environment for both arms.
        history: Signatures of ``prefix`` by call (name + arguments).
        history_tools: Signatures of ``prefix`` by tool name only.
    """

    trajectory: int
    cut: int
    context: tuple[BaseEvent, ...]
    prefix: tuple[Action, ...]
    history: frozenset[str]
    history_tools: frozenset[str]

    @property
    def size(self) -> ContextSize:
        return ContextSize.of(self.context)


@dataclass(frozen=True, slots=True)
class CompactedContext:
    """What a strategy made of a boundary's history — the POST context."""

    events: tuple[BaseEvent, ...]
    usage: tuple[UsageEvent, ...]
    """Token records the strategy sent while compacting (its own LLM calls)."""

    @property
    def size(self) -> ContextSize:
        return ContextSize.of(self.events)


def resumable_cuts(events: Sequence[BaseEvent]) -> list[int]:
    """Every ``i`` such that ``events[:i]`` ends with a ``ToolResultsEvent``."""
    return [i + 1 for i, e in enumerate(events) if isinstance(e, ToolResultsEvent)]


def select_cuts(
    events: Sequence[BaseEvent],
    *,
    every: int = 5,
    min_prefix: int = 3,
) -> list[int]:
    """Resumable cuts spaced by executed actions.

    A cut is taken once at least ``min_prefix`` actions have executed, then again
    each time ``every`` more have. The last resumable point is skipped when the
    recorded run ended right after it, since there is nothing left to decide.

    Args:
        events: A recorded trajectory.
        every: Actions between consecutive cuts.
        min_prefix: Actions that must precede the first cut; compacting a
            history of one or two calls tests little.
    """
    if every < 1 or min_prefix < 1:
        raise ValueError("every and min_prefix must be at least 1")
    answered_at = sorted(at for _, at in actions_with_positions(events))
    total = len(answered_at)
    cuts: list[int] = []
    next_at = min_prefix
    executed = 0
    for cut in resumable_cuts(events):
        while executed < total and answered_at[executed] < cut:
            executed += 1
        if executed >= total:
            break
        if executed >= next_at:
            cuts.append(cut)
            next_at = executed + every
    return cuts


def make_boundary(events: Sequence[BaseEvent], cut: int, *, trajectory: int = 0) -> Boundary:
    """The boundary of ``events`` at ``cut``.

    Raises:
        ValueError: ``cut`` is not a resumable point of ``events``.
    """
    if cut < 1 or cut > len(events) or not isinstance(events[cut - 1], ToolResultsEvent):
        raise ValueError(f"cut {cut} is not right after a ToolResultsEvent; use resumable_cuts()")
    context = tuple(events[:cut])
    prefix = tuple(actions_from_events(context))
    return Boundary(
        trajectory=trajectory,
        cut=cut,
        context=context,
        prefix=prefix,
        history=history_signatures(prefix, "call"),
        history_tools=history_signatures(prefix, "tool"),
    )


async def compact_context(strategy: CompactStrategy, boundary: Boundary) -> CompactedContext:
    """Run ``strategy`` on a boundary's history and return the POST context.

    The strategy runs on a throwaway stream with no knowledge store, so nothing
    it drops is persisted anywhere, and its own spend is collected rather than
    lost.

    Raises:
        ValueError: The strategy dropped the boundary's final
            ``ToolResultsEvent``, so the POST arm would not resume from the same
            point as the PRE arm.
    """
    stream = MemoryStream()
    usage = _UsageCollector()
    stream.where(UsageEvent).subscribe(usage.add)
    context: Context = ConversationContext(stream)
    compacted = tuple(await strategy.compact(list(boundary.context), context, None))
    if not compacted or compacted[-1] != boundary.context[-1]:
        raise ValueError(
            f"{type(strategy).__name__} did not keep the boundary's last ToolResultsEvent; "
            "both arms must resume from the same trigger"
        )
    return CompactedContext(events=compacted, usage=tuple(usage.events))


class _UsageCollector:
    __slots__ = ("events",)

    def __init__(self) -> None:
        self.events: list[UsageEvent] = []

    async def add(self, event: UsageEvent) -> None:
        self.events.append(event)
