# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Execution-regression burden: the arithmetic the verifier scores with.

The agent is itself unstable, so a post-compaction rollout merely *behaving
differently* from a pre-compaction one is not evidence that compaction hurt.
Only two directional regressions count, both read from real execution:

* **blocked** — an action whose execution failed;
* **refetch** — an action repeating a call already executed, either before the
  boundary or earlier in the same rollout.

Over the first ``k`` actions of a rollout, **wasted** counts the actions that
are blocked *or* refetch, each once.

Burden only counts actions taken, so an agent that stops wastes nothing — and
finishes nothing. A third, separate signal covers that: a rollout is
**stopped** at ``k`` when the agent ended its turn before executing ``k``
actions. Read it next to ``wasted``: a strategy that makes the agent abandon
the task can look harmless on burden alone. For one boundary, with PRE the rollouts
resumed from the raw history (the control) and POST the rollouts resumed from
the compacted history::

    delta_x(k) = mean over POST of x(k) - mean over PRE of x(k)   (x = blocked, refetch, wasted, stopped)
    harm(k)    = max(delta_blocked(k), 0) + max(delta_refetch(k), 0)

A positive delta means compaction added burden. ``harm`` clips the two channels
separately so that an improvement in one never cancels a regression in the
other; ``delta_wasted`` is the single objective, because a wasted step is what
a compaction strategy can actually reduce, whereas ``harm`` can be lowered by
trading an error for a refetch.

This module is pure arithmetic over :class:`~.actions.Action` sequences and
needs no agent, so recorded rollouts can be scored on their own.
"""

from collections.abc import Iterable, Sequence, Set
from dataclasses import dataclass
from typing import Literal

from .actions import Action

__all__ = (
    "ArmBurden",
    "BoundaryDelta",
    "Burden",
    "SignatureKey",
    "arm_burden",
    "burden",
    "history_signatures",
    "score_boundary",
)

SignatureKey = Literal["call", "tool"]
"""What makes two actions "the same" for refetch.

``"call"`` (the default) compares tool name and canonical arguments. ``"tool"``
compares the tool name alone — a coarser diagnostic, since legitimate
pagination reuses a tool with new arguments.
"""


@dataclass(frozen=True, slots=True)
class Burden:
    """Burden of one rollout over its first ``k`` actions."""

    blocked: int
    refetch: int
    wasted: int
    actions: int
    stopped: bool
    """The rollout ended before executing ``k`` actions."""


@dataclass(frozen=True, slots=True)
class ArmBurden:
    """Mean burden over the rollouts of one arm (PRE or POST) at one horizon."""

    blocked: float
    refetch: float
    wasted: float
    actions: float
    stopped: float
    """Share of the arm's rollouts that ended before ``k`` actions."""
    rollouts: int


@dataclass(frozen=True, slots=True)
class BoundaryDelta:
    """POST minus PRE at one boundary and one horizon ``k``."""

    horizon: int
    pre: ArmBurden
    post: ArmBurden

    @property
    def blocked(self) -> float:
        return self.post.blocked - self.pre.blocked

    @property
    def refetch(self) -> float:
        return self.post.refetch - self.pre.refetch

    @property
    def wasted(self) -> float:
        return self.post.wasted - self.pre.wasted

    @property
    def stopped(self) -> float:
        return self.post.stopped - self.pre.stopped

    @property
    def harm(self) -> float:
        """Separately clipped two-channel burden; never negative."""
        return max(self.blocked, 0.0) + max(self.refetch, 0.0)


def history_signatures(actions: Iterable[Action], key: SignatureKey = "call") -> frozenset[str]:
    """The signatures of ``actions`` — every call executed before a boundary.

    All prior calls are included, blocked ones too: a later action repeats
    history if that exact call was ever run. There is no whitelist and no
    "reasonable repeat" filter.
    """
    return frozenset(_key(a, key) for a in actions)


def burden(
    actions: Sequence[Action],
    history: Set[str],
    horizon: int,
    *,
    key: SignatureKey = "call",
) -> Burden:
    """Burden of one rollout over its first ``horizon`` actions.

    An action is a refetch when its signature is in ``history`` or was already
    executed earlier in this rollout, so a rollout that loops is charged for it.
    """
    blocked = refetch = wasted = 0
    seen: set[str] = set()
    window = actions[:horizon]
    for action in window:
        signature = _key(action, key)
        is_refetch = signature in history or signature in seen
        seen.add(signature)
        blocked += action.blocked
        refetch += is_refetch
        wasted += action.blocked or is_refetch
    return Burden(blocked=blocked, refetch=refetch, wasted=wasted, actions=len(window), stopped=len(window) < horizon)


def arm_burden(
    rollouts: Sequence[Sequence[Action]],
    history: Set[str],
    horizon: int,
    *,
    key: SignatureKey = "call",
) -> ArmBurden:
    """Mean burden of an arm's rollouts at one horizon."""
    if not rollouts:
        raise ValueError("an arm needs at least one rollout to be scored")
    scored = [burden(r, history, horizon, key=key) for r in rollouts]
    n = len(scored)
    return ArmBurden(
        blocked=sum(b.blocked for b in scored) / n,
        refetch=sum(b.refetch for b in scored) / n,
        wasted=sum(b.wasted for b in scored) / n,
        actions=sum(b.actions for b in scored) / n,
        stopped=sum(b.stopped for b in scored) / n,
        rollouts=n,
    )


def score_boundary(
    pre: Sequence[Sequence[Action]],
    post: Sequence[Sequence[Action]],
    history: Set[str],
    horizons: Iterable[int],
    *,
    key: SignatureKey = "call",
) -> tuple[BoundaryDelta, ...]:
    """POST minus PRE at one boundary, for every horizon.

    Both arms are scored against the same ``history``: whatever the context
    shows the agent, the environment executed the same calls before the
    boundary.
    """
    return tuple(
        BoundaryDelta(
            horizon=k,
            pre=arm_burden(pre, history, k, key=key),
            post=arm_burden(post, history, k, key=key),
        )
        for k in horizons
    )


def _key(action: Action, key: SignatureKey) -> str:
    return action.signature if key == "call" else action.name
