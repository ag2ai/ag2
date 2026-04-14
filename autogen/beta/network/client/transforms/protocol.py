# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Transform protocol, context, and error types — Phase 5a.1.

The :class:`Transform` protocol is deliberately tiny: one async
``__call__`` that takes an :class:`Envelope` plus a
:class:`TransformContext` and returns either a (possibly mutated)
envelope or ``None`` to reject. Every form (named / python / http)
satisfies it structurally.

:class:`TransformContext` is the stable surface that user-written
named transforms build against; its field layout is frozen in Phase
5a.1 so 5b's sidecar adapters and every long-lived tenant transform
continue to work without rewrites. Additive fields in later phases
are allowed; removing or renaming is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from ...envelope import Envelope

if TYPE_CHECKING:
    from ..actor_client import ActorClient

# TransformStage lives in ``rule.py`` because it is a rule-schema
# concept (what the JSON documents) rather than a client-runtime
# concept — keeping it next to ``TransformSpec`` lets ``Rule.from_dict``
# validate the field without importing this module.
from ...rule import TransformStage

__all__ = (
    "Transform",
    "TransformContext",
    "TransformError",
    "TransformLookupError",
    "TransformRejected",
)


Direction = Literal["inbound", "outbound"]


@dataclass(slots=True)
class TransformContext:
    """Per-invocation metadata passed to every :class:`Transform`.

    Field layout is frozen in Phase 5a.1. Later phases may add fields
    (they are additive); nothing is removed or renamed.

    - ``stage`` — which of the four pipeline stages this call belongs to.
    - ``client`` — the :class:`ActorClient` hosting the pipeline.
      Transforms can read ``client.actor`` / ``client.identity`` /
      ``client.actor_id`` but must not mutate the rule; rule edits go
      through the hub's ``PUT /v1/actors/{id}/rule`` or ``set_rule``
      paths so the change flows through ``RuleChangedFrame``.
    - ``session_id`` — the session the envelope belongs to. ``None``
      is reserved for future session-less envelopes.
    - ``rule_version`` — the ``Rule.version`` of the pipeline this
      transform is running in. A named transform that caches state
      across envelopes can compare against this to decide when to
      reset.
    - ``direction`` — ``"outbound"`` for ``pre_send`` / ``post_send``,
      ``"inbound"`` for ``pre_receive`` / ``post_receive``. Derived
      from ``stage`` as syntactic sugar for filters.
    """

    stage: TransformStage
    client: ActorClient
    session_id: str | None
    rule_version: int
    direction: Direction


@runtime_checkable
class Transform(Protocol):
    """Async callable invoked by :class:`TransformPipeline` on every envelope.

    Implementations should not raise for business-logic rejections —
    they should return ``None``. Raising is reserved for unexpected
    errors (and the pipeline treats them as reject + log, matching
    §4.1's fail-fast stance).

    Stateful transforms (rate limiters, circuit breakers) hold their
    state on the instance; the pipeline keeps the same instance alive
    for as long as the rule version stays constant, and drops it on
    the next :class:`RuleChangedFrame` rebuild.
    """

    async def __call__(
        self, envelope: Envelope, ctx: TransformContext
    ) -> Envelope | None: ...


class TransformError(Exception):
    """Base class for every transform-layer failure."""


class TransformLookupError(TransformError):
    """A :class:`NamedTransform` referenced an unknown registry entry.

    Raised at pipeline-build time (not per-envelope), so a rule with a
    dangling ``apply: "not_registered"`` is caught as soon as the
    :class:`RuleChangedFrame` arrives.
    """


class TransformRejected(TransformError):
    """Raised by the pipeline when a ``pre_send`` transform rejects.

    ``pre_send`` rejections short-circuit before the link frame is
    emitted, so the local ``Session.send`` raises this exception for
    the caller to handle. ``pre_receive`` rejections produce a nack on
    the inbound path and never propagate out of the pipeline.
    """

    def __init__(self, reason: str, *, stage: TransformStage) -> None:
        super().__init__(f"transform rejected at {stage.value}: {reason}")
        self.reason = reason
        self.stage = stage
