# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Memory-injection observability — what retrieval put into the prompt.

:class:`MemoryInjected` is the record :class:`~ag2.policies.RetrievalMemoryPolicy`
publishes in transparent mode: one event per assembly that injected anything,
carrying every injected item with its store path, retrieval score, and kind.
Evaluation reads it — the memory scorers in ``ag2.eval.scorers.memory`` parse
these events out of a :class:`~ag2.eval.Trace` — and so can any subscriber that
wants to see what the agent was reminded of.

Like the other lifecycle events it is ``__transient__``: published to the
stream's subscribers, never part of history, the model input, or the token
budget.
"""

from dataclasses import dataclass
from typing import Literal

from .base import BaseEvent, Field

__all__ = ("MemoryInjected", "MemoryItem")


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """One injected memory, attributed to its source path in the store."""

    path: str
    """Store path the memory came from — a fact record or an episode summary."""

    score: float
    """Raw similarity of the chunk to the query that retrieved it."""

    kind: Literal["fact", "episode"]
    """Whether the item is a fact record or a conversation summary."""


class MemoryInjected(BaseEvent):
    """Announces the memory block an assembly policy injected for one model call."""

    __transient__ = True

    items: list[MemoryItem] = Field(default_factory=list)
    """What was injected, best first — the order the block presented them."""

    query: str = ""
    """The user input the memories were retrieved for."""
