# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A control strategy for measuring the verifier's own noise floor."""

from ag2.annotations import Context
from ag2.events import BaseEvent
from ag2.knowledge import KnowledgeStore

__all__ = ("IdentityCompact",)


class IdentityCompact:
    """A :class:`~ag2.compact.CompactStrategy` that returns the history unchanged.

    Verifying it is an A/A test: PRE and POST see the same context, so any
    delta it scores is the agent's run-to-run variance at the chosen number of
    samples. Run it beside the strategies under test; a strategy's delta is
    only evidence when it clears this one.
    """

    async def compact(
        self,
        events: list[BaseEvent],
        context: Context,
        store: KnowledgeStore | None,
    ) -> list[BaseEvent]:
        return list(events)
