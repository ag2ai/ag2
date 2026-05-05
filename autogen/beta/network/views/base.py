# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``ViewPolicy`` Protocol — pure projection from WAL to ModelEvents.

Pure function. Called once per turn before the participant's LLM call.
Translates ``Envelope``s into ``BaseEvent``s (``ModelRequest`` for
inbound envelopes, ``ModelMessage`` for the participant's own past
turns) in chronological order. The current turn's ``ModelRequest`` is
appended by the caller.

Built-ins: :class:`FullTranscript` and :class:`WindowedSummary`.
"""

from typing import Protocol

from autogen.beta.events import BaseEvent

from ..envelope import Envelope
from ..session import SessionMetadata

__all__ = ("ViewPolicy",)


class ViewPolicy(Protocol):
    """Per-participant projection.

    Implementations must be deterministic functions of the input WAL
    slice — calling ``project`` twice with the same input must produce
    the same events.
    """

    name: str

    async def project(
        self,
        wal: list[Envelope],
        *,
        participant_id: str,
        session: SessionMetadata,
    ) -> list[BaseEvent]:
        """Convert the WAL slice this participant should see into model events."""
        ...
