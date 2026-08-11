# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``ViewPolicy`` Protocol — pure projection from WAL to ModelEvents.

Pure function. Called once per turn before the participant's LLM call.
Translates ``Envelope``s into ``BaseEvent``s (``ModelRequest`` for
inbound envelopes, ``ModelMessage`` for the participant's own past
turns) in chronological order. The current turn's ``ModelRequest`` is
appended by the caller.

The same ``project`` method is used for both historical WAL slices and
for the single current-turn envelope — the handler calls
``view.project([current_envelope], ...)`` to ensure consistent labeling
between history and the triggering message.

Built-ins: :class:`FullTranscript` and :class:`WindowedSummary`.
"""

from collections.abc import Callable
from typing import Protocol

from ag2.events import BaseEvent

from ..channel import ChannelMetadata
from ..envelope import Envelope

__all__ = ("EnvelopeRenderer", "NameResolver", "ViewPolicy", "default_name_resolver")


EnvelopeRenderer = Callable[[Envelope], "str | None"]
"""Project a single envelope to its LLM-visible string, or ``None``
to skip. Supplied by the channel's adapter via
``ChannelAdapter.render_envelope`` so view policies stay
adapter-neutral."""


NameResolver = Callable[[str], str]
"""Resolve an ``agent_id`` to its human-facing ``Passport.name``.

Supplied by the default handler (sourced from :meth:`Hub.name_for`) so
view policies can label projection lines without depending on hub
internals. The resolver is called once per visible envelope. If the id
is unknown the resolver returns the raw id so projection never fails."""


def default_name_resolver(agent_id: str) -> str:
    """Identity ``NameResolver`` — returns ``agent_id`` verbatim.

    Used by call sites that want to invoke a view policy without
    plumbing a real hub-backed resolver (tests, headless utilities).
    Production code paths (the default notify handler) always pass a
    real resolver backed by the hub's passport directory.
    """
    return agent_id


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
        channel: ChannelMetadata,
        render_envelope: EnvelopeRenderer,
        name_for: NameResolver = default_name_resolver,
    ) -> list[BaseEvent]:
        """Convert the WAL slice this participant should see into model
        events.

        ``render_envelope`` is provided by the channel's adapter so the
        view stays adapter-neutral. ``name_for`` resolves an
        ``agent_id`` to its human-facing ``Passport.name`` for views
        that want to label projection lines with the sender — built-in
        views that don't label simply ignore it.
        """
        ...
