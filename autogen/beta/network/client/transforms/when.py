# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Minimum viable ``when`` matcher for transform gating — Phase 5a.1.

Two keys only:

- ``event: <event_type>`` — exact match against ``envelope.event_type``.
  E.g. ``"ag2.msg.text"`` or ``"ag2.session.invite"``.
- ``session_type: <type>`` — exact match against the session's type.
  Looked up from the ``ActorClient``'s session-type cache; returns
  ``False`` (the transform skips) if the session id is unknown.

Unknown keys are ignored (forward-compatible) rather than raising,
because 5b and later phases may add more keys. A matcher that returns
``True`` is a license to run the transform; ``False`` skips it.

Complex filters (regex, combinators) deliberately do not ship here —
a transform that needs anything more expressive writes a
:class:`PythonTransform` with its own filtering logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...envelope import Envelope

if TYPE_CHECKING:
    from ..actor_client import ActorClient

__all__ = ("when_matches",)


def when_matches(
    when: dict[str, Any], envelope: Envelope, client: ActorClient
) -> bool:
    """Return True if the envelope satisfies every declared filter key.

    Empty ``when`` always matches. The matcher is AND across all keys;
    OR semantics are not shipped in 5a.1.
    """

    if not when:
        return True

    event_pred = when.get("event")
    if event_pred is not None and envelope.event_type != event_pred:
        return False

    session_type_pred = when.get("session_type")
    if session_type_pred is not None:
        resolved = client._session_type_for(envelope.session_id)
        if resolved is None or resolved != session_type_pred:
            return False

    return True
