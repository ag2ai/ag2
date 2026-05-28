# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Identifier helpers for the network layer.

Uses ``uuid.uuid7`` if the Python stdlib provides it (3.14+) for
chronologically sortable identifiers; falls back to ``uuid.uuid4`` on
older runtimes. Both produce 32-char hex strings, so the wire format is
stable across versions.

Also exposes :func:`parse_hub_urn` — the canonical helper for splitting
``hub://<hub_id>/<agent_id>`` URNs into their parts. Non-URN inputs
pass through unchanged so callers can use it on any audience id
without branching on shape.
"""

import uuid

__all__ = ("make_id", "parse_hub_urn")


_HUB_URN_PREFIX = "hub://"


def make_id() -> str:
    """Return a fresh UUID-based identifier as a 32-char hex string.

    Prefers UUID7 (time-ordered, cross-process sortable) when available;
    falls back to UUID4 on older Pythons. The in-process hub stamps every
    envelope under a per-channel lock so process-local ordering is
    already serialised; the time-ordered prefix matters once the
    transport spans processes.
    """
    if hasattr(uuid, "uuid7"):
        return uuid.uuid7().hex  # type: ignore[attr-defined]
    return uuid.uuid4().hex


def parse_hub_urn(s: str) -> tuple[str | None, str]:
    """Split a ``hub://<hub_id>/<agent_id>`` URN into its parts.

    Returns ``(hub_id, agent_id)`` for valid URNs and ``(None, s)`` for
    any other input — including malformed URNs (missing slash, empty
    hub_id, empty agent_id). Idempotent: callers that pass the
    ``agent_id`` half back in get the same value out.

    The canonical inverse is ``f"hub://{hub_id}/{agent_id}"`` — there
    is no helper for that direction because the format is trivial and
    keeping it inline at call sites makes intent obvious.
    """
    if not isinstance(s, str) or not s.startswith(_HUB_URN_PREFIX):
        return None, s
    rest = s[len(_HUB_URN_PREFIX):]
    hub_id, sep, agent_id = rest.partition("/")
    if not sep or not hub_id or not agent_id:
        return None, s
    return hub_id, agent_id
