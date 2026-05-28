# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Identifier helpers for the network layer.

:func:`make_id` returns time-ordered 32-char hex ids so the delivery
cursor and reconnect replay can compare them by sort order identically
on every supported runtime.

Also exposes :func:`parse_hub_urn` — the canonical helper for splitting
``hub://<hub_id>/<agent_id>`` URNs into their parts. Non-URN inputs
pass through unchanged so callers can use it on any audience id
without branching on shape.
"""

import os
import time

__all__ = ("make_id", "parse_hub_urn")


_HUB_URN_PREFIX = "hub://"


def make_id() -> str:
    """Return a fresh time-ordered identifier as a 32-char hex string.

    The leading 8 bytes are the big-endian nanosecond wall clock, so the
    hex sorts lexicographically in creation order; the trailing 8 bytes
    are random for uniqueness. The delivery cursor and reconnect replay
    compare ids by sort order, so this ordering has to hold on every
    supported runtime — ``uuid.uuid7`` exists only on 3.14+ and
    ``uuid.uuid4`` is unordered, so neither is a portable basis.
    """
    return time.time_ns().to_bytes(8, "big").hex() + os.urandom(8).hex()


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
    rest = s[len(_HUB_URN_PREFIX) :]
    hub_id, sep, agent_id = rest.partition("/")
    if not sep or not hub_id or not agent_id:
        return None, s
    return hub_id, agent_id
