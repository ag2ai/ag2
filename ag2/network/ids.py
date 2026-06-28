# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Identifier helpers for the network layer.

:func:`make_id` returns 32-char hex ids whose leading 8 bytes are the
wall-clock time, so they are *best-effort* time-ordered: unique and
roughly sortable, but two ids minted within one clock tick are ordered
only by their random suffix (``time.time_ns`` has coarse resolution on
some runtimes — notably Windows, at ~15 ms).

Where strict creation-order sorting is a correctness requirement — the
delivery cursor and reconnect replay compare envelope ids by sort order —
use :class:`_MonotonicIds`, which clamps each timestamp to strictly above
the previous one (the UUIDv7 monotonicity guarantee). The hub owns one
per instance and mints every envelope id through it.

Also exposes :func:`parse_hub_urn` — the canonical helper for splitting
``hub://<hub_id>/<agent_id>`` URNs into their parts. Non-URN inputs
pass through unchanged so callers can use it on any audience id
without branching on shape.
"""

import os
import threading
import time

__all__ = ("make_id", "parse_hub_urn")


_HUB_URN_PREFIX = "hub://"


def make_id() -> str:
    """Return a fresh, best-effort time-ordered 32-char hex string.

    The leading 8 bytes are the big-endian nanosecond wall clock; the
    trailing 8 bytes are random for uniqueness. Two ids minted in the same
    clock tick share the time prefix and so sort only by their random
    suffix — fine for ids that need only uniqueness (agent, channel,
    endpoint, request). Anything that must sort in *strict* creation order
    — envelope ids, which the delivery cursor and reconnect replay compare
    by sort order — must come from :class:`_MonotonicIds` instead.

    (``uuid.uuid7`` would give strict ordering but exists only on 3.14+,
    and ``uuid.uuid4`` is unordered, so neither is a portable basis.)
    """
    return time.time_ns().to_bytes(8, "big").hex() + os.urandom(8).hex()


class _MonotonicIds:
    """Callable returning strictly-monotonic, time-ordered ids.

    Each id has the same shape as :func:`make_id` — an 8-byte big-endian
    timestamp plus 8 random bytes — but consecutive calls are guaranteed
    to sort in call order even when the wall clock does not advance between
    them. Each timestamp is clamped to strictly above the previous one (the
    UUIDv7 monotonicity guarantee); that is what makes the result a safe
    basis for the delivery cursor and reconnect replay, which compare
    envelope ids by sort order. As a side benefit it is also robust to the
    wall clock stepping *backwards* (NTP slew), which a plain timestamp is
    not.

    State is per-instance, so each owner (e.g. a ``Hub``) gets an isolated
    sequence — there is no shared process-global counter, so separate
    owners never perturb one another's ordering and nothing leaks across
    them (or across tests). The lock keeps the read-modify-write atomic if
    an instance is ever called off the event loop (e.g. from a worker
    thread); the critical section is a few arithmetic ops and holds no
    ``await``, so it never stalls the loop. Envelope minting already runs
    under the hub's per-channel WAL lock on a single event loop, so this
    lock is defense-in-depth rather than a load-bearing guard.
    """

    def __init__(self) -> None:
        self._last_ns = 0
        self._lock = threading.Lock()

    def __call__(self) -> str:
        with self._lock:
            ns = time.time_ns()
            if ns <= self._last_ns:
                ns = self._last_ns + 1
            self._last_ns = ns
        return ns.to_bytes(8, "big").hex() + os.urandom(8).hex()


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
