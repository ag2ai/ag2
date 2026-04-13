# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""UUID7 helpers for the network layer.

Every resource in the V3 network (actor, session, task, envelope, rule,
subscription, hub) is identified by a UUID7 stamped by the hub at the moment
of creation. UUID7 is time-ordered, so sorting by id matches creation order —
a property the WAL and inbox rely on.

Python 3.12's stdlib does not expose ``uuid7``, so we implement it here per
the draft RFC 9562 layout::

    | 48 bits unix_ts_ms | 4 bits ver (7) | 12 bits rand_a
    | 2 bits var (10)    | 62 bits rand_b

Within a single process we enforce strict monotonicity: if two calls land on
the same millisecond, the second id is derived by bumping ``rand_a``. This
guarantees id ordering never regresses even under rapid allocation.
"""

from __future__ import annotations

import os
import threading
import time
from uuid import UUID

_UUID7_LOCK = threading.Lock()
_LAST_MS: int = 0
_LAST_RAND_A: int = 0
_LAST_RAND_B: int = 0


def _rand_bits(n: int) -> int:
    byte_count = (n + 7) // 8
    raw = int.from_bytes(os.urandom(byte_count), "big")
    return raw & ((1 << n) - 1)


def uuid7() -> UUID:
    """Generate a monotonic UUID7."""

    global _LAST_MS, _LAST_RAND_A, _LAST_RAND_B

    with _UUID7_LOCK:
        now_ms = int(time.time() * 1000)

        if now_ms > _LAST_MS:
            _LAST_MS = now_ms
            # Reserve the top of rand_a so we have 4k counter room before
            # we need to roll ms forward under a same-ms burst.
            _LAST_RAND_A = _rand_bits(11)
            _LAST_RAND_B = _rand_bits(62)
        else:
            # Clock did not advance (or went backwards). Bump rand_a — the
            # highest non-version bits — to preserve strict integer
            # monotonicity. If rand_a saturates, roll ms forward and reseed
            # both random halves.
            if _LAST_RAND_A < 0xFFF:
                _LAST_RAND_A += 1
            else:
                _LAST_MS += 1
                _LAST_RAND_A = _rand_bits(11)
                _LAST_RAND_B = _rand_bits(62)

        ms = _LAST_MS
        rand_a = _LAST_RAND_A
        rand_b = _LAST_RAND_B

    version = 0x7
    variant = 0b10
    high = (ms & 0xFFFFFFFFFFFF) << 16 | (version << 12) | rand_a
    low = (variant << 62) | rand_b
    return UUID(int=(high << 64) | low)


def new_id() -> str:
    """Return a new UUID7 rendered as its canonical string form."""

    return str(uuid7())


def extract_ms(value: str | UUID) -> int:
    """Extract the millisecond timestamp embedded in a UUID7 id.

    Useful for debugging — the tests use this to assert time ordering.
    """

    uid = value if isinstance(value, UUID) else UUID(value)
    return (uid.int >> 80) & 0xFFFFFFFFFFFF
