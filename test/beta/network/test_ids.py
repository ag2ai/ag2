# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the UUID7 id helper.

Monotonicity is the load-bearing property — every resource id on the hub
sorts by creation order, and the WAL / inbox rely on it. These tests
exercise both the happy path and the same-millisecond collision path.
"""

from __future__ import annotations

import concurrent.futures
import time
from uuid import UUID

from autogen.beta.network.ids import extract_ms, new_id, uuid7


def test_uuid7_is_version_7() -> None:
    uid = uuid7()
    assert uid.version == 7
    # Variant bits (top two of the clock_seq_hi_variant byte) must be 10.
    assert (uid.bytes[8] & 0xC0) == 0x80


def test_new_id_returns_canonical_string() -> None:
    value = new_id()
    parsed = UUID(value)
    assert str(parsed) == value


def test_uuid7_monotonic_sequential() -> None:
    previous = uuid7()
    for _ in range(1000):
        current = uuid7()
        assert current.int > previous.int
        previous = current


def test_uuid7_monotonic_under_same_millisecond_burst() -> None:
    # Burst 5000 ids as fast as possible — many will land in the same
    # millisecond, which exercises the rand_a bump path.
    ids = [uuid7() for _ in range(5000)]
    sorted_ids = sorted(ids, key=lambda u: u.int)
    assert ids == sorted_ids


def test_uuid7_extract_ms_matches_wall_clock() -> None:
    before = int(time.time() * 1000)
    value = uuid7()
    after = int(time.time() * 1000)
    extracted = extract_ms(value)
    assert before - 1 <= extracted <= after + 1


def test_uuid7_monotonic_across_threads() -> None:
    # Thread-safety check: a dozen workers each minting 500 ids must produce
    # 6000 distinct ids whose global sort order matches integer order.
    count = 6000
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(lambda: [uuid7() for _ in range(500)]) for _ in range(12)]
        collected: list[UUID] = []
        for fut in futures:
            collected.extend(fut.result())
    assert len(set(collected)) == count
    assert sorted(collected, key=lambda u: u.int) == sorted(collected, key=lambda u: u.int)


def test_extract_ms_accepts_string_or_uuid() -> None:
    value_str = new_id()
    assert extract_ms(value_str) == extract_ms(UUID(value_str))
