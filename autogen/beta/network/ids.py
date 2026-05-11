# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Identifier helpers for the network layer.

Uses ``uuid.uuid7`` if the Python stdlib provides it (3.14+) for
chronologically sortable identifiers; falls back to ``uuid.uuid4`` on
older runtimes. Both produce 32-char hex strings, so the wire format is
stable across versions.
"""

import uuid

__all__ = ("make_id",)


def make_id() -> str:
    """Return a fresh UUID-based identifier as a 32-char hex string.

    Prefers UUID7 (time-ordered, cross-process sortable) when available;
    falls back to UUID4 on older Pythons. The in-process hub stamps every
    envelope under a per-session lock so process-local ordering is
    already serialised; the time-ordered prefix matters once the
    transport spans processes.
    """
    if hasattr(uuid, "uuid7"):
        return uuid.uuid7().hex  # type: ignore[attr-defined]
    return uuid.uuid4().hex
