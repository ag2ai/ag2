# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Cosine similarity for dense vectors, in pure Python.

``numpy`` is not a core ``ag2`` dependency (see ``pyproject.toml``), so there is
no accelerated path: a single fused dot/norm pass over lists of floats is fast
enough for the index sizes a knowledge store holds, and staying dependency-free
keeps retrieval importable anywhere ``ag2`` is.
"""

from collections.abc import Sequence

__all__ = ("cosine_similarity",)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between two equal-length vectors.

    Returns ``0.0`` when either vector is all zeros — an unnormalized or empty
    embedding is uninformative, and a zero vector has no direction to compare.
    Raises ``ValueError`` when the dimensions differ, which signals an index
    mixing vector spaces (the condition ADR 0036's header check prevents).
    """
    if len(a) != len(b):
        raise ValueError(f"vectors must have the same dimension, got {len(a)} and {len(b)}")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a**0.5 * norm_b**0.5)
