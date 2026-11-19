# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Embedders — turn text into the fixed-length vectors the memory index ranks.

An :class:`Embedder` is the only provider-specific piece of retrieval, and the
protocol is deliberately tiny — one async method plus two identity fields — so
any embedding backend wraps in a few lines via :class:`CallableEmbedder. No
provider adapter ships here: retrieval stays dependency-free, and applications
bring their own embedding call.

``name`` and ``dimension`` are stored in a persisted index's header. Loading an
index written by a different embedder — or a reconfigured one — must never mix
vector spaces, so the loader treats the mismatch as a rebuild signal rather
than serving incomparable scores (ADR 0036).
"""

import hashlib
import inspect
import math
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Protocol, runtime_checkable

__all__ = ("CallableEmbedder", "Embedder", "HashingEmbedder")


@runtime_checkable
class Embedder(Protocol):
    """Text to fixed-length vectors, asynchronously and in input order."""

    dimension: int
    """Length of every vector this embedder produces; pinned in the index header."""

    name: str
    """Identity of the embedding function; pinned in the index header."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts``, returning exactly one vector per input, in order."""
        ...


_TOKEN = re.compile(r"\w+")


class HashingEmbedder:
    """Deterministic, dependency-free feature-hashing embedder.

    Tokens and adjacent token n-grams (1-grams and 2-grams by default) are
    hashed into ``dimension`` buckets with the sign trick, and each vector is
    L2-normalized, so cosine similarity reduces to comparing bag-of-features
    overlap. Identical text always yields the identical vector across runs and
    platforms (hashing goes through ``hashlib.blake2b``, never Python's salted
    ``hash()``), which is what tests and the offline eval path need.

    It is not a semantic embedder: paraphrases share only their literal words.
    Use it for tests, dedup-shaped workloads, and as the offline default; wire
    a real model through :class:`CallableEmbedder` for meaning-level retrieval.

    Args:
        dimension: Bucket count — the vector length.
        ngram: Token n-gram sizes to hash. ``(1, 2)`` hashes every token and
            every adjacent token pair.
        seed: Salt for the feature hash, so a different seed re-shuffles buckets
            without changing the ordering structure.
    """

    def __init__(self, dimension: int = 256, ngram: tuple[int, ...] = (1, 2), seed: int = 0) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if not ngram or any(n <= 0 for n in ngram):
            raise ValueError(f"ngram must be non-empty positive sizes, got {ngram}")
        self.dimension = dimension
        self.name = f"hashing(dimension={dimension},ngram={','.join(str(n) for n in sorted(ngram))},seed={seed})"
        self._ngram = tuple(sorted(ngram))
        self._seed = seed

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _features(self, text: str) -> Iterator[str]:
        tokens = _TOKEN.findall(text.casefold())
        for n in self._ngram:
            for start in range(len(tokens) - n + 1):
                yield " ".join(tokens[start : start + n])

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for feature in self._features(text):
            bucket, sign = _feature_bucket(feature, self.dimension, self._seed)
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


def _feature_bucket(feature: str, dimension: int, seed: int) -> tuple[int, float]:
    """Hash one feature to its ``(bucket, sign)`` — stable across processes.

    ``hashlib.blake2b`` instead of ``hash()``: the builtin is salted per
    process for strings, which would make the same text embed differently in
    two runs and silently break every persisted index.
    """
    digest = hashlib.blake2b(f"{seed}:{feature}".encode(), digest_size=16).digest()
    bucket = int.from_bytes(digest[:8], "big") % dimension
    sign = 1.0 if digest[8] & 1 else -1.0
    return bucket, sign


EmbedFn = Callable[[Sequence[str]], Any]
"""A user-supplied embedding function: sync, or async (returning an awaitable)."""


class CallableEmbedder:
    """Wrap any embedding callable as an :class:`Embedder`.

    Accepts a sync or an ``async def`` function mapping a sequence of texts to
    the same number of vectors. Sync functions run inline, so a blocking
    client should be wrapped in an ``async def`` by the caller::

        async def embed(texts):
            return await openai_client.embeddings.create(input=texts, ...)

        embedder = CallableEmbedder(embed, dimension=1536, name="openai/text-embedding-3-small")

    Inputs are split into batches of ``batch_size`` and one call is made per
    batch, so a provider's request-size limit bounds the callable's input.

    Args:
        fn: The embedding function. Called with a sequence of strings; must
            return one vector per input, in order.
        dimension: Length of the vectors ``fn`` returns.
        name: Identity pinned in the index header.
        batch_size: Maximum texts per ``fn`` call (must be positive).
    """

    def __init__(self, fn: EmbedFn, dimension: int, name: str, batch_size: int = 32) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._fn = fn
        self._batch_size = batch_size
        self.dimension = dimension
        self.name = name

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            result = self._fn(batch)
            if inspect.isawaitable(result):
                result = await result
            vectors.extend(result)
        return vectors
