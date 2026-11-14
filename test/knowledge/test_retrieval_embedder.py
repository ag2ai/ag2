# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the embedder layer: HashingEmbedder determinism, CallableEmbedder
batching, and the cosine helper."""

import pytest

from ag2.knowledge.retrieval import CallableEmbedder, HashingEmbedder
from ag2.knowledge.retrieval._vector import cosine_similarity


@pytest.mark.asyncio
class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_opposite_vectors_score_minus_one(self) -> None:
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_scores_zero(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same dimension"):
            cosine_similarity([1.0], [1.0, 2.0])

    def test_scale_invariance(self) -> None:
        assert cosine_similarity([1.0, 2.0], [10.0, 20.0]) == pytest.approx(1.0)


@pytest.mark.asyncio
class TestHashingEmbedder:
    async def test_same_text_always_yields_the_same_vector(self) -> None:
        embedder = HashingEmbedder()

        first = await embedder.embed(["deploy the service at noon"])
        second = await embedder.embed(["deploy the service at noon"])

        assert first[0] == second[0]

    async def test_one_vector_per_input_in_order(self) -> None:
        embedder = HashingEmbedder()

        vectors = await embedder.embed(["alpha", "beta gamma", "alpha"])

        assert len(vectors) == 3
        assert vectors[0] == vectors[2]
        assert vectors[1] != vectors[0]

    async def test_vectors_are_l2_normalized(self) -> None:
        embedder = HashingEmbedder()

        [vector] = await embedder.embed(["a longer piece of text with several tokens"])

        assert sum(value * value for value in vector) == pytest.approx(1.0)
        assert len(vector) == embedder.dimension

    async def test_related_text_scores_higher_than_unrelated(self) -> None:
        embedder = HashingEmbedder()

        vectors = await embedder.embed(["the user likes tea", "the user likes tea a lot", "quarterly revenue fell"])

        same = cosine_similarity(vectors[0], vectors[1])
        other = cosine_similarity(vectors[0], vectors[2])
        assert same > other

    async def test_empty_text_yields_zero_vector(self) -> None:
        embedder = HashingEmbedder()

        [vector] = await embedder.embed([""])

        assert vector == [0.0] * embedder.dimension

    async def test_embeddings_are_stable_across_instances_and_processes(self) -> None:
        # A fresh instance must reproduce the same vector: persisted indexes
        # rely on this across processes.
        first = await HashingEmbedder(dimension=64, seed=7).embed(["stable across runs"])
        second = await HashingEmbedder(dimension=64, seed=7).embed(["stable across runs"])

        assert first[0] == second[0]

    async def test_different_seed_shuffles_buckets(self) -> None:
        first = await HashingEmbedder(seed=1).embed(["some text"])
        second = await HashingEmbedder(seed=2).embed(["some text"])

        assert first[0] != second[0]

    def test_name_records_the_configuration(self) -> None:
        name = HashingEmbedder(dimension=64, seed=3).name

        assert "64" in name
        assert "seed=3" in name


class TestHashingEmbedderValidation:
    def test_rejects_non_positive_dimension(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            HashingEmbedder(dimension=0)

    def test_rejects_empty_ngram(self) -> None:
        with pytest.raises(ValueError, match="ngram"):
            HashingEmbedder(ngram=())


@pytest.mark.asyncio
class TestCallableEmbedder:
    async def test_wraps_a_sync_function(self) -> None:
        def embed(texts) -> list[list[float]]:
            return [[float(len(text))] for text in texts]

        embedder = CallableEmbedder(embed, dimension=1, name="sync-fake")

        assert await embedder.embed(["ab", "cdef"]) == [[2.0], [4.0]]
        assert embedder.dimension == 1
        assert embedder.name == "sync-fake"

    async def test_wraps_an_async_function(self) -> None:
        async def embed(texts) -> list[list[float]]:
            return [[1.0] for _ in texts]

        embedder = CallableEmbedder(embed, dimension=1, name="async-fake")

        assert await embedder.embed(["x", "y"]) == [[1.0], [1.0]]

    async def test_batches_calls_by_batch_size(self) -> None:
        calls: list[int] = []

        def embed(texts) -> list[list[float]]:
            calls.append(len(texts))
            return [[0.0] for _ in texts]

        embedder = CallableEmbedder(embed, dimension=1, name="batched", batch_size=2)

        vectors = await embedder.embed(["a", "b", "c", "d", "e"])

        assert calls == [2, 2, 1]
        assert len(vectors) == 5

    def test_rejects_non_positive_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            CallableEmbedder(lambda texts: [], dimension=1, name="x", batch_size=0)
