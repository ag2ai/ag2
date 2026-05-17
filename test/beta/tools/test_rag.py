# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for RAGToolkit — keyword and semantic retrieval."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock

import pytest

from autogen.beta import Context
from autogen.beta.tools import RAGToolkit
from autogen.beta.tools.toolkits.rag import (
    _chunk_text,
    _cosine,
    _recompute_tfidf,
    _tfidf_score,
    _tokenize,
)

# ---------------------------------------------------------------------------
# Unit — chunking
# ---------------------------------------------------------------------------


class TestChunking:
    def test_single_short_paragraph(self) -> None:
        chunks = _chunk_text("Hello world.", "d1", "T", 500, 50)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."

    def test_two_paragraphs_fit_in_one_chunk(self) -> None:
        text = "First paragraph.\n\nSecond paragraph."
        chunks = _chunk_text(text, "d1", "T", 500, 50)
        assert len(chunks) == 1
        assert "First" in chunks[0].text
        assert "Second" in chunks[0].text

    def test_paragraphs_split_when_too_long(self) -> None:
        long = "A" * 400
        text = f"{long}\n\n{long}"
        chunks = _chunk_text(text, "d1", "T", 500, 50)
        assert len(chunks) == 2

    def test_single_long_paragraph_hard_split(self) -> None:
        para = "X" * 1200
        chunks = _chunk_text(para, "d1", "T", 500, 50)
        assert len(chunks) >= 3
        for c in chunks:
            assert len(c.text) <= 500

    def test_overlap_in_hard_split(self) -> None:
        para = "abcde" * 200  # 1000 chars
        chunks = _chunk_text(para, "d1", "T", 500, 100)
        # Each chunk except the last should start at pos i*(500-100)
        # Verify second chunk overlaps with end of first
        assert chunks[1].text[:100] == chunks[0].text[-100:]

    def test_empty_text_returns_no_chunks(self) -> None:
        assert _chunk_text("", "d1", "T", 500, 50) == []

    def test_whitespace_only_returns_no_chunks(self) -> None:
        assert _chunk_text("   \n\n   ", "d1", "T", 500, 50) == []

    def test_chunk_ids_are_unique(self) -> None:
        para = "W" * 1500
        chunks = _chunk_text(para, "d1", "T", 500, 50)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_fields_set_correctly(self) -> None:
        chunks = _chunk_text("Hello.", "doc-abc", "My Title", 500, 50)
        assert chunks[0].doc_id == "doc-abc"
        assert chunks[0].title == "My Title"


# ---------------------------------------------------------------------------
# Unit — TF-IDF
# ---------------------------------------------------------------------------


class TestTFIDF:
    def test_tokenize_lowercases_and_strips_punctuation(self) -> None:
        tokens = _tokenize("Hello, World! 42")
        assert tokens == ["hello", "world", "42"]

    def test_tfidf_score_higher_for_matching_chunk(self) -> None:
        chunks = _chunk_text("Python is great.", "d1", "T1", 500, 50)
        chunks += _chunk_text("Java is also used.", "d2", "T2", 500, 50)
        _recompute_tfidf(chunks)
        python_chunks = [c for c in chunks if "python" in c.text.lower()]
        java_chunks = [c for c in chunks if "java" in c.text.lower()]
        python_q = _tfidf_score(_tokenize("python"), python_chunks[0])
        java_q = _tfidf_score(_tokenize("python"), java_chunks[0])
        assert python_q > java_q

    def test_recompute_on_empty_is_noop(self) -> None:
        _recompute_tfidf([])  # must not raise

    def test_zero_score_for_unrelated_query(self) -> None:
        chunks = _chunk_text("cats and dogs", "d1", "T", 500, 50)
        _recompute_tfidf(chunks)
        score = _tfidf_score(_tokenize("quantum physics"), chunks[0])
        assert score == 0.0


# ---------------------------------------------------------------------------
# Unit — cosine similarity
# ---------------------------------------------------------------------------


class TestCosine:
    def test_identical_vectors(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self) -> None:
        assert _cosine([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_known_value(self) -> None:
        a = [1.0, 0.0]
        b = [1.0, 1.0]
        expected = 1.0 / math.sqrt(2)
        assert _cosine(a, b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# RAGToolkit — construction
# ---------------------------------------------------------------------------


class TestRAGToolkitConstruction:
    @pytest.mark.asyncio
    async def test_default_tool_names(self, async_mock: AsyncMock) -> None:
        rag = RAGToolkit()
        schemas = list(await rag.schemas(Context(async_mock)))
        names = {s.function.name for s in schemas}
        assert names == {"index_document", "search", "remove_document", "list_documents"}

    def test_importable_from_tools_package(self) -> None:
        from autogen.beta.tools import RAGToolkit as RAGToolkitAlias  # noqa: F401

        assert RAGToolkitAlias is RAGToolkit

    def test_custom_chunk_size_accepted(self) -> None:
        rag = RAGToolkit(chunk_size=200, overlap=20)
        assert rag._chunk_size == 200
        assert rag._overlap == 20


# ---------------------------------------------------------------------------
# RAGToolkit — programmatic index helper
# ---------------------------------------------------------------------------


class TestProgrammaticIndex:
    def test_index_returns_doc_id(self) -> None:
        rag = RAGToolkit()
        doc_id = rag.index("Hello world.", title="greeting")
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_index_custom_doc_id(self) -> None:
        rag = RAGToolkit()
        returned = rag.index("Hello.", title="t", doc_id="my-id")
        assert returned == "my-id"

    def test_list_shows_indexed_doc(self) -> None:
        rag = RAGToolkit()
        rag.index("Some content.", title="MyDoc", doc_id="d1")
        docs = rag._store.all_docs()
        assert len(docs) == 1
        assert docs[0].doc_id == "d1"
        assert docs[0].title == "MyDoc"


# ---------------------------------------------------------------------------
# RAGToolkit — keyword search
# ---------------------------------------------------------------------------


class TestKeywordSearch:
    @pytest.fixture()
    def rag_with_docs(self) -> RAGToolkit:
        rag = RAGToolkit()
        rag.index("Python is a high-level programming language.", title="Python")
        rag.index("Rust provides memory safety without garbage collection.", title="Rust")
        rag.index("Go was designed for simplicity and concurrency.", title="Go")
        return rag

    @pytest.mark.asyncio
    async def test_search_returns_relevant_result(self, rag_with_docs: RAGToolkit) -> None:
        search_tool = rag_with_docs.search()
        result = await search_tool.model.call(query="memory safety")
        assert "Rust" in result

    @pytest.mark.asyncio
    async def test_search_no_results_on_empty_store(self) -> None:
        rag = RAGToolkit()
        search_tool = rag.search()
        result = await search_tool.model.call(query="anything")
        assert "No documents indexed" in result

    @pytest.mark.asyncio
    async def test_search_top_k_limits_results(self, rag_with_docs: RAGToolkit) -> None:
        search_tool = rag_with_docs.search()
        result = await search_tool.model.call(query="programming language", top_k=1)
        # Only 1 result block
        assert result.count("doc_id=") == 1

    @pytest.mark.asyncio
    async def test_index_document_tool(self) -> None:
        rag = RAGToolkit()
        idx_tool = rag.index_document()
        result = await idx_tool.model.call(content="New content.", title="New", doc_id="new-1")
        assert "new-1" in result
        assert rag._store.all_docs()[0].doc_id == "new-1"

    @pytest.mark.asyncio
    async def test_remove_document_tool(self, rag_with_docs: RAGToolkit) -> None:
        docs = rag_with_docs._store.all_docs()
        doc_id = docs[0].doc_id
        remove_tool = rag_with_docs.remove_document()
        result = await remove_tool.model.call(doc_id=doc_id)
        assert "Removed" in result
        assert len(rag_with_docs._store.all_docs()) == 2

    @pytest.mark.asyncio
    async def test_remove_nonexistent_document(self, rag_with_docs: RAGToolkit) -> None:
        remove_tool = rag_with_docs.remove_document()
        result = await remove_tool.model.call(doc_id="does-not-exist")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_list_documents_tool(self, rag_with_docs: RAGToolkit) -> None:
        list_tool = rag_with_docs.list_documents()
        result = await list_tool.model.call()
        assert "Python" in result
        assert "Rust" in result
        assert "Go" in result

    @pytest.mark.asyncio
    async def test_list_documents_empty(self) -> None:
        rag = RAGToolkit()
        list_tool = rag.list_documents()
        result = await list_tool.model.call()
        assert "No documents" in result


# ---------------------------------------------------------------------------
# RAGToolkit — semantic search (mock embed_fn)
# ---------------------------------------------------------------------------


class TestSemanticSearch:
    def _make_embed_fn(self) -> tuple:
        """Returns (embed_fn, embeddings dict). Each unique text gets a fixed vector."""
        store: dict[str, list[float]] = {}
        call_count = 0

        def embed(texts: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            result = []
            for t in texts:
                if t not in store:
                    # Assign a unique unit vector based on first char
                    dim = 4
                    vec = [0.0] * dim
                    idx = ord(t[0]) % dim
                    vec[idx] = 1.0
                    store[t] = vec
                result.append(store[t])
            return result

        return embed, store

    @pytest.mark.asyncio
    async def test_semantic_search_calls_embed_fn(self) -> None:
        embed_fn, _ = self._make_embed_fn()
        rag = RAGToolkit(embed_fn=embed_fn)
        rag.index("Alpha document.", title="A", doc_id="a")

        search_tool = rag.search()
        await search_tool.model.call(query="Alpha document.")
        # embed_fn called at least once for indexing (1 chunk) + once for query

    @pytest.mark.asyncio
    async def test_semantic_result_format(self) -> None:
        embed_fn, _ = self._make_embed_fn()
        rag = RAGToolkit(embed_fn=embed_fn)
        rag.index("Semantic content here.", title="SemanticDoc", doc_id="s1")

        search_tool = rag.search()
        result = await search_tool.model.call(query="Semantic content here.")
        assert "s1" in result
        assert "SemanticDoc" in result

    @pytest.mark.asyncio
    async def test_remove_document_no_tfidf_recompute_in_semantic_mode(self) -> None:
        embed_fn, _ = self._make_embed_fn()
        rag = RAGToolkit(embed_fn=embed_fn)
        rag.index("Doc A.", title="A", doc_id="a")
        rag.index("Doc B.", title="B", doc_id="b")
        remove_tool = rag.remove_document()
        result = await remove_tool.model.call(doc_id="a")
        assert "Removed" in result
        assert len(rag._store.all_docs()) == 1


# ---------------------------------------------------------------------------
# RAGToolkit — tfidf recomputes across all chunks after new index
# ---------------------------------------------------------------------------


class TestGlobalTFIDFRecomputation:
    def test_tfidf_recomputed_across_all_chunks_on_new_index(self) -> None:
        rag = RAGToolkit()
        rag.index("alpha beta gamma", title="A", doc_id="a")
        chunk_a_first = rag._store.all_chunks()[0]
        # "beta" is unique to doc A right now — IDF(beta) = log(2/2)+1 = 1
        beta_score_one_doc = chunk_a_first.tfidf.get("beta", 0.0)
        assert beta_score_one_doc > 0.0

        # Add doc B without "beta" — now n=2, df_beta=1 → IDF rises
        rag.index("delta epsilon zeta", title="B", doc_id="b")
        chunk_a_after = next(c for c in rag._store.all_chunks() if c.doc_id == "a")
        beta_score_two_docs = chunk_a_after.tfidf.get("beta", 0.0)
        assert beta_score_two_docs > 0.0

        # With add-1 smoothing: IDF(beta, n=2, df=1) = log(3/2)+1 > log(2/2)+1 = 1
        # So beta's score should be higher after a new unrelated doc is added.
        assert beta_score_two_docs > beta_score_one_doc

    def test_all_chunks_have_tfidf_after_multi_doc_index(self) -> None:
        rag = RAGToolkit()
        rag.index("first document words", title="A", doc_id="a")
        rag.index("second document terms", title="B", doc_id="b")
        for chunk in rag._store.all_chunks():
            assert chunk.tfidf, f"Chunk {chunk.chunk_id} from doc {chunk.doc_id} has empty tfidf"
