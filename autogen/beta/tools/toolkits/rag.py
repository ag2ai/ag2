# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""RAGToolkit — retrieval-augmented generation tools for AG2 Beta agents.

Gives an agent five tools: ``index_document``, ``search``, ``remove_document``,
``list_documents``, and ``configure_search``.

Documents are automatically split into overlapping chunks.  By default the
toolkit uses TF-IDF keyword scoring (pure stdlib — no extra dependencies).
Pass an *embed_fn* to switch to cosine-similarity semantic search.

Basic usage::

    from autogen.beta import Agent
    from autogen.beta.tools import RAGToolkit
    from autogen.beta.config import OpenAIConfig

    rag = RAGToolkit()
    agent = Agent("assistant", config=OpenAIConfig("gpt-4o-mini"), tools=[rag])

Pre-index documents before the agent starts::

    rag = RAGToolkit()
    rag.index("Python was created by Guido van Rossum in 1991.", title="Python history")
    rag.index("Rust was designed by Graydon Hoare.", title="Rust history")
    agent = Agent("assistant", config=OpenAIConfig("gpt-4o-mini"), tools=[rag])

Semantic search with an OpenAI embedding function::

    from openai import OpenAI

    client = OpenAI()


    def embed(texts: list[str]) -> list[list[float]]:
        resp = client.embeddings.create(input=texts, model="text-embedding-3-small")
        return [d.embedding for d in resp.data]


    rag = RAGToolkit(embed_fn=embed)
"""

from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import Field

from autogen.beta.middleware import ToolMiddleware
from autogen.beta.tools.final import Toolkit, tool
from autogen.beta.tools.final.function_tool import FunctionTool

__all__ = ("RAGToolkit",)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

EmbedFn = Callable[[list[str]], list[list[float]]]

# ---------------------------------------------------------------------------
# Internal data classes
# ---------------------------------------------------------------------------


@dataclass
class _Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    tfidf: dict[str, float] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)


@dataclass
class _DocMeta:
    doc_id: str
    title: str
    chunk_ids: list[str]
    char_count: int


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------


class _Store:
    def __init__(self) -> None:
        self._chunks: dict[str, _Chunk] = {}
        self._docs: dict[str, _DocMeta] = {}

    def add(self, doc_id: str, title: str, chunks: list[_Chunk]) -> None:
        self._docs[doc_id] = _DocMeta(
            doc_id=doc_id,
            title=title,
            chunk_ids=[c.chunk_id for c in chunks],
            char_count=sum(len(c.text) for c in chunks),
        )
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def remove(self, doc_id: str) -> bool:
        meta = self._docs.pop(doc_id, None)
        if meta is None:
            return False
        for cid in meta.chunk_ids:
            self._chunks.pop(cid, None)
        return True

    def all_docs(self) -> list[_DocMeta]:
        return list(self._docs.values())

    def all_chunks(self) -> list[_Chunk]:
        return list(self._chunks.values())


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_OVERLAP = 50
_DEFAULT_TOP_K = 5
_DEFAULT_MIN_SCORE = 0.0
_PARA_RE = re.compile(r"\n\s*\n")


def _chunk_text(
    text: str,
    doc_id: str,
    title: str,
    chunk_size: int,
    overlap: int,
) -> list[_Chunk]:
    paragraphs = [p.strip() for p in _PARA_RE.split(text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[_Chunk] = []
    buf = ""

    for para in paragraphs:
        candidate = (buf + "\n\n" + para).lstrip() if buf else para
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                chunks.append(_make_chunk(buf, doc_id, title))
            # Para alone may exceed chunk_size — hard split with overlap
            if len(para) <= chunk_size:
                buf = para
            else:
                step = max(1, chunk_size - overlap)
                for i in range(0, len(para), step):
                    piece = para[i : i + chunk_size]
                    if piece:
                        chunks.append(_make_chunk(piece, doc_id, title))
                buf = ""

    if buf:
        chunks.append(_make_chunk(buf, doc_id, title))
    return chunks


def _make_chunk(text: str, doc_id: str, title: str) -> _Chunk:
    return _Chunk(chunk_id=str(uuid.uuid4()), doc_id=doc_id, title=title, text=text)


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _recompute_tfidf(chunks: list[_Chunk]) -> None:
    """Recompute TF-IDF weights for every chunk in *chunks* (in place)."""
    n = len(chunks)
    if n == 0:
        return
    tfs: list[Counter[str]] = [Counter(_tokenize(c.text)) for c in chunks]
    df: Counter[str] = Counter()
    for tf in tfs:
        df.update(tf.keys())
    idf = {term: math.log((n + 1) / (freq + 1)) + 1 for term, freq in df.items()}
    for chunk, tf in zip(chunks, tfs):
        total = sum(tf.values()) or 1
        chunk.tfidf = {term: (count / total) * idf[term] for term, count in tf.items()}


def _tfidf_score(query_tokens: list[str], chunk: _Chunk) -> float:
    return sum(chunk.tfidf.get(t, 0.0) for t in query_tokens)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# RAGToolkit
# ---------------------------------------------------------------------------


class RAGToolkit(Toolkit):
    """Retrieval-Augmented Generation toolkit.

    Gives an agent five tools: ``index_document``, ``search``,
    ``remove_document``, ``list_documents``, and ``configure_search``.

    Documents are automatically split into overlapping chunks.  By default
    TF-IDF keyword search is used (no external dependencies).  Pass *embed_fn*
    to switch to cosine-similarity semantic search.

    Args:
        embed_fn: Optional callable that maps a list of text strings to a list
            of float vectors.  When provided, cosine similarity is used for
            retrieval instead of TF-IDF.
        chunk_size: Maximum number of characters per chunk (default 500).
        overlap: Character overlap between consecutive hard-split chunks
            (default 50).
        default_top_k: Default maximum results returned by ``search`` when the
            caller omits *top_k* (default 5).  The model can update this at
            runtime via the ``configure_search`` tool.
        default_min_score: Minimum relevance score for results (default 0.0,
            meaning no filtering).  The model can update this via
            ``configure_search``.
        middleware: Optional sequence of ``ToolMiddleware`` applied to every
            tool in the toolkit.
    """

    __slots__ = ("_store", "_embed_fn", "_chunk_size", "_overlap", "_default_top_k", "_default_min_score")

    def __init__(
        self,
        *,
        embed_fn: EmbedFn | None = None,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        overlap: int = _DEFAULT_OVERLAP,
        default_top_k: int = _DEFAULT_TOP_K,
        default_min_score: float = _DEFAULT_MIN_SCORE,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        self._store = _Store()
        self._embed_fn = embed_fn
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._default_top_k = default_top_k
        self._default_min_score = default_min_score

        super().__init__(
            self.index_document(),
            self.search(),
            self.remove_document(),
            self.list_documents(),
            self.configure_search(),
            name="rag_toolkit",
            middleware=middleware,
        )

    # ------------------------------------------------------------------
    # Pre-indexing helper (outside agent tool calls)
    # ------------------------------------------------------------------

    def index(self, content: str, *, title: str = "", doc_id: str | None = None) -> str:
        """Index *content* programmatically before the agent starts.

        Returns the *doc_id* assigned to the document.
        """
        return self._do_index(content, title=title, doc_id=doc_id or str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Internal indexing
    # ------------------------------------------------------------------

    def _do_index(self, content: str, *, title: str, doc_id: str) -> str:
        chunks = _chunk_text(
            content,
            doc_id=doc_id,
            title=title or doc_id,
            chunk_size=self._chunk_size,
            overlap=self._overlap,
        )
        if not chunks:
            return doc_id

        if self._embed_fn is not None:
            embeddings = self._embed_fn([c.text for c in chunks])
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb

        self._store.add(doc_id, title or doc_id, chunks)

        if self._embed_fn is None:
            _recompute_tfidf(self._store.all_chunks())

        return doc_id

    # ------------------------------------------------------------------
    # Tool factories
    # ------------------------------------------------------------------

    def index_document(
        self,
        *,
        name: str = "index_document",
        description: str = (
            "Index a document so it can be retrieved with search(). "
            "The document is automatically split into overlapping chunks. "
            "Returns the document ID."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        @tool(name=name, description=description, middleware=middleware)
        def _index_document(
            content: Annotated[str, Field(description="Full text of the document to index.")],
            title: Annotated[str, Field(description="Optional human-readable title for this document.")] = "",
            doc_id: Annotated[str, Field(description="Optional stable ID. Auto-generated UUID if omitted.")] = "",
        ) -> str:
            actual_id = doc_id.strip() or str(uuid.uuid4())
            self._do_index(content, title=title, doc_id=actual_id)
            return f"Indexed as {actual_id!r} ({len(content)} chars)"

        return _index_document

    def search(
        self,
        *,
        name: str = "search",
        description: str = (
            "Search the indexed documents for chunks relevant to the query. "
            "Returns the top matching passages with their source document IDs and titles. "
            "Uses the configured defaults for top_k and min_score when those parameters are omitted."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        @tool(name=name, description=description, middleware=middleware)
        def _search(
            query: Annotated[str, Field(description="Natural-language query to search for.")],
            top_k: Annotated[
                int | None,
                Field(
                    description="Maximum number of results (1–20). Uses the configured default when omitted.",
                    ge=1,
                    le=20,
                ),
            ] = None,
        ) -> str:
            chunks = self._store.all_chunks()
            if not chunks:
                return "No documents indexed yet."

            effective_top_k = top_k if top_k is not None else self._default_top_k

            if self._embed_fn is not None:
                q_emb = self._embed_fn([query])[0]
                scored = [(c, _cosine(q_emb, c.embedding)) for c in chunks if c.embedding]
            else:
                tokens = _tokenize(query)
                scored = [(c, _tfidf_score(tokens, c)) for c in chunks]

            scored.sort(key=lambda x: x[1], reverse=True)
            top = [(c, s) for c, s in scored[:effective_top_k] if s >= self._default_min_score]

            if not top:
                return "No relevant results found."

            parts: list[str] = []
            for rank, (chunk, score) in enumerate(top, 1):
                header = f"[{rank}] doc_id={chunk.doc_id!r} title={chunk.title!r} score={score:.3f}"
                parts.append(f"{header}\n{chunk.text}")
            return "\n\n---\n\n".join(parts)

        return _search

    def remove_document(
        self,
        *,
        name: str = "remove_document",
        description: str = "Remove a previously indexed document and all its chunks.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        @tool(name=name, description=description, middleware=middleware)
        def _remove_document(
            doc_id: Annotated[str, Field(description="Document ID returned by index_document.")],
        ) -> str:
            removed = self._store.remove(doc_id)
            if not removed:
                return f"Document {doc_id!r} not found."
            if self._embed_fn is None:
                _recompute_tfidf(self._store.all_chunks())
            return f"Removed document {doc_id!r}."

        return _remove_document

    def configure_search(
        self,
        *,
        name: str = "configure_search",
        description: str = (
            "Persist search defaults for this RAG session. "
            "top_k sets how many results search() returns by default; "
            "min_score filters out results whose relevance score is below the threshold "
            "(0.0 means no filtering). "
            "Omit a parameter to leave its current value unchanged. "
            "Returns a summary of the active configuration."
        ),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        @tool(name=name, description=description, middleware=middleware)
        def _configure_search(
            top_k: Annotated[
                int | None,
                Field(
                    description="Default number of results returned by search() (1–20). Omit to keep current value.",
                    ge=1,
                    le=20,
                ),
            ] = None,
            min_score: Annotated[
                float | None,
                Field(
                    description="Minimum relevance score threshold (0.0–1.0). Results below this are excluded. Omit to keep current value.",
                    ge=0.0,
                    le=1.0,
                ),
            ] = None,
        ) -> str:
            if top_k is not None:
                self._default_top_k = top_k
            if min_score is not None:
                self._default_min_score = min_score
            return f"Search configured — top_k: {self._default_top_k}, min_score: {self._default_min_score:.3f}"

        return _configure_search

    def list_documents(
        self,
        *,
        name: str = "list_documents",
        description: str = "List all indexed documents with their IDs, titles, and sizes.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        @tool(name=name, description=description, middleware=middleware)
        def _list_documents() -> str:
            docs = self._store.all_docs()
            if not docs:
                return "No documents indexed."
            header = f"{'doc_id':<36}  {'chars':>6}  title"
            sep = "-" * 70
            rows = [header, sep]
            for doc in docs:
                rows.append(f"{doc.doc_id:<36}  {doc.char_count:>6}  {doc.title}")
            return "\n".join(rows)

        return _list_documents
