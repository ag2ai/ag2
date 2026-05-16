# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""MemoryToolkit — provider-agnostic memory tools for AG2 Beta agents.

Gives an agent four tools: remember, recall, forget, and list_memories.
Memories are stored either in-process (default) or persisted to an SQLite
database when a *storage_path* is supplied.

Example::

    from autogen.beta import Agent
    from autogen.beta.tools import MemoryToolkit
    from autogen.beta.config import OpenAIConfig

    memory = MemoryToolkit()
    agent = Agent("assistant", config=OpenAIConfig("gpt-4o-mini"), tools=[memory])
    # The agent can now call remember/recall/forget/list_memories autonomously.

SQLite persistence::

    memory = MemoryToolkit(storage_path="/tmp/agent-memory.db")
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

from pydantic import Field

from autogen.beta.middleware import ToolMiddleware
from autogen.beta.tools.final import Toolkit, tool
from autogen.beta.tools.final.function_tool import FunctionTool

__all__ = ("MemoryToolkit",)


# ---------------------------------------------------------------------------
# Storage backends
# ---------------------------------------------------------------------------


class _InMemoryStore:
    """Simple dict-backed in-process memory store."""

    def __init__(self) -> None:
        # key → (content, created_at)
        self._store: dict[str, tuple[str, float]] = {}

    def store(self, key: str, content: str) -> None:
        self._store[key] = (content, time.time())

    def retrieve(self, key: str) -> tuple[str, float] | None:
        return self._store.get(key)

    def search(self, query: str, max_results: int) -> list[tuple[str, str, float]]:
        q = query.lower()
        hits = [(k, content, ts) for k, (content, ts) in self._store.items() if q in content.lower() or q in k.lower()]
        hits.sort(key=lambda x: x[2], reverse=True)
        return hits[:max_results]

    def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def list_all(self) -> list[tuple[str, str, float]]:
        return [
            (k, content, ts) for k, (content, ts) in sorted(self._store.items(), key=lambda x: x[1][1], reverse=True)
        ]


class _SQLiteStore:
    """SQLite-backed persistent memory store."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._path, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS memories "
                "(key TEXT PRIMARY KEY, content TEXT NOT NULL, created_at REAL NOT NULL)"
            )

    def store(self, key: str, content: str) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO memories (key, content, created_at) VALUES (?, ?, ?)",
                (key, content, time.time()),
            )

    def retrieve(self, key: str) -> tuple[str, float] | None:
        with self._connect() as con:
            row = con.execute("SELECT content, created_at FROM memories WHERE key = ?", (key,)).fetchone()
        return (row[0], row[1]) if row else None

    def search(self, query: str, max_results: int) -> list[tuple[str, str, float]]:
        q = f"%{query}%"
        with self._connect() as con:
            rows = con.execute(
                "SELECT key, content, created_at FROM memories "
                "WHERE content LIKE ? OR key LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (q, q, max_results),
            ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def delete(self, key: str) -> bool:
        with self._connect() as con:
            cur = con.execute("DELETE FROM memories WHERE key = ?", (key,))
        return cur.rowcount > 0

    def list_all(self) -> list[tuple[str, str, float]]:
        with self._connect() as con:
            rows = con.execute("SELECT key, content, created_at FROM memories ORDER BY created_at DESC").fetchall()
        return [(r[0], r[1], r[2]) for r in rows]


# ---------------------------------------------------------------------------
# Toolkit
# ---------------------------------------------------------------------------


class MemoryToolkit(Toolkit):
    """Toolkit that gives an agent tools to store and recall memories.

    Memories persist within a session (in-memory backend, default) or across
    sessions (SQLite backend, when *storage_path* is set).

    Args:
        storage_path:
            Path to the SQLite database file.  When ``None`` (default)
            memories are kept in-process and lost when the session ends.
        middleware:
            Optional tool-level middleware applied to every memory tool.

    Example::

        from autogen.beta import Agent
        from autogen.beta.tools import MemoryToolkit
        from autogen.beta.config import OpenAIConfig

        # Session-scoped (in-memory)
        memory = MemoryToolkit()

        # Persistent across sessions
        memory = MemoryToolkit(storage_path="~/.my-agent/memories.db")

        agent = Agent("assistant", config=OpenAIConfig("gpt-4o-mini"), tools=[memory])
    """

    __slots__ = ("_store",)

    def __init__(
        self,
        storage_path: str | Path | None = None,
        *,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        if storage_path is None:
            self._store: _InMemoryStore | _SQLiteStore = _InMemoryStore()
        else:
            self._store = _SQLiteStore(Path(storage_path).expanduser().resolve())

        super().__init__(
            self._remember_tool(middleware=middleware),
            self._recall_tool(middleware=middleware),
            self._forget_tool(middleware=middleware),
            self._list_memories_tool(middleware=middleware),
            name="memory_toolkit",
            middleware=(),
        )

    # ------------------------------------------------------------------
    # Tool factories
    # ------------------------------------------------------------------

    def _remember_tool(self, *, middleware: Iterable[ToolMiddleware] = ()) -> FunctionTool:
        store = self._store

        @tool(
            name="remember",
            description=(
                "Store a piece of information in memory for later retrieval. "
                "Supply an optional key to give the memory a name; if omitted "
                "a unique key is generated. Returns the key used."
            ),
            middleware=middleware,
        )
        def _remember(
            content: Annotated[str, Field(description="The information to remember.")],
            key: Annotated[
                str,
                Field(description="Optional name for this memory. Generated automatically if empty."),
            ] = "",
        ) -> str:
            mem_key = key.strip() or f"mem-{uuid.uuid4().hex[:8]}"
            store.store(mem_key, content)
            return f"Stored memory '{mem_key}'."

        return _remember

    def _recall_tool(self, *, middleware: Iterable[ToolMiddleware] = ()) -> FunctionTool:
        store = self._store

        @tool(
            name="recall",
            description=(
                "Search stored memories for entries that match a query. "
                "Returns the best matching memories (up to max_results)."
            ),
            middleware=middleware,
        )
        def _recall(
            query: Annotated[str, Field(description="Search query — matches against memory content and keys.")],
            max_results: Annotated[
                int,
                Field(description="Maximum number of memories to return.", ge=1, le=50),
            ] = 5,
        ) -> str:
            hits = store.search(query, max_results)
            if not hits:
                return f"No memories found matching '{query}'."
            lines = [f"Found {len(hits)} memory(ies) matching '{query}':\n"]
            for key, content, ts in hits:
                created = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
                lines.append(f"[{key}] ({created})\n{content}\n")
            return "\n".join(lines)

        return _recall

    def _forget_tool(self, *, middleware: Iterable[ToolMiddleware] = ()) -> FunctionTool:
        store = self._store

        @tool(
            name="forget",
            description="Delete a stored memory by its key.",
            middleware=middleware,
        )
        def _forget(
            key: Annotated[str, Field(description="The key of the memory to delete.")],
        ) -> str:
            if store.delete(key):
                return f"Deleted memory '{key}'."
            return f"No memory found with key '{key}'."

        return _forget

    def _list_memories_tool(self, *, middleware: Iterable[ToolMiddleware] = ()) -> FunctionTool:
        store = self._store

        @tool(
            name="list_memories",
            description="List all stored memories, most recent first.",
            middleware=middleware,
        )
        def _list_memories() -> str:
            entries = store.list_all()
            if not entries:
                return "No memories stored."
            lines = [f"{len(entries)} stored memory(ies):\n"]
            for key, content, ts in entries:
                created = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
                preview = content[:80] + ("..." if len(content) > 80 else "")
                lines.append(f"[{key}] ({created}) {preview}")
            return "\n".join(lines)

        return _list_memories
