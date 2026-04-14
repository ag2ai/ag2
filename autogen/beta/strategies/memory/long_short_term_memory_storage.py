# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import tiktoken

from ...annotations import Context
from ...context import StreamId
from ...events import (
    AudioUrlInput,
    BaseEvent,
    BinaryInput,
    DocumentUrlInput,
    FileIdInput,
    ImageUrlInput,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallsEvent,
    ToolResultsEvent,
    VideoUrlInput,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Memory block data classes
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class L1MemoryBlock:
    """Level-1 short-term memory block.

    Created by compressing conversation turns that have scrolled out of the
    active context window. The raw source text is always available for
    next-level retrieval.
    """

    id: str
    abstract: str
    source_text: str
    created_at: str = field(default_factory=_utc_now_iso)


@dataclass
class L2MemoryBlock:
    """Level-2 long-term memory block.

    Created by asynchronously consolidating one or more L1 blocks into a
    single consolidated representation that persists across sessions.
    """

    id: str
    abstract: str
    source_blocks: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)


@dataclass
class CoreMemoryBlock:
    """Singleton core memory block.

    Stores stable, identity-level facts about the user (name, occupation,
    preferences, goals, constraints).  Unlike L1/L2 blocks, its full
    *content* is always shown verbatim in the context window so the agent
    always has instant access to the most important user context — no
    ``memory_lookup`` call required.

    There is at most one core memory block per storage instance.  It is
    created on the first session end and updated on every subsequent one.
    """

    content: str
    updated_at: str = field(default_factory=_utc_now_iso)


# Callable type for the user-provided summarisation function
Summarizer = Callable[[str], Awaitable[str]]

# Callable type for the core memory consolidation function.
# Receives (current_core_block_or_None, new_l1_blocks) and returns
# the new content string for the singleton core memory block.
CoreConsolidator = Callable[
    ["CoreMemoryBlock | None", list["L1MemoryBlock"]],
    Awaitable[str],
]


# ---------------------------------------------------------------------------
# L2 operation types (compact diff format for LLM-driven consolidation)
# ---------------------------------------------------------------------------


@dataclass
class CreateL2Op:
    """Create a brand-new L2 block.

    ``new_l1_ids`` — IDs drawn from the *pending* L1 batch that will be
    recorded as the source blocks of the new L2 block.
    """

    abstract: str
    new_l1_ids: list[str] = field(default_factory=list)


@dataclass
class UpdateL2Op:
    """Update the abstract of an existing L2 block and optionally link new L1 sources.

    Existing ``source_blocks`` are preserved automatically by the storage;
    ``new_l1_ids`` are *appended* to them (not a full replacement).
    """

    id: str
    abstract: str
    new_l1_ids: list[str] = field(default_factory=list)


@dataclass
class DeleteL2Op:
    """Delete an existing L2 block."""

    id: str


# Union of all operation types returned by a Consolidator
L2Operation = CreateL2Op | UpdateL2Op | DeleteL2Op

# Callable type for the user-provided L2 consolidation function.
# Receives (existing_l2_blocks, new_l1_blocks, max_l2_blocks) and returns
# a list of operations describing only what changed.  Unchanged blocks are
# not mentioned, keeping the LLM output compact.
Consolidator = Callable[
    [list["L2MemoryBlock"], list["L1MemoryBlock"], int],
    Awaitable[list[L2Operation]],
]


def parse_l2_operation(d: dict) -> L2Operation:
    """Parse a plain dict (e.g. decoded from LLM JSON output) into a typed ``L2Operation``.

    Expected shapes::

        {"op": "create", "abstract": "...", "new_l1_ids": ["L1-abc"]}
        {"op": "update", "id": "L2-xyz", "abstract": "...", "new_l1_ids": ["L1-def"]}
        {"op": "delete", "id": "L2-xyz"}

    ``new_l1_ids`` is optional and defaults to ``[]``.
    """
    op = d.get("op")
    if op == "create":
        return CreateL2Op(abstract=d["abstract"], new_l1_ids=d.get("new_l1_ids", []))
    if op == "update":
        return UpdateL2Op(id=d["id"], abstract=d["abstract"], new_l1_ids=d.get("new_l1_ids", []))
    if op == "delete":
        return DeleteL2Op(id=d["id"])
    raise ValueError(f"Unknown L2 operation type: {op!r}")


# ---------------------------------------------------------------------------
# Default prompts for the built-in summarizer and consolidator
# ---------------------------------------------------------------------------

# Soft upper bound (word count) communicated to the default LLM prompts for L1/L2 abstracts.
_DEFAULT_MEMORY_ABSTRACT_MAX_WORDS = 500

_DEFAULT_SUMMARIZER_SYSTEM = (
    "You are a summariser for a memory index. Reply with only the summary, no preamble. "
    "The summary becomes each block’s abstract (shown in the pinned memory header); the agent "
    "chooses which blocks to load from it. You may write one paragraph or several; stay within "
    f"{_DEFAULT_MEMORY_ABSTRACT_MAX_WORDS} words total. Keep concrete specifics that distinguish "
    "this segment from others."
)
_DEFAULT_SUMMARIZER_USER = (
    "Summarise the following. You may use one paragraph or several; keep the total under "
    "{max_words} words. Preserve retrieval-critical specifics from the "
    "source including but not limited to: people’s names and roles; organisation, product, and place "
    "names; dates, times, time zones; numbers, versions, IDs, URLs, and file "
    "paths; explicit decisions, outcomes, and open questions. Prefer named entities over vague "
    "phrases like “the user” or “a meeting” when a proper noun or time is known. Do not strip "
    "information the agent would need to decide whether to call memory_lookup for this block.\n\n"
    "{text}"
)

_DEFAULT_CONSOLIDATOR_SYSTEM = (
    "You are a long-term memory manager for an AI assistant. Each block’s abstract is the primary "
    "hint the agent sees before retrieving full text; abstracts must stay specific enough "
    "(names, places, times, key facts) to support accurate lookup decisions. "
    "An abstract may be several paragraphs (still within the word limit given in the user "
    "message). Return only valid JSON with no markdown fences."
)
_DEFAULT_CONSOLIDATOR_USER = """\
You manage long-term memory for an AI assistant.
Max allowed memory blocks: {max_blocks}

Existing long-term memory (id, creation time, abstract):
{existing_text}

New short-term blocks from this session to integrate (id, creation time, abstract):
{new_l1_text}

Decide how to update the long-term memory. You may:
  - create : add a new block for genuinely new topics
  - update : rewrite an existing block's abstract to absorb related new info
  - delete : remove a block that is now fully superseded or irrelevant

Rules:
  - Only mention blocks that actually change; unchanged blocks are omitted.
  - new_l1_ids must be IDs drawn from the new short-term blocks listed above.
  - Keep total blocks <= {max_blocks}.
  - Each "abstract" (create or update) may be up to {max_abstract_words} words—one paragraph or several. Prefer enough detail for retrieval over extreme brevity. For paragraph breaks inside JSON string values, use \\n escapes.
  - Every "abstract" you output (create or update) is a retrieval index: carry forward concrete details from the relevant L1 abstracts—people’s names and roles; organisation, product, and place names; dates, times, time zones, and deadlines; numbers, versions, IDs, URLs, and file paths; decisions, outcomes, and open questions. Merge related L1 material without collapsing into vague prose; if two topics differ by who/when/where, keep those distinctions visible.

Return ONLY a JSON array of operations, for example:
[
  {{"op": "create",  "abstract": "...", "new_l1_ids": ["L1-abc"]}},
  {{"op": "update",  "id": "L2-xyz",   "abstract": "...", "new_l1_ids": ["L1-def"]}},
  {{"op": "delete",  "id": "L2-old"}}
]\
"""


def _build_default_summarizer(config: Any) -> "Summarizer":
    """Return a summarizer backed by *config* using the default prompt."""
    async def _summarizer(text: str) -> str:
        from ...agent import Agent  # local import to avoid circular dependency
        agent = Agent("summariser", prompt=_DEFAULT_SUMMARIZER_SYSTEM, config=config)
        reply = await agent.ask(
            _DEFAULT_SUMMARIZER_USER.format(
                max_words=_DEFAULT_MEMORY_ABSTRACT_MAX_WORDS,
                text=text,
            )
        )
        return (await reply.content()) or ""
    return _summarizer


_DEFAULT_CORE_MEMORY_SYSTEM = (
    "You maintain a concise core memory profile for an AI assistant. "
    "Reply with only the updated profile text, no preamble or explanation."
)
_DEFAULT_CORE_MEMORY_USER = """\
You are updating a core memory profile that captures essential, persistent facts about the user.

Current core memory (empty if this is the first session):
{current_content}

New information from this session (L1 memory blocks, id + abstract):
{new_l1_text}

Update the core memory profile. Focus on stable, identity-level facts such as:
  - Name, age, location
  - Occupation, role, current projects
  - Preferences, recurring goals, known constraints
  - Anything the user explicitly wants the assistant to always remember

Rules:
  - Keep it concise (a few sentences or a short bullet list).
  - Merge new information with existing — do not lose previous facts unless they are corrected.
  - Omit session-specific or transient details (those belong in L2 blocks).

Reply with only the updated core memory text.\
"""


def _build_default_core_consolidator(config: Any) -> "CoreConsolidator":
    """Return a core memory consolidator backed by *config* using the default prompt."""
    async def _core_consolidator(
        current: "CoreMemoryBlock | None",
        new_l1: "list[L1MemoryBlock]",
    ) -> str:
        current_content = current.content if current else "(none yet)"
        new_l1_text = (
            "\n".join(f"  [{b.id}] (created {b.created_at}) {b.abstract}" for b in new_l1)
            or "  (none)"
        )
        prompt = _DEFAULT_CORE_MEMORY_USER.format(
            current_content=current_content,
            new_l1_text=new_l1_text,
        )
        from ...agent import Agent  # local import to avoid circular dependency
        agent = Agent("core-memory-manager", prompt=_DEFAULT_CORE_MEMORY_SYSTEM, config=config)
        reply = await agent.ask(prompt)
        return (await reply.content()) or current_content

    return _core_consolidator


def _build_default_consolidator(config: Any) -> "Consolidator":
    """Return a smart LLM-driven consolidator backed by *config* using the default prompt."""
    async def _consolidator(
        existing: list["L2MemoryBlock"],
        new_l1: list["L1MemoryBlock"],
        max_blocks: int,
    ) -> "list[L2Operation]":
        if not existing and not new_l1:
            return []

        existing_text = (
            "\n".join(f"  [{b.id}] (created {b.created_at}) {b.abstract}" for b in existing)
            or "  (none yet)"
        )
        new_l1_text = (
            "\n".join(f"  [{b.id}] (created {b.created_at}) {b.abstract}" for b in new_l1)
            or "  (none)"
        )
        prompt = _DEFAULT_CONSOLIDATOR_USER.format(
            max_blocks=max_blocks,
            max_abstract_words=_DEFAULT_MEMORY_ABSTRACT_MAX_WORDS,
            existing_text=existing_text,
            new_l1_text=new_l1_text,
        )

        from ...agent import Agent  # local import to avoid circular dependency
        agent = Agent("memory-manager", prompt=_DEFAULT_CONSOLIDATOR_SYSTEM, config=config)
        reply = await agent.ask(prompt)
        raw = (await reply.content()) or "[]"
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return [parse_l2_operation(d) for d in json.loads(raw)]

    return _consolidator


# ---------------------------------------------------------------------------
# LongShortTermMemoryStorage
# ---------------------------------------------------------------------------


class LongShortTermMemoryStorage:
    """Two-level agent memory storage compatible with the ag2-beta Storage protocol.

    Context composition returned by ``get_history``::

        ┌──────────────────────────────────────────────┐
        │  [pinned] Core memory (full text, always on) │  ← identity profile
        │  [pinned] L2 block summaries (IDs + abs)     │  ← long-term index
        │  [pinned] L1 block summaries (IDs + abs)     │  ← short-term index
        ├──────────────────────────────────────────────┤
        │  recent raw conversation events              │  ← active window
        └──────────────────────────────────────────────┘

    **Core memory** — singleton block, always fully visible.
    Created on the first session end and updated on every subsequent one.
    Stores stable user identity facts (name, role, preferences, goals).
    Unlike L1/L2 blocks, no ``memory_lookup`` call is needed; the full
    content is always in the context window.

    **L1 short-term memory** — compressed within a session.
    Triggered lazily inside ``get_history`` whenever the raw-event token
    budget exceeds *token_threshold* (by default, ``memory_lookup`` tool
    lines are excluded from that measurement and from L1 ``source_text``).
    On each **lazy** compression, ``memory_lookup`` call/result *events* are
    also dropped from raw history for everything **before** the last
    *recent_turns* user messages; the recent suffix is left intact so the
    model still sees fresh lookups. At **session end**, ``consolidate()``
    flushes **all** remaining raw into a **single** L1 block (again stripping
    ``memory_lookup`` tool events from that summary). L1 blocks exist only for
    the lifetime of the current Python process.

    **L2 long-term memory** — persisted across sessions to a JSON file.
    L2 consolidation is *not* triggered automatically during the session;
    it must be invoked explicitly after the agent task ends::

        # Option A – explicit call
        reply = await agent.ask("hello", stream=stream)
        await storage.consolidate()

        # Option B – async context manager (recommended)
        async with storage.session():
            reply = await agent.ask("hello", stream=stream)
        # consolidation + JSON save happen automatically on exit

    **Minimal usage** — pass only a model *config* and a store path::

        from autogen.beta.config.openai.config import OpenAIConfig
        from autogen.beta.history import LongShortTermMemoryStorage

        config = OpenAIConfig(model="gpt-4o-mini", api_key="sk-...")
        storage = LongShortTermMemoryStorage(
            config=config,
            l2_store_path="agent_memory.json",
        )

    Built-in default prompts are used for both summarisation and L2
    consolidation.  Pass custom *summarizer* / *consolidator* callables only
    when you need to override the defaults.

    **Custom consolidation logic**

    The consolidator receives all existing L2 blocks, the new L1 blocks, and
    the max-block cap, then returns a compact list of *operations* — only the
    blocks that actually change are mentioned (unchanged blocks are omitted to
    save tokens)::

        async def my_consolidator(
            existing: list[L2MemoryBlock],
            new_l1: list[L1MemoryBlock],
            max_blocks: int,
        ) -> list[L2Operation]:
            # JSON the LLM should produce:
            # [
            #   {"op": "create",  "abstract": "...", "new_l1_ids": ["L1-abc"]},
            #   {"op": "update",  "id": "L2-xyz", "abstract": "...", "new_l1_ids": ["L1-def"]},
            #   {"op": "delete",  "id": "L2-old"}
            # ]
            ...

        storage = LongShortTermMemoryStorage(
            config=config,
            consolidator=my_consolidator,
            l2_store_path="memory.json",
        )
    """

    def __init__(
        self,
        *,
        config: Any | None = None,
        summarizer: "Summarizer | None" = None,
        consolidator: "Consolidator | None" = None,
        core_consolidator: "CoreConsolidator | None" = None,
        l2_store_path: str | Path | None = None,
        token_counter: Callable[[str], int] | None = None,
        token_threshold: int = 10_000,
        max_l2_blocks: int = 20,
        recent_turns: int = 5,
        l1_ignore_tool_names: frozenset[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        config:
            Model configuration (e.g. ``OpenAIConfig``) passed to the
            built-in summariser, consolidator, and core memory consolidator.
            Required unless both *summarizer* and *consolidator* are supplied
            explicitly.
        summarizer:
            Optional async callable ``(text: str) -> str`` used for L1
            compression.  When ``None`` a default implementation backed by
            *config* is used.
        consolidator:
            Optional async callable
            ``(existing_l2, new_l1, max_blocks) -> list[L2Operation]``
            that fully controls how L2 blocks are updated at session end.
            When ``None`` a smart LLM-driven default backed by *config* is
            used; if *config* is also ``None`` the built-in simple strategy
            (append / merge-oldest) is used as the final fallback.
        core_consolidator:
            Optional async callable
            ``(current_core: CoreMemoryBlock | None, new_l1: list[L1MemoryBlock]) -> str``
            that rewrites the singleton core memory block at session end.
            When ``None`` a default LLM-driven implementation backed by
            *config* is used; if *config* is also ``None`` the summariser
            is used as a simple fallback.
        l2_store_path:
            Path to a JSON file used to persist memory across sessions.
            L2 blocks, their referenced L1 blocks, and the core memory block
            are all stored here.  L1 blocks that are no longer referenced by
            any L2 block are pruned on each save.
            The file is read on construction (if it exists) and written after
            every successful ``consolidate()`` call.  Pass ``None`` to keep
            all memory in-process only.
        token_counter:
            Optional callable ``(text: str) -> int``.  Defaults to counting
            tokens with ``tiktoken`` using the ``gpt-4o`` model encoding
            (``o200k_base``).  Pass :func:`_char_token_estimate` to restore the
            previous character-based heuristic (1 token ≈ 4 characters).
        token_threshold:
            Raw-event token budget that triggers L1 compression (default 10 000).
        max_l2_blocks:
            Hard upper limit on the total number of L2 blocks.
        recent_turns:
            Number of most-recent user/assistant turn pairs kept as raw events
            in the active context window; older turns are compressed into L1.
        l1_ignore_tool_names:
            Tool names whose call and result lines are omitted when measuring
            raw context size for L1 compression and when building L1
            ``source_text`` / summariser input (so large ``memory_lookup``
            payloads do not spuriously trigger compression or get re-summarised
            into a new L1 block).  ``None`` defaults to ``frozenset({"memory_lookup"})``.
            Pass ``frozenset()`` to disable filtering.
        """
        if summarizer is None and config is None:
            raise ValueError(
                "LongShortTermMemoryStorage requires either 'config' (to use the built-in "
                "summariser) or an explicit 'summarizer' callable."
            )

        self._summarizer: Summarizer = summarizer or _build_default_summarizer(config)

        if consolidator is not None:
            self._consolidator: Consolidator = consolidator
        elif config is not None:
            self._consolidator = _build_default_consolidator(config)
        else:
            self._consolidator = self._default_consolidator

        if core_consolidator is not None:
            self._core_consolidator: CoreConsolidator = core_consolidator
        elif config is not None:
            self._core_consolidator = _build_default_core_consolidator(config)
        else:
            self._core_consolidator = self._default_core_consolidator

        self._l2_store_path = Path(l2_store_path) if l2_store_path else None
        self._token_counter = token_counter or _gpt4o_token_count
        self._token_threshold = token_threshold
        self._max_l2_blocks = max_l2_blocks
        self._recent_turns = recent_turns
        self._l1_ignore_tool_names: frozenset[str] = (
            frozenset(l1_ignore_tool_names)
            if l1_ignore_tool_names is not None
            else frozenset({"memory_lookup"})
        )

        # per-stream raw event store (only recent, un-compressed events)
        self._raw_events: defaultdict[StreamId, list[BaseEvent]] = defaultdict(list)

        # per-stream L1 blocks produced in the current session
        self._l1_blocks: defaultdict[StreamId, list[L1MemoryBlock]] = defaultdict(list)

        # L1 blocks accumulated this session, waiting for end-of-task consolidation
        self._pending_l1_for_l2: list[L1MemoryBlock] = []

        # L1 blocks loaded from disk (from previous sessions).
        # Contains only blocks still referenced by a current L2 block.
        self._persisted_l1_blocks: list[L1MemoryBlock] = []

        # L2 blocks loaded from disk; updated only by consolidate()
        self._l2_blocks: list[L2MemoryBlock] = []

        # Singleton core memory block; None until the first session ends
        self._core_memory: CoreMemoryBlock | None = None

        self._load_store()

    # ------------------------------------------------------------------
    # Storage protocol
    # ------------------------------------------------------------------

    async def save_event(self, event: "BaseEvent", context: "Context") -> None:
        stream_id = context.stream.id
        if event not in self._raw_events[stream_id]:
            self._raw_events[stream_id].append(event)

    async def get_history(self, stream_id: "StreamId") -> Iterable["BaseEvent"]:
        await self._maybe_compress_to_l1(stream_id)

        result: list[BaseEvent] = []

        header = self._build_memory_header(stream_id)
        if header is not None:
            result.append(header)

        result.extend(self._raw_events[stream_id])
        return result

    async def set_history(self, stream_id: "StreamId", events: Iterable["BaseEvent"]) -> None:
        # Replace only the raw (un-compressed) events.  L1/L2 blocks are
        # intentionally preserved so that middleware-driven history rewrites
        # (e.g. context-window trimming) do not destroy the memory index.
        self._raw_events[stream_id] = list(events)

    async def drop_history(self, stream_id: "StreamId") -> None:
        self._raw_events.pop(stream_id, None)
        self._l1_blocks.pop(stream_id, None)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def session(self) -> AsyncIterator[None]:
        """Async context manager for a single agent task session.

        On exit, ``consolidate()`` is called automatically, which merges all
        L1 blocks accumulated during the session into L2 and saves them to
        the JSON store (if *l2_store_path* was provided).

        Example::

            async with storage.session():
                reply = await agent.ask("hello", stream=stream)
        """
        try:
            yield
        finally:
            await self.consolidate()

    async def consolidate(self) -> None:
        """Consolidate all pending L1 blocks into L2 and core memory, then persist.

        Call this once after the agent task (session) has completed.  Using
        the ``session()`` context manager is the preferred way to ensure this
        is called automatically.

        Order of operations:
        1. ``_flush_remaining_to_l1`` — force **all** remaining raw events
           into one final L1 block (``memory_lookup`` tool traffic excluded).
        2. L2 consolidation — the consolidator updates/creates/deletes L2
           blocks based on the pending L1 blocks.  If this step fails the
           pending blocks are restored and nothing is saved.
        3. Core memory consolidation — the core consolidator rewrites the
           singleton core memory block.  If this step fails the previous
           core memory is retained, but L2 changes are still saved.
        4. Persist everything to the JSON store.

        If there are no pending L1 blocks the method is a no-op.
        """
        # Capture every remaining raw event in one L1 block before L2 merge.
        await self._flush_remaining_to_l1()

        if not self._pending_l1_for_l2:
            return

        pending = list(self._pending_l1_for_l2)
        self._pending_l1_for_l2.clear()

        # --- L2 consolidation ---
        try:
            ops = await self._consolidator(
                list(self._l2_blocks),
                pending,
                self._max_l2_blocks,
            )
            self._l2_blocks = self._apply_l2_ops(ops)
        except Exception:
            logger.exception("LongShortTermMemoryStorage: L2 consolidation failed; restoring pending blocks")
            self._pending_l1_for_l2.extend(pending)
            return

        # --- Core memory consolidation ---
        try:
            new_content = await self._core_consolidator(self._core_memory, pending)
            self._core_memory = CoreMemoryBlock(content=new_content)
            logger.info(
                "LongShortTermMemoryStorage: core memory updated (%d chars)",
                len(new_content),
            )
        except Exception:
            logger.exception(
                "LongShortTermMemoryStorage: core memory consolidation failed; "
                "keeping previous core memory"
            )

        self._save_store()

    async def _default_consolidator(
        self,
        existing: list[L2MemoryBlock],
        new_l1: list[L1MemoryBlock],
        max_blocks: int,
    ) -> list[L2Operation]:
        """Built-in consolidation strategy used when no *consolidator* is provided.

        - If the cap is not yet reached, emits a single ``CreateL2Op``.
        - If the cap is already reached, emits a single ``UpdateL2Op`` that
          rewrites the oldest block's abstract and appends the new L1 sources.
        """
        l1_summary_text = "\n\n".join(f"[{b.id}] {b.abstract}" for b in new_l1)
        new_l1_ids = [b.id for b in new_l1]

        if len(existing) < max_blocks:
            abstract = await self._summarizer(l1_summary_text)
            return [CreateL2Op(abstract=abstract, new_l1_ids=new_l1_ids)]

        oldest = existing[0]
        combined = f"{oldest.abstract}\n\nNew information:\n{l1_summary_text}"
        abstract = await self._summarizer(combined)
        return [UpdateL2Op(id=oldest.id, abstract=abstract, new_l1_ids=new_l1_ids)]

    async def _default_core_consolidator(
        self,
        current: "CoreMemoryBlock | None",
        new_l1: list[L1MemoryBlock],
    ) -> str:
        """Built-in core memory strategy used when neither *core_consolidator*
        nor *config* is provided.

        Merges the existing core memory content (if any) with the new L1
        abstracts using the summariser, producing a concise updated profile.
        """
        new_l1_text = "\n\n".join(f"[{b.id}] {b.abstract}" for b in new_l1)
        if current:
            combined = f"Existing core memory:\n{current.content}\n\nNew information:\n{new_l1_text}"
        else:
            combined = new_l1_text
        return await self._summarizer(combined)

    def _apply_l2_ops(self, ops: list[L2Operation]) -> list[L2MemoryBlock]:
        """Apply a list of ``L2Operation`` diffs to ``self._l2_blocks`` and return
        the resulting block list.

        - ``CreateL2Op`` — appends a new block (ID is auto-generated here, not by the LLM).
        - ``UpdateL2Op`` — replaces the abstract; ``new_l1_ids`` are *appended* to the
          existing ``source_blocks`` without the LLM having to repeat them.
        - ``DeleteL2Op`` — removes the block.

        Ordering: deletions and updates are applied first (preserving insertion order),
        then creations are appended at the end.
        """
        blocks: dict[str, L2MemoryBlock] = {b.id: b for b in self._l2_blocks}
        order: list[str] = [b.id for b in self._l2_blocks]
        pending_creates: list[CreateL2Op] = []

        for op in ops:
            if isinstance(op, DeleteL2Op):
                blocks.pop(op.id, None)
                if op.id in order:
                    order.remove(op.id)

            elif isinstance(op, UpdateL2Op):
                if op.id in blocks:
                    old = blocks[op.id]
                    blocks[op.id] = L2MemoryBlock(
                        id=op.id,
                        abstract=op.abstract,
                        source_blocks=old.source_blocks + op.new_l1_ids,
                        created_at=old.created_at,  # preserve original creation time
                    )
                else:
                    logger.warning("UpdateL2Op: block %r not found; skipping", op.id)

            elif isinstance(op, CreateL2Op):
                pending_creates.append(op)

        for create_op in pending_creates:
            block_id = f"L2-{uuid4().hex[:8]}"
            blocks[block_id] = L2MemoryBlock(
                id=block_id,
                abstract=create_op.abstract,
                source_blocks=create_op.new_l1_ids,
            )
            order.append(block_id)

        return [blocks[bid] for bid in order if bid in blocks]

    # ------------------------------------------------------------------
    # Retrieval tool (expose as an agent tool)
    # ------------------------------------------------------------------

    async def memory_lookup(self, block_id: str) -> str:
        """Retrieve the full content of a memory block by its ID.

        The agent memory is organised into two tiers:

        **L1 blocks (short-term memory)**
          Created during a session when the active context window exceeds the
          token threshold.  Each L1 block stores a compressed *abstract* and
          the *full raw source text* of the original conversation turns.
          IDs look like ``"L1-a3f8c120"``.

          Calling ``memory_lookup("L1-<id>")`` returns the complete source
          text so you can read the exact conversation that was summarised.

        **L2 blocks (long-term memory)**
          Created at the end of a session by consolidating one or more L1
          blocks into a single consolidated entry that persists across
          sessions.  Each L2 block stores an *abstract* and the list of L1
          block IDs it was built from.
          IDs look like ``"L2-b7e91d04"``.

          Calling ``memory_lookup("L2-<id>")`` returns the IDs and abstracts
          of the underlying L1 blocks.  You can then call
          ``memory_lookup("L1-<id>")`` on any of those to read the full text.

        How to call this tool:
          Pass exactly one argument — the block ID string (e.g.
          ``memory_lookup("L2-b7e91d04")``).  Block IDs are shown in the
          pinned memory header at the top of the conversation context.

        Returns a plain-text string describing the block content, or an error
        message if the ID is not recognised.
        """
        # Current-session L1 blocks + L1 blocks persisted from previous sessions
        all_l1 = [b for blocks in self._l1_blocks.values() for b in blocks] + self._persisted_l1_blocks

        for l2 in self._l2_blocks:
            if l2.id == block_id:
                found_ids = []
                missing_ids = []
                lines = [
                    f"[L2 Block {l2.id}]",
                    f"Created: {l2.created_at}",
                    f"Abstract: {l2.abstract}",
                    "",
                    "Source L1 Blocks:",
                ]
                for l1_id in l2.source_blocks:
                    l1 = next((b for b in all_l1 if b.id == l1_id), None)
                    if l1:
                        found_ids.append(l1_id)
                        lines.append(f"  [{l1.id}] (created {l1.created_at}) {l1.abstract}")
                    else:
                        missing_ids.append(l1_id)
                        logger.warning(
                            "memory_lookup: L1 block %r referenced by L2 block %r was not found "
                            "in the current session or persistent store",
                            l1_id,
                            block_id,
                        )
                        lines.append(f"  [{l1_id}] (source block not found)")
                logger.info(
                    "memory_lookup: L2 block %r retrieved (abstract=%r, source_blocks=%d found / %d missing)",
                    block_id,
                    l2.abstract[:80] + ("…" if len(l2.abstract) > 80 else ""),
                    len(found_ids),
                    len(missing_ids),
                )
                return "\n".join(lines)

        l1 = next((b for b in all_l1 if b.id == block_id), None)
        if l1:
            logger.info(
                "memory_lookup: L1 block %r retrieved (abstract=%r, source_text_chars=%d)",
                block_id,
                l1.abstract[:80] + ("…" if len(l1.abstract) > 80 else ""),
                len(l1.source_text),
            )
            return f"[L1 Block {l1.id}]\nCreated: {l1.created_at}\nAbstract: {l1.abstract}\n\nSource Text:\n{l1.source_text}"

        logger.warning("memory_lookup: block %r not found (known L2=%d, known L1=%d)", block_id, len(self._l2_blocks), len(all_l1))
        return f"No memory block found with ID: {block_id!r}. Valid IDs are shown in the memory header."

    def memory_lookup_tool(self) -> Callable[[str], Awaitable[str]]:
        """Return the ``memory_lookup`` coroutine bound to this storage instance.

        Pass the result directly to ``Agent(tools=[...])`` or ``agent.ask(tools=[...])``.
        """
        return self.memory_lookup

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _flush_remaining_to_l1(self) -> None:
        """Force-compress **all** of each stream's remaining raw events into one L1 block.

        Called at the start of ``consolidate()`` so nothing is left only in the
        raw buffer when L2 runs.  Unlike lazy L1, this does **not** honour
        *recent_turns*: the entire tail becomes a single block.

        ``memory_lookup`` tool call/result events are pruned from the event
        list first, and transcript lines for ignored tools are still omitted via
        ``_events_to_text``, so bulky retrievals are not embedded in L1
        ``source_text``.
        """
        for stream_id, raw in list(self._raw_events.items()):
            if not raw:
                continue
            pruned: list[BaseEvent] = []
            for ev in raw:
                p = _prune_l1_ignored_tools_from_event(
                    ev,
                    ignore_tool_names=self._l1_ignore_tool_names,
                )
                if p is not None:
                    pruned.append(p)

            source_text = _events_to_text(pruned, ignore_tool_names=self._l1_ignore_tool_names)
            if not source_text.strip():
                self._raw_events[stream_id] = []
                continue
            try:
                abstract = await self._summarizer(source_text)
            except Exception:
                logger.exception(
                    "LongShortTermMemoryStorage: end-of-session L1 flush failed for stream %r; skipping",
                    stream_id,
                )
                continue

            block_id = f"L1-{uuid4().hex[:8]}"
            l1_block = L1MemoryBlock(id=block_id, abstract=abstract, source_text=source_text)
            self._l1_blocks[stream_id].append(l1_block)
            self._pending_l1_for_l2.append(l1_block)
            self._raw_events[stream_id] = []

    async def _maybe_compress_to_l1(self, stream_id: "StreamId") -> None:
        """Compress the oldest conversation turns into an L1 block when the
        raw-event token budget is exceeded."""
        raw_list = list(self._raw_events[stream_id])
        if not raw_list:
            return

        total_text = _events_to_text(raw_list, ignore_tool_names=self._l1_ignore_tool_names)
        if self._token_counter(total_text) <= self._token_threshold:
            return

        turn_starts = [i for i, e in enumerate(raw_list) if isinstance(e, ModelRequest)]
        if len(turn_starts) <= self._recent_turns:
            return  # cannot compress without losing all recent context

        protected = turn_starts[-self._recent_turns]
        raw_list, cutoff = _prune_l1_ignored_tools_from_head_and_rejoin(
            raw_list,
            protected,
            ignore_tool_names=self._l1_ignore_tool_names,
        )
        self._raw_events[stream_id] = raw_list

        total_text = _events_to_text(raw_list, ignore_tool_names=self._l1_ignore_tool_names)
        if self._token_counter(total_text) <= self._token_threshold:
            return

        to_compress = raw_list[:cutoff]
        if not to_compress:
            return

        source_text = _events_to_text(to_compress, ignore_tool_names=self._l1_ignore_tool_names)
        try:
            abstract = await self._summarizer(source_text)
        except Exception:
            logger.exception("LongShortTermMemoryStorage: L1 summarisation failed; skipping compression")
            return

        block_id = f"L1-{uuid4().hex[:8]}"
        l1_block = L1MemoryBlock(id=block_id, abstract=abstract, source_text=source_text)

        self._l1_blocks[stream_id].append(l1_block)
        self._raw_events[stream_id] = raw_list[cutoff:]

        # Accumulate for end-of-session L2 consolidation
        self._pending_l1_for_l2.append(l1_block)

    def _build_memory_header(self, stream_id: "StreamId") -> ModelRequest | None:
        """Build the pinned memory-index message prepended to every context window."""
        core = self._core_memory
        l2 = self._l2_blocks
        l1 = self._l1_blocks[stream_id]

        if not core and not l2 and not l1:
            return None

        lines: list[str] = [
            "## Memory Context",
            "",
            "Your memory is organised into three tiers:",
            "",
            "  Core Memory   — Stable, identity-level facts about the user (name, preferences,",
            "                  goals, constraints). Full content is always shown verbatim below;",
            "                  no lookup is needed.",
            "",
            "  L1 Short-Term — Compressed summaries of recent conversation turns that have",
            "                  scrolled out of the active context window. Each L1 block covers",
            "                  a batch of turns from the current or most recent session.",
            "",
            "  L2 Long-Term  — Cross-session consolidations built from one or more L1 blocks.",
            "                  They persist across conversations and represent older, broader history.",
            "",
            "Each L1/L2 entry below shows: [id] (created <timestamp>) <abstract>",
            f"Each abstract is a retrieval-oriented summary (one or more paragraphs; at most about "
            f"{_DEFAULT_MEMORY_ABSTRACT_MAX_WORDS} words). Call memory_lookup(block_id) to retrieve",
            "the full source text of any block whose abstract looks necessary to the current",
            "request. You can retrieve many times to get enough information. Do NOT infer details beyond what the abstract states without looking up",
            "the block first.",
        ]

        if core:
            lines.append("\n### Core Memory (important identity details)")
            lines.append(core.content)
            lines.append(f"(last updated: {core.updated_at})")

        if l2:
            lines.append("\n### Long-Term Memory — L2 Blocks (cross-session, oldest history)")
            for block in l2:
                lines.append(f"  [{block.id}] (created {block.created_at}) {block.abstract}")

        if l1:
            lines.append("\n### Short-Term Memory — L1 Blocks (recent turns, current session)")
            for block in l1:
                lines.append(f"  [{block.id}] (created {block.created_at}) {block.abstract}")

        return ModelRequest.ensure_request(["\n".join(lines)])

    # ------------------------------------------------------------------
    # JSON persistence for L1 + L2 blocks
    # ------------------------------------------------------------------

    def _load_store(self) -> None:
        """Load L1, L2, and core memory blocks from the JSON store file.

        File format::

            {
              "core_memory": { ...CoreMemoryBlock fields... },   # optional
              "l2_blocks":   [ { ...L2MemoryBlock fields... }, ... ],
              "l1_blocks":   [ { ...L1MemoryBlock fields... }, ... ]
            }

        Legacy files that contain only a bare JSON array are treated as
        L2-only (no persisted L1 blocks, no core memory) for backward
        compatibility.
        """
        if self._l2_store_path is None or not self._l2_store_path.exists():
            return
        try:
            data = json.loads(self._l2_store_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                # Legacy format: plain list of L2 blocks
                self._l2_blocks = [L2MemoryBlock(**{"created_at": "unknown", **item}) for item in data]
                self._persisted_l1_blocks = []
                self._core_memory = None
            else:
                self._l2_blocks = [
                    L2MemoryBlock(**{"created_at": "unknown", **item})
                    for item in data.get("l2_blocks", [])
                ]
                self._persisted_l1_blocks = [
                    L1MemoryBlock(**{"created_at": "unknown", **item})
                    for item in data.get("l1_blocks", [])
                ]
                core_data = data.get("core_memory")
                self._core_memory = (
                    CoreMemoryBlock(**{"updated_at": "unknown", **core_data})
                    if core_data else None
                )
        except Exception:
            logger.exception(
                "LongShortTermMemoryStorage: failed to load store from %s; starting empty",
                self._l2_store_path,
            )

    def _save_store(self) -> None:
        """Persist L2 blocks and their referenced L1 blocks to the JSON store file.

        Only L1 blocks that are referenced by at least one current L2 block are
        written, which prevents the file from growing without bound.  After a
        successful write the in-memory ``_persisted_l1_blocks`` list is updated
        to reflect exactly what is on disk.
        """
        if self._l2_store_path is None:
            return

        # Collect every L1 block ID referenced by any current L2 block
        referenced_ids: set[str] = {lid for l2 in self._l2_blocks for lid in l2.source_blocks}

        # Pool: persisted L1 from previous sessions + new L1 from this session
        all_l1 = (
            list(self._persisted_l1_blocks)
            + [b for blocks in self._l1_blocks.values() for b in blocks]
        )
        # Deduplicate by ID (current session may have re-loaded the same block)
        seen: set[str] = set()
        deduped_l1: list[L1MemoryBlock] = []
        for b in all_l1:
            if b.id not in seen:
                seen.add(b.id)
                deduped_l1.append(b)

        l1_to_save = [b for b in deduped_l1 if b.id in referenced_ids]

        payload: dict = {
            "l2_blocks": [asdict(b) for b in self._l2_blocks],
            "l1_blocks": [asdict(b) for b in l1_to_save],
        }
        if self._core_memory is not None:
            payload["core_memory"] = asdict(self._core_memory)

        try:
            self._l2_store_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # Mirror what is now on disk so the next in-process lookup is consistent
            self._persisted_l1_blocks = l1_to_save
        except Exception:
            logger.exception(
                "LongShortTermMemoryStorage: failed to save store to %s",
                self._l2_store_path,
            )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _prune_l1_ignored_tools_from_event(
    event: BaseEvent,
    *,
    ignore_tool_names: frozenset[str],
) -> BaseEvent | None:
    """Drop or trim *event* so that ignored tool calls/results are removed.

    Returns ``None`` if the event carries no remaining visible content.
    """
    if isinstance(event, ModelResponse):
        calls = event.tool_calls.calls if event.tool_calls else []
        kept_calls = [c for c in calls if c.name not in ignore_tool_names]
        if event.message or kept_calls:
            if len(kept_calls) == len(calls):
                return event
            new_tc = ToolCallsEvent(calls=kept_calls) if kept_calls else ToolCallsEvent()
            return ModelResponse(
                message=event.message,
                tool_calls=new_tc,
                usage=dict(event.usage),
                response_force=event.response_force,
                model=event.model,
                provider=event.provider,
                finish_reason=event.finish_reason,
                images=list(event.images),
            )
        return None

    if isinstance(event, ToolCallsEvent):
        kept = [c for c in event.calls if c.name not in ignore_tool_names]
        if not kept:
            return None
        if len(kept) == len(event.calls):
            return event
        return ToolCallsEvent(calls=kept)

    if isinstance(event, ToolResultsEvent):
        kept = [r for r in event.results if r.name not in ignore_tool_names]
        if not kept:
            return None
        if len(kept) == len(event.results):
            return event
        return ToolResultsEvent(results=kept)

    return event


def _prune_l1_ignored_tools_from_head_and_rejoin(
    raw: list[BaseEvent],
    protected_index: int,
    *,
    ignore_tool_names: frozenset[str],
) -> tuple[list[BaseEvent], int]:
    """Prune ignored-tool events from ``raw[:protected_index]`` only; keep suffix verbatim.

    Returns ``(new_raw, cutoff)`` where *cutoff* is the start index of the
    preserved suffix in *new_raw* (``len(pruned_head)``), suitable for L1
    compression of ``new_raw[:cutoff]``.
    """
    if protected_index <= 0:
        return raw, 0
    head, tail = raw[:protected_index], raw[protected_index:]
    if not ignore_tool_names:
        return head + tail, len(head)
    stripped: list[BaseEvent] = []
    for ev in head:
        pruned = _prune_l1_ignored_tools_from_event(ev, ignore_tool_names=ignore_tool_names)
        if pruned is not None:
            stripped.append(pruned)
    return stripped + tail, len(stripped)


_gpt4o_encoding: tiktoken.Encoding | None = None


def _gpt4o_token_count(text: str) -> int:
    """Count tokens using tiktoken's encoding for ``gpt-4o`` (``o200k_base``)."""
    global _gpt4o_encoding
    if _gpt4o_encoding is None:
        _gpt4o_encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(_gpt4o_encoding.encode(text))


def _char_token_estimate(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


def _model_request_inputs_to_user_text(request: ModelRequest) -> str:
    """Flatten ``ModelRequest.inputs`` to plain text for L1 transcript sizing."""
    chunks: list[str] = []
    for inp in request.inputs:
        if isinstance(inp, TextInput):
            chunks.append(inp.content)
        elif isinstance(inp, (ImageUrlInput, AudioUrlInput, DocumentUrlInput, VideoUrlInput)):
            chunks.append(f"[{type(inp).__name__}: {inp.url}]")
        elif isinstance(inp, FileIdInput):
            fn = f" ({inp.filename})" if inp.filename else ""
            chunks.append(f"[file_id: {inp.file_id}{fn}]")
        elif isinstance(inp, BinaryInput):
            chunks.append(f"[binary {inp.kind}: {inp.media_type}, {len(inp.data)} bytes]")
        else:
            chunks.append(f"[{type(inp).__name__}]")
    return "\n".join(chunks)


def _events_to_text(
    events: Iterable[BaseEvent],
    *,
    ignore_tool_names: frozenset[str] = frozenset(),
) -> str:
    """Render events as a transcript for L1 sizing and summarisation.

    Lines for tools whose names appear in *ignore_tool_names* are omitted so
    bulky ``memory_lookup`` payloads do not inflate the L1 token budget or
    get embedded in new L1 ``source_text``.
    """
    lines: list[str] = []
    for event in events:
        if isinstance(event, ModelRequest):
            lines.append(f"User: {_model_request_inputs_to_user_text(event)}")
        elif isinstance(event, ModelResponse):
            if event.message:
                lines.append(f"Assistant: {event.message.content}")
            if event.tool_calls:
                for call in event.tool_calls.calls:
                    if call.name not in ignore_tool_names:
                        lines.append(f"Assistant calls tool '{call.name}': {call.arguments}")
        elif isinstance(event, ToolCallsEvent):
            for call in event.calls:
                if call.name not in ignore_tool_names:
                    lines.append(f"Assistant calls tool '{call.name}': {call.arguments}")
        elif isinstance(event, ToolResultsEvent):
            for r in event.results:
                if r.name not in ignore_tool_names:
                    lines.append(f"Tool result: {r.content}")
    return "\n".join(lines)
