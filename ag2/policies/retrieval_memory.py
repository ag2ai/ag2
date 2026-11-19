# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""RetrievalMemoryPolicy — injects relevance-retrieved memories, not recency.

Where :class:`~ag2.policies.EpisodicMemoryPolicy` injects the last N summaries
and :class:`~ag2.policies.WorkingMemoryPolicy` injects the whole working-memory
blob, this policy retrieves the chunks relevant to *this* request: the query is
the latest user input (ADR 0027's origin boundary), the candidates come from a
:class:`~ag2.knowledge.retrieval.MemoryRetriever` over the store's memory
index, and the injection is bounded by a token budget, deduplicated per source
file, and attributed to its source path.

Superseded and forgotten facts are never injected — that is the point of the
fact lifecycle — and the index is never trusted alone on that: before
injection, every selected fact's file is re-read and verified against the
store, so an index that lags a status change cannot inject a dead fact. Every
retrieval failure is logged and swallowed: the turn proceeds without injected
memory rather than failing the agent.

One event is published per injection in transparent mode —
:class:`~ag2.events.MemoryInjected` — so evaluation (and any subscriber) can
see exactly what the agent was reminded of. This is the one deliberate
exception to "policies must not touch the stream", matching how
:class:`~ag2.policies.AlertPolicy` publishes its FATAL halt: an injection that
cannot be observed cannot be evaluated.
"""

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from operator import attrgetter
from weakref import WeakKeyDictionary

from ag2.annotations import Context
from ag2.events import BaseEvent, MemoryInjected, MemoryItem, ModelRequest, TextInput, estimated_text_tokens
from ag2.events.input_events import DataInput
from ag2.knowledge import (
    CONVERSATIONS_PREFIX,
    FACTS_PREFIX,
    Fact,
    FactStore,
    KnowledgeStore,
)
from ag2.knowledge.retrieval import (
    Embedder,
    IndexEntry,
    MemoryIndex,
    MemoryRetriever,
    RetrievedMemory,
)

__all__ = ("RetrievalMemoryPolicy",)

logger = logging.getLogger(__name__)

_RETRIEVE_POOL_FACTOR = 2
"""How much larger than ``k_episodes + k_facts`` the retrieval ask is, to
absorb per-path dedup losses before the episode/fact split."""


def _last_user_text(events: list[BaseEvent]) -> str:
    """Text of the latest user request — ADR 0027's origin boundary, not wire role.

    Service-origin requests (inbox drains, framework updates) are not user
    input even though they reach the model as user content. Empty when the
    window holds no user request at all.
    """
    for event in reversed(events):
        if isinstance(event, ModelRequest) and event.origin == "user":
            pieces = [
                part.content if isinstance(part, TextInput) else str(part.data) if isinstance(part, DataInput) else ""
                for part in event.parts
            ]
            return " ".join(piece for piece in pieces if piece).strip()
    return ""


def _injectable(entry: IndexEntry) -> bool:
    """Entries without a dead fact status — episode files carry no status at all."""
    return entry.metadata.get("status") not in ("superseded", "forgotten")


def _alive_fact(fact: Fact | None) -> bool:
    """Whether a re-read fact may be injected: active and not past its expiry.

    ``None`` — the file is gone or unreadable — counts as alive: there is no
    evidence of death, and the indexer prunes orphaned entries on its next sync.
    """
    if fact is None:
        return True
    if fact.status != "active":
        return False
    return fact.expires_at is None or fact.expires_at > datetime.now(UTC)


def _newest_per_subject_kind(selected: list[RetrievedMemory], facts: dict[str, Fact]) -> list[RetrievedMemory]:
    """One fact per ``(subject, kind)`` among the selected — the newest wins.

    A supersede that crashed between its two writes leaves both records
    ``active``; injecting both would present the agent with both sides of a
    contradiction, so the older one is dropped. Facts without a subject are
    never grouped — a subject is what makes two statements about the same
    thing, and unrelated subjectless facts must not crowd each other out.
    """
    newest: dict[tuple[str, str], str] = {}
    doomed: set[str] = set()
    for item in selected:
        fact = facts.get(item.path)
        if fact is None or fact.subject is None:
            continue
        key = (fact.subject, fact.kind)
        incumbent = newest.get(key)
        if incumbent is not None and facts[incumbent].updated_at >= fact.updated_at:
            doomed.add(item.path)
            continue
        if incumbent is not None:
            doomed.add(incumbent)
        newest[key] = item.path
    return [item for item in selected if item.path not in doomed]


def _kind_of(path: str) -> str:
    return "fact" if path.startswith(FACTS_PREFIX) else "episode"


class RetrievalMemoryPolicy:
    """Injects the memories most relevant to the current request.

    Build it around a ready retriever::

        policy = RetrievalMemoryPolicy(MemoryRetriever(index, embedder, indexer))

    or around an embedder plus a per-store index factory, for the common case
    where the index is a :class:`~ag2.knowledge.retrieval.StoreBackedIndex`
    over whatever store the invocation resolved::

        policy = RetrievalMemoryPolicy(
            embedder=HashingEmbedder(),
            index_factory=lambda store: StoreBackedIndex(
                store, embedder_name=embedder.name, dimension=embedder.dimension
            ),
        )

    Candidates are retrieved as one pool, split into at most ``k_facts`` fact
    items and ``k_episodes`` episode items by score, ordered by score, and
    truncated to ``token_budget`` — the first item that does not fit ends the
    injection, and the best item is always injected even when it alone exceeds
    the budget (a target, not a guarantee, like the reduction policies).

    Args:
        retriever: The retriever to query. Exactly one of ``retriever`` and
            ``embedder`` + ``index_factory``.
        embedder: Embeds the query; requires ``index_factory``.
        index_factory: Builds (and caches) the index per resolved store;
            requires ``embedder``.
        k_episodes: At most this many conversation-summary items.
        k_facts: At most this many fact items.
        token_budget: Approximate token budget for the injected block
            (``chars_per_token``-style estimate, as in the token-budget policy).
        min_score: Drop results scoring below this.
        include_superseded: When False (the default), entries whose metadata
            marks them ``superseded`` or ``forgotten`` are never retrieved, and
            every selected fact's file is re-read and verified before injection
            (active, unexpired, no newer active fact on the same subject) — the
            index metadata alone is not trusted. When True, lifecycle filtering
            is skipped entirely: the operator has opted out of it.
        transparent: Attribute every item to its source path, append the
            injection note, and publish :class:`~ag2.events.MemoryInjected`.
        sync_on_apply: Run the retriever's indexer before retrieving — the
            freshly written memory is searchable on the very next turn, at the
            cost of a sync pass per model call.
        per_path: How many chunks of one source file to inject at most.
    """

    name = "retrieval_memory"

    def __init__(
        self,
        retriever: MemoryRetriever | None = None,
        *,
        embedder: Embedder | None = None,
        index_factory: Callable[[KnowledgeStore], MemoryIndex] | None = None,
        k_episodes: int = 3,
        k_facts: int = 8,
        token_budget: int = 1500,
        min_score: float = 0.0,
        include_superseded: bool = False,
        transparent: bool = True,
        sync_on_apply: bool = False,
        per_path: int = 1,
    ) -> None:
        if retriever is None and (embedder is None or index_factory is None):
            raise ValueError("RetrievalMemoryPolicy needs a retriever, or both embedder and index_factory.")
        if retriever is not None and (embedder is not None or index_factory is not None):
            raise ValueError("Pass either retriever or embedder+index_factory, not both.")
        self._retriever = retriever
        self._embedder = embedder
        self._index_factory = index_factory
        # One index per store, weakly held: a policy outlives the stores it
        # served, and the cache must not keep a dead store alive.
        self._indexes: WeakKeyDictionary[KnowledgeStore, MemoryIndex] = WeakKeyDictionary()
        self._k_episodes = k_episodes
        self._k_facts = k_facts
        self._token_budget = token_budget
        self._min_score = min_score
        self._include_superseded = include_superseded
        self._transparent = transparent
        self._sync_on_apply = sync_on_apply
        self._per_path = per_path

    async def apply(
        self,
        prompts: list[str],
        events: list[BaseEvent],
        context: Context,
    ) -> tuple[list[str], list[BaseEvent]]:
        try:
            return await self._apply(prompts, events, context)
        except Exception:
            logger.exception("RetrievalMemoryPolicy failed; injecting nothing")
            return prompts, events

    async def _apply(
        self,
        prompts: list[str],
        events: list[BaseEvent],
        context: Context,
    ) -> tuple[list[str], list[BaseEvent]]:
        store = context.dependencies.get(KnowledgeStore)
        if store is None:
            return prompts, events
        query = _last_user_text(events)
        if not query:
            return prompts, events

        retriever = self._retriever if self._retriever is not None else self._retriever_for(store)
        if self._sync_on_apply:
            await retriever.sync()
        pool_k = (self._k_episodes + self._k_facts) * _RETRIEVE_POOL_FACTOR
        retrieved = await retriever.retrieve(
            query,
            k=pool_k,
            min_score=self._min_score,
            filter=None if self._include_superseded else _injectable,
            per_path=self._per_path,
        )
        selected = self._select(retrieved)
        if not self._include_superseded:
            selected = await self._verified_facts(selected, store)
        if not selected:
            return prompts, events

        block = self._block(selected)
        prompts = prompts + [block]
        if self._transparent:
            prompts = prompts + [self._note(selected)]
            with suppress(Exception):
                await context.send(
                    MemoryInjected(
                        items=[
                            MemoryItem(path=item.path, score=item.score, kind=_kind_of(item.path)) for item in selected
                        ],
                        query=query,
                    )
                )
        return prompts, events

    def _retriever_for(self, store: KnowledgeStore) -> MemoryRetriever:
        """The retriever for one store, building and caching its index on first use."""
        index = self._indexes.get(store)
        if index is None:
            index = self._index_factory(store)
            self._indexes[store] = index
        return MemoryRetriever(index, self._embedder)

    async def _verified_facts(self, selected: list[RetrievedMemory], store: KnowledgeStore) -> list[RetrievedMemory]:
        """Re-read the fact files behind the selection; drop the dead and the contradicted.

        The index's metadata can lag the store — a status change between syncs
        leaves a stale entry — so facts are never trusted from the index alone
        (defense in depth over the indexer's metadata refresh). Each selected
        fact's file is read once, bounded by the selection size:

        * a fact whose file now reads ``superseded``/``forgotten``, or whose
          ``expires_at`` has passed, is dropped;
        * a fact whose file is gone or unreadable has no evidence of death and
          stays — the indexer prunes such orphans on its next sync;
        * of the survivors, two still-active facts sharing a subject and kind
          collapse to the newest ``updated_at`` — the readable half of a
          supersede that crashed between its two writes must not inject both
          sides.
        """
        fact_items = [item for item in selected if item.path.startswith(FACTS_PREFIX)]
        if not fact_items:
            return selected
        try:
            facts = await self._read_facts(FactStore(store), fact_items)
        except Exception:
            logger.warning("Fact verification read failed; injecting only episodes", exc_info=True)
            return [item for item in selected if not item.path.startswith(FACTS_PREFIX)]
        survivors = [item for item in selected if _alive_fact(facts.get(item.path))]
        return _newest_per_subject_kind(survivors, facts)

    async def _read_facts(self, fact_store: FactStore, items: list[RetrievedMemory]) -> dict[str, Fact]:
        """Read each selected fact's file once, keyed by store path."""
        paths: dict[str, str] = {}
        for item in items:
            fact_id = fact_store.fact_id(item.path)
            if fact_id is not None:
                paths[fact_id] = item.path
        facts = await asyncio.gather(*(fact_store.get(fact_id) for fact_id in paths))
        return {path: fact for fact, path in zip(facts, paths.values(), strict=True) if fact is not None}

    def _select(self, retrieved: list[RetrievedMemory]) -> list[RetrievedMemory]:
        """Split the pool into its episode and fact quotas, then order by score."""
        episodes = [item for item in retrieved if item.path.startswith(CONVERSATIONS_PREFIX)][: self._k_episodes]
        facts = [item for item in retrieved if item.path.startswith(FACTS_PREFIX)][: self._k_facts]
        return sorted([*episodes, *facts], key=attrgetter("score"), reverse=True)

    def _block(self, selected: list[RetrievedMemory]) -> str:
        """The injected block, truncated to the token budget by score order.

        Prefix semantics: items are taken in score order until one does not
        fit. The best item is always taken even when it alone exceeds the
        budget — the budget is a target, not a guarantee, the same contract
        the reduction policies keep.
        """
        header = "## Retrieved Memory"
        used = estimated_text_tokens(header)
        lines: list[str] = []
        for item in selected:
            line = self._line(item)
            cost = estimated_text_tokens(line) + 1  # the newline that joins it
            if used + cost > self._token_budget and lines:
                break
            used += cost
            lines.append(line)
        return "\n".join([header, *lines])

    def _line(self, item: RetrievedMemory) -> str:
        text = item.text.strip()
        if self._transparent:
            return f"- {text} (source: {item.path}, score {item.score:.2f})"
        return f"- {text}"

    def _note(self, selected: list[RetrievedMemory]) -> str:
        facts = sum(1 for item in selected if item.path.startswith(FACTS_PREFIX))
        episodes = len(selected) - facts
        return f"[{self.name}] Injected {len(selected)} memories ({facts} facts, {episodes} episodes)."
