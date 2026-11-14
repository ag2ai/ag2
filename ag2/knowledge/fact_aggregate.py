# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Fact extraction and reconciliation — the write side of the fact lifecycle.

:class:`FactExtractionAggregate` is an :class:`~ag2.aggregate.AggregateStrategy`
(triggered like the built-in aggregates) that turns conversation into candidate
facts, reconciles each candidate against the similar facts already stored, and
writes the outcome through :class:`~ag2.knowledge.FactStore`:

* **add** — new information;
* **update** — a refinement of an existing fact on the same subject;
* **supersede** — a contradiction, recorded as a new fact linked to the one it
  replaces (the old record is kept, never injected again);
* **noop** — a duplicate of what is already stored.

Two extractors: the model path (one structured-output call per aggregation) and
a :class:`StaticFactExtractor` plug-in that replaces the model entirely —
tests and deterministic pipelines use the second.
"""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable
from weakref import WeakKeyDictionary

from fast_depends.pydantic import PydanticSerializer
from pydantic import BaseModel, Field

from ag2.aggregate import _emit_aggregation_usage
from ag2.annotations import Context
from ag2.events import BaseEvent, ModelRequest, render_for_prompt
from ag2.knowledge.base import KnowledgeStore
from ag2.knowledge.facts import Fact, FactKind, FactSource, FactStore
from ag2.knowledge.retrieval import KnowledgeIndexer, MemoryRetriever
from ag2.knowledge.retrieval.index import IndexEntry
from ag2.response import ResponseSchema
from ag2.stream import MemoryStream

if TYPE_CHECKING:
    from ag2.config import ModelConfig

__all__ = (
    "CandidateMatch",
    "ExtractedFact",
    "ExtractedFacts",
    "ExtractionReport",
    "FactDecision",
    "FactExtractionAggregate",
    "FactReconciler",
    "HeuristicReconciler",
    "ModelReconciler",
    "ScoredFact",
    "StaticFactExtractor",
)

logger = logging.getLogger(__name__)

DEFAULT_FACT_EXTRACTION_PROMPT = (
    "Extract durable facts from this conversation — preferences, standing "
    "instructions, events, and anything worth remembering in later sessions. "
    "State each fact as one standalone sentence, name its subject when there "
    "is one, and set an expiry only for facts that are explicitly time-bound. "
    "Return nothing when the conversation produced nothing durable.\n\n"
    "## Conversation\n{events}"
)
"""Default prompt template for :class:`FactExtractionAggregate`.

The ``{events}`` placeholder receives a textual rendering of the new
conversation. Pass a custom template via the ``prompt`` keyword argument;
omitting the placeholder skips the conversation entirely.
"""

_RECONCILIATION_PROMPT = (
    "For each numbered candidate fact, decide how to reconcile it with the "
    "existing facts listed beneath it:\n"
    "- add: new information, nothing similar is stored\n"
    "- update: a refinement of an existing fact on the same subject "
    "(give existing_id)\n"
    "- supersede: contradicts an existing fact and replaces it (give existing_id)\n"
    "- noop: a duplicate of an existing fact (give existing_id)\n"
    "Decide for every candidate index.\n\n{candidates}"
)

_MATCH_K = 4
"""How many similar existing facts each candidate is reconciled against."""

_MATCH_MIN_SCORE = 0.1
"""Similarity floor below which a stored fact is not considered a match at all."""

_MAX_TEXT_CHARS = 400
"""Candidate and existing texts are truncated to this length in the prompt."""


class ExtractedFact(BaseModel):
    """One candidate fact as an extractor reports it, before it has an identity."""

    text: str = Field(description="The fact, stated as one standalone sentence.")
    subject: str | None = Field(default=None, description="The entity the fact is about, when there is one.")
    kind: FactKind = Field(default="fact", description="What kind of statement this is.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="How sure the extractor is.")
    expires_at: str | None = Field(
        default=None, description="ISO-8601 datetime after which the fact no longer holds, when known."
    )


class ExtractedFacts(BaseModel):
    """The extraction call's structured output — the batch of candidate facts."""

    facts: list[ExtractedFact] = Field(
        default_factory=list, description="Every fact worth keeping, without duplicates."
    )


@dataclass(frozen=True, slots=True)
class ScoredFact:
    """An existing stored fact paired with its similarity to a candidate."""

    fact: Fact
    score: float


@dataclass(frozen=True, slots=True)
class CandidateMatch:
    """One candidate fact together with the similar active facts already stored."""

    candidate: ExtractedFact
    matches: tuple[ScoredFact, ...] = ()


ReconcileAction = Literal["add", "update", "supersede", "noop"]


@dataclass(frozen=True, slots=True)
class FactDecision:
    """What a reconciler decided to do with one candidate fact.

    ``existing`` is the stored fact the action applies to — always set for
    ``update`` / ``supersede`` / ``noop`` decisions that name a real fact.
    """

    candidate: ExtractedFact
    action: ReconcileAction
    existing: Fact | None = None
    reason: str = ""


@runtime_checkable
class FactReconciler(Protocol):
    """Decides, per candidate, how it reconciles with the stored facts.

    Implementations that call a model emit their usage onto ``context`` (the
    aggregate's real stream) the way the aggregation strategies do, and record
    ``last_usage`` so ``AggregationCompleted`` can report the call.
    """

    async def reconcile(self, candidates: Sequence[CandidateMatch], context: Context) -> list[FactDecision]:
        """One :class:`FactDecision` per candidate, in input order."""
        ...


@runtime_checkable
class StaticFactExtractor(Protocol):
    """A deterministic candidate-fact source, plugged in place of the model call.

    Implement this to extract facts without an LLM — from parsed tool output,
    a rules pass, or a test fixture.
    """

    async def extract(self, events: Sequence[BaseEvent], store: KnowledgeStore) -> list[ExtractedFact]:
        """Return the candidate facts visible in this conversation."""
        ...


class HeuristicReconciler:
    """Reconciles candidates from retrieval similarity alone — no model call.

    For each candidate, against its best-matching stored fact:

    * similarity at or above ``dup_threshold`` — **noop** (duplicate);
    * same ``subject`` and ``kind`` at or above ``update_threshold`` —
      **update** (a refinement of what is stored);
    * same ``subject`` and ``kind`` below ``update_threshold`` — **supersede**
      (a contradiction about the same subject);
    * otherwise — **add**.

    The thresholds are heuristics over embedding cosine; when they are not
    good enough, a :class:`ModelReconciler` judges the same matches instead.

    Args:
        dup_threshold: Similarity at which a candidate counts as a duplicate.
        update_threshold: Similarity at which a same-subject candidate counts
            as a refinement rather than a contradiction.
    """

    def __init__(self, *, dup_threshold: float = 0.92, update_threshold: float = 0.6) -> None:
        self._dup_threshold = dup_threshold
        self._update_threshold = update_threshold
        self.last_usage: dict = {}

    async def reconcile(self, candidates: Sequence[CandidateMatch], context: Context) -> list[FactDecision]:
        del context  # no model call — nothing to account
        return [self._decide(match) for match in candidates]

    def _decide(self, match: CandidateMatch) -> FactDecision:
        candidate = match.candidate
        for scored in match.matches:
            if scored.score >= self._dup_threshold:
                return FactDecision(
                    candidate,
                    "noop",
                    scored.fact,
                    f"duplicate of {scored.fact.id} (similarity {scored.score:.2f})",
                )
        same_subject = next((scored for scored in match.matches if _same_subject_kind(candidate, scored.fact)), None)
        if same_subject is not None:
            if same_subject.score >= self._update_threshold:
                return FactDecision(
                    candidate, "update", same_subject.fact, f"refines {same_subject.fact.id} on the same subject"
                )
            return FactDecision(
                candidate, "supersede", same_subject.fact, f"contradicts {same_subject.fact.id} on the same subject"
            )
        return FactDecision(candidate, "add", None, "no similar active fact")


class _ReconcileDecision(BaseModel):
    """One decision inside the reconciliation call's structured output."""

    index: int = Field(description="The candidate's number in the prompt.")
    action: Literal["add", "update", "supersede", "noop"]
    existing_id: str | None = Field(default=None, description="The existing fact the action applies to; None for add.")
    reason: str = Field(default="", description="A brief justification.")


class ReconciliationPlan(BaseModel):
    """The reconciliation call's structured output — one decision per candidate."""

    decisions: list[_ReconcileDecision] = Field(default_factory=list)


class ModelReconciler:
    """Decides a whole batch of candidates with one structured-output call.

    On any model or parse failure the batch falls back to
    :class:`HeuristicReconciler` — per candidate when only some decisions are
    unusable, wholesale when the call itself failed. Every failure is recorded
    in :attr:`failures`; candidates are never dropped.

    Args:
        config: Model config for the reconciliation call.
        fallback: Reconciler used when the model path fails; a default
            :class:`HeuristicReconciler` when omitted.
    """

    def __init__(self, config: "ModelConfig", *, fallback: HeuristicReconciler | None = None) -> None:
        self._config = config
        self._fallback = fallback or HeuristicReconciler()
        self._serializer = PydanticSerializer(
            pydantic_config={"arbitrary_types_allowed": True},
            use_fastdepends_errors=False,
        )
        self._schema = ResponseSchema(ReconciliationPlan)
        self.failures: list[str] = []
        self.last_usage: dict = {}

    async def reconcile(self, candidates: Sequence[CandidateMatch], context: Context) -> list[FactDecision]:
        if not candidates:
            return []
        try:
            plan = await self._ask(candidates, context)
        except Exception as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
            return await self._fallback.reconcile(candidates, context)

        decisions: list[FactDecision | None] = [None] * len(candidates)
        undecided: list[int] = []
        for index, match in enumerate(candidates):
            decision = self._from_plan(plan, index, match)
            if decision is None:
                undecided.append(index)
            else:
                decisions[index] = decision
        if undecided:
            fallback = await self._fallback.reconcile([candidates[index] for index in undecided], context)
            for position, index in enumerate(undecided):
                decisions[index] = fallback[position]
        return [decision for decision in decisions if decision is not None]

    async def _ask(self, candidates: Sequence[CandidateMatch], context: Context) -> ReconciliationPlan:
        client = self._config.create()
        prompt = _RECONCILIATION_PROMPT.format(candidates=_render_candidates(candidates))
        response = await client(
            [ModelRequest.ensure_request([prompt])],
            Context(MemoryStream()),
            tools=[],
            response_schema=self._schema,
            serializer=self._serializer,
        )
        self.last_usage = response.usage if getattr(response, "usage", None) else {}
        await _emit_aggregation_usage(response, context)
        return await self._schema.validate(response.content or "", context)

    def _from_plan(self, plan: ReconciliationPlan, index: int, match: CandidateMatch) -> FactDecision | None:
        decision = next((d for d in plan.decisions if d.index == index), None)
        if decision is None:
            return None
        if decision.action == "add" and decision.existing_id is None:
            return FactDecision(match.candidate, "add", None, decision.reason)
        existing = next((scored.fact for scored in match.matches if scored.fact.id == decision.existing_id), None)
        if existing is not None and decision.action in ("update", "supersede", "noop"):
            return FactDecision(match.candidate, decision.action, existing, decision.reason)
        return None


@dataclass(frozen=True, slots=True)
class ExtractionReport:
    """Counts from the last :meth:`FactExtractionAggregate.aggregate` run.

    Attributes:
        added: Candidates written as new facts.
        updated: Existing facts rewritten with refined text.
        superseded: Existing facts replaced by a new linked fact.
        noop: Candidates dropped as duplicates.
        failed: Candidates whose write failed — reported, never silently lost
            (a later run reconciles them again).
    """

    added: int = 0
    updated: int = 0
    superseded: int = 0
    noop: int = 0
    failed: int = 0


class FactExtractionAggregate:
    """Extracts durable facts from conversation and reconciles them into the store.

    Satisfies :class:`~ag2.aggregate.AggregateStrategy` — wire it through
    ``KnowledgePlugin(aggregate=FactExtractionAggregate(...), aggregate_trigger=...)``.
    Candidates beyond ``max_facts_per_run`` are dropped for that run; applied
    decisions are counted in :attr:`last_report`, and the call's tokens are
    reported in :attr:`last_usage`.

    Concurrent ``aggregate`` calls on one store are serialized: each store
    gets an ``asyncio.Lock`` (weakly held on the instance) held across
    match → reconcile → apply → sync, so two runs cannot both reconcile
    against the same stored facts and both add near-duplicates. **Share one
    aggregate instance** across the invocations that write the same store —
    the lock lives on the instance, so separately built aggregates do not
    synchronize with each other.

    When ``indexer`` is given, candidates are matched against the stored facts
    through the indexer's own index (the same one retrieval reads), and the
    index is synced after the writes. Without an indexer every candidate sees
    no matches, so reconciliation degrades to add-or-noop against nothing —
    usable, but no duplicate or contradiction is detected.

    Args:
        config: Model config for extraction and (by default) reconciliation.
            Required unless ``extractor`` is provided.
        reconciler: Decides add/update/supersede/noop; defaults to
            :class:`HeuristicReconciler`, and to the config's
            :class:`ModelReconciler` when a config is given.
        indexer: The indexer whose index and embedder match candidates against
            stored facts, and which is synced after the writes.
        extractor: A :class:`StaticFactExtractor` replacing the model call —
            for deterministic pipelines and tests. When given, ``config`` is
            unused unless a :class:`ModelReconciler` was passed explicitly.
        max_facts_per_run: Upper bound on candidates extracted per run.
        prompt: Extraction prompt template with an ``{events}`` placeholder.

    Raises:
        ValueError: If neither a config nor an extractor is provided.
    """

    def __init__(
        self,
        config: "ModelConfig | None" = None,
        *,
        reconciler: FactReconciler | None = None,
        indexer: KnowledgeIndexer | None = None,
        extractor: StaticFactExtractor | None = None,
        max_facts_per_run: int = 20,
        prompt: str = DEFAULT_FACT_EXTRACTION_PROMPT,
    ) -> None:
        if extractor is None and config is None:
            raise ValueError(
                "FactExtractionAggregate needs config (model extraction) or extractor (static extraction)."
            )
        self._config = config
        self._extractor = extractor
        self._reconciler = reconciler or (ModelReconciler(config) if config is not None else HeuristicReconciler())
        self._indexer = indexer
        self._retriever = MemoryRetriever(indexer.index, indexer.embedder) if indexer is not None else None
        self._max_facts = max(1, max_facts_per_run)
        self._prompt = prompt
        self._serializer = PydanticSerializer(
            pydantic_config={"arbitrary_types_allowed": True},
            use_fastdepends_errors=False,
        )
        self._schema = ResponseSchema(ExtractedFacts)
        self.last_usage: dict = {}
        self.last_report = ExtractionReport()
        # One lock per store, weakly held: the aggregate outlives the stores it
        # served, and the registry must not keep a dead store alive.
        self._store_locks: WeakKeyDictionary[KnowledgeStore, asyncio.Lock] = WeakKeyDictionary()

    async def aggregate(
        self,
        events: list[BaseEvent],
        context: Context,
        store: KnowledgeStore,
    ) -> None:
        if not events:
            return
        fact_store = FactStore(store)
        candidates = await self._extract(events, context, store)
        candidates = candidates[: self._max_facts]
        # Serialized per store: a concurrent run must reconcile against what
        # this run already wrote, not against the same pre-write snapshot.
        async with self._lock_for(store):
            matches = await self._matches(candidates, fact_store)
            decisions = await self._reconciler.reconcile(matches, context)
            if not self.last_usage:
                # A static extractor makes no model call; adopt the reconciler's
                # usage so the completion event still reports the run's LLM work.
                self.last_usage = getattr(self._reconciler, "last_usage", {})

            now = datetime.now(UTC)
            counts = {"added": 0, "updated": 0, "superseded": 0, "noop": 0, "failed": 0}
            for decision in decisions:
                try:
                    counts[await self._apply(decision, fact_store, context, events, now)] += 1
                except Exception as exc:
                    counts["failed"] += 1
                    logger.warning("Applying fact decision failed (%s): %s", decision.action, exc)
            self.last_report = ExtractionReport(**counts)

            if self._indexer is not None:
                await self._indexer.sync()

    def _lock_for(self, store: KnowledgeStore) -> asyncio.Lock:
        """The aggregate's reconcile+apply lock for one store, created on first use."""
        lock = self._store_locks.get(store)
        if lock is None:
            lock = asyncio.Lock()
            self._store_locks[store] = lock
        return lock

    async def _extract(self, events: list[BaseEvent], context: Context, store: KnowledgeStore) -> list[ExtractedFact]:
        if self._extractor is not None:
            return list(await self._extractor.extract(events, store))
        return await self._extract_with_model(events, context)

    async def _extract_with_model(self, events: list[BaseEvent], context: Context) -> list[ExtractedFact]:
        client = self._config.create()
        prompt = self._prompt.format(events="\n".join(render_for_prompt(event) for event in events))
        response = await client(
            [ModelRequest.ensure_request([prompt])],
            Context(MemoryStream()),
            tools=[],
            response_schema=self._schema,
            serializer=self._serializer,
        )
        self.last_usage = response.usage if getattr(response, "usage", None) else {}
        await _emit_aggregation_usage(response, context)
        extracted = await self._schema.validate(response.content or "", context)
        return extracted.facts

    async def _matches(self, candidates: Sequence[ExtractedFact], fact_store: FactStore) -> list[CandidateMatch]:
        """Retrieve each candidate's similar active facts through the shared index."""
        if self._retriever is None:
            return [CandidateMatch(candidate, ()) for candidate in candidates]
        matches: list[CandidateMatch] = []
        for candidate in candidates:
            retrieved = await self._retriever.retrieve(
                candidate.text,
                k=_MATCH_K,
                min_score=_MATCH_MIN_SCORE,
                filter=_injectable_entry,
            )
            scored: list[ScoredFact] = []
            for item in retrieved:
                fact_id = fact_store.fact_id(item.path)
                fact = await fact_store.get(fact_id) if fact_id is not None else None
                if fact is not None and fact.status == "active":
                    scored.append(ScoredFact(fact, item.score))
            matches.append(CandidateMatch(candidate, tuple(scored)))
        return matches

    async def _apply(
        self,
        decision: FactDecision,
        fact_store: FactStore,
        context: Context,
        events: Sequence[BaseEvent],
        now: datetime,
    ) -> Literal["added", "updated", "superseded", "noop"]:
        if decision.action == "add":
            await fact_store.put(self._new_fact(decision.candidate, context, events, ()))
            return "added"
        if decision.existing is None:
            logger.warning("Fact decision %s has no existing fact to apply to; treating as noop", decision.action)
            return "noop"
        if decision.action == "update":
            existing = decision.existing
            existing.text = decision.candidate.text
            existing.confidence = decision.candidate.confidence
            existing.expires_at = _parse_expires_at(decision.candidate.expires_at)
            existing.updated_at = now
            await fact_store.put(existing)
            return "updated"
        if decision.action == "supersede":
            replacement = self._new_fact(decision.candidate, context, events, (decision.existing.id,))
            await fact_store.supersede(decision.existing.id, replacement)
            return "superseded"
        return "noop"

    def _new_fact(
        self, candidate: ExtractedFact, context: Context, events: Sequence[BaseEvent], supersedes: Sequence[str]
    ) -> Fact:
        return Fact(
            text=candidate.text,
            subject=candidate.subject,
            kind=candidate.kind,
            confidence=candidate.confidence,
            expires_at=_parse_expires_at(candidate.expires_at),
            source=FactSource(stream_id=str(context.stream.id), event_ids=_source_event_ids(events)),
            supersedes=list(supersedes),
        )


def _same_subject_kind(candidate: ExtractedFact, fact: Fact) -> bool:
    return candidate.subject is not None and candidate.subject == fact.subject and candidate.kind == fact.kind


def _injectable_entry(entry: IndexEntry) -> bool:
    """Only entries that do not carry a dead fact status — episodes have none."""
    return entry.metadata.get("status") not in ("superseded", "forgotten")


def _render_candidates(candidates: Sequence[CandidateMatch]) -> str:
    lines: list[str] = []
    for index, match in enumerate(candidates):
        candidate = match.candidate
        lines.append(f"[{index}] ({candidate.kind}, subject={candidate.subject!r}) {candidate.text}")
        for scored in match.matches:
            text = (
                scored.fact.text
                if len(scored.fact.text) <= _MAX_TEXT_CHARS
                else f"{scored.fact.text[:_MAX_TEXT_CHARS]}…"
            )
            lines.append(f"    existing {scored.fact.id} ({scored.fact.kind}, similarity {scored.score:.2f}): {text}")
    return "\n".join(lines)


def _parse_expires_at(raw: str | None) -> datetime | None:
    """Parse an extracted expiry; unparseable values are dropped with a warning."""
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        logger.warning("Ignoring unparseable expires_at %r", raw)
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _source_event_ids(events: Sequence[BaseEvent]) -> list[str]:
    """Ids of the events a fact was extracted from, where the events carry one."""
    ids: list[str] = []
    for event in events:
        for attr in ("event_id", "response_id", "id"):
            value = getattr(event, attr, None)
            if isinstance(value, str):
                ids.append(value)
                break
    return ids
