# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Harness feature smoke: assembly policies, knowledge store, compaction,
aggregation — every opt-in Actor primitive exercised against a real LLM.

Uses Gemini 3 Flash Preview as the default driver (fast, cheap, capable).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from autogen.beta import Actor, KnowledgeConfig
from autogen.beta.aggregate import (
    AggregateTrigger,
    ConversationSummaryAggregate,
    WorkingMemoryAggregate,
)
from autogen.beta.compact import (
    CompactTrigger,
    SummarizeCompact,
    TailWindowCompact,
)
from autogen.beta.events.lifecycle import (
    AggregationCompleted,
    CompactionCompleted,
)
from autogen.beta.knowledge import (
    DefaultBootstrap,
    DiskKnowledgeStore,
    MemoryKnowledgeStore,
)
from autogen.beta.policies import (
    ConversationPolicy,
    EpisodicMemoryPolicy,
    SlidingWindowPolicy,
    TokenBudgetPolicy,
    WorkingMemoryPolicy,
)
from autogen.beta.stream import MemoryStream


pytestmark = [pytest.mark.asyncio, pytest.mark.gemini]


# ---------------------------------------------------------------------------
# Assembly policies
# ---------------------------------------------------------------------------


async def test_conversation_policy_basic(gemini_flash_config) -> None:
    """ConversationPolicy filters stream to conversation + tool events.

    The agent should still produce a sensible reply when the policy is active.
    """
    agent = Actor(
        "conv",
        prompt="Be concise.",
        config=gemini_flash_config,
        assembly=[ConversationPolicy()],
    )
    reply = await agent.ask("Say 'hello'.")
    assert reply.body is not None
    assert "hello" in reply.body.lower()


async def test_sliding_window_trims_long_history(gemini_flash_config) -> None:
    """SlidingWindowPolicy trims history to the most-recent N events.

    We stuff history, then ask for a recall and expect the model NOT to recall
    ancient turns because the window trimmed them out.
    """
    # Window of 4 conversation events = ~2 turns
    agent = Actor(
        "sliding",
        prompt="Be concise. Only answer from what's in the conversation history.",
        config=gemini_flash_config,
        assembly=[ConversationPolicy(), SlidingWindowPolicy(max_events=4)],
    )

    r1 = await agent.ask("Remember the word 'elephant'.")
    r2 = await r1.ask("Remember the word 'volcano'.")
    r3 = await r2.ask("Remember the word 'nebula'.")
    # After r3, history has ~6 ModelRequests/Responses. With max_events=4,
    # the sliding window drops the oldest turns — so 'elephant' should be gone.
    r4 = await r3.ask("Which three words did I ask you to remember? Be precise.")
    assert r4.body is not None
    body = r4.body.lower()
    # Most-recent word must survive
    assert "nebula" in body
    # First-turn word must be evicted by the sliding window —
    # this is the assertion that proves the policy actually trimmed history.
    assert "elephant" not in body


async def test_token_budget_policy_clamps_history(gemini_flash_config) -> None:
    """TokenBudgetPolicy with a very tight budget evicts older turns.

    Build up a multi-turn history of fruits, then ask the agent to recall
    them. With ``max_tokens=5`` the prompt sent to the LLM cannot include the
    earlier turns, so the model cannot produce the fruit list. The control
    (no policy) is verified separately to keep this test cheap.
    """
    agent = Actor(
        "budget",
        prompt="One-word answers.",
        config=gemini_flash_config,
        assembly=[ConversationPolicy(), TokenBudgetPolicy(max_tokens=5)],
    )
    r = await agent.ask("Say 'apple'.")
    r = await r.ask("Say 'banana'.")
    r = await r.ask("Say 'cherry'.")
    r = await r.ask("Say 'date'.")
    final = await r.ask("List every fruit I asked you about previously, comma separated.")
    assert final.body is not None
    body = final.body.lower()
    # Tight budget must have evicted the earlier turns from the LLM's view —
    # the model cannot list fruits it never saw.
    earlier_fruits = ("apple", "banana", "cherry", "date")
    assert not any(f in body for f in earlier_fruits), (
        f"TokenBudgetPolicy(max_tokens=5) failed to trim — model recalled fruits: {body!r}"
    )


async def test_multiple_policies_compose(gemini_flash_config) -> None:
    """Composed policies: Conversation → SlidingWindow → TokenBudget."""
    agent = Actor(
        "composed",
        prompt="Brief.",
        config=gemini_flash_config,
        assembly=[
            ConversationPolicy(),
            SlidingWindowPolicy(max_events=20),
            TokenBudgetPolicy(max_tokens=2000),
        ],
    )
    reply = await agent.ask("What is 1+1? Number only.")
    assert reply.body is not None
    assert "2" in reply.body


# ---------------------------------------------------------------------------
# Knowledge store
# ---------------------------------------------------------------------------


async def test_memory_knowledge_store_crud() -> None:
    """Direct KnowledgeStore CRUD — no LLM needed."""
    store = MemoryKnowledgeStore()
    await store.write("/notes/alpha.md", "content-alpha")
    await store.write("/notes/beta.md", "content-beta")

    assert await store.read("/notes/alpha.md") == "content-alpha"
    assert "alpha.md" in (await store.list("/notes"))
    assert "beta.md" in (await store.list("/notes"))

    await store.delete("/notes/alpha.md")
    assert await store.read("/notes/alpha.md") is None
    assert "alpha.md" not in (await store.list("/notes"))


async def test_disk_knowledge_store_persists(tmp_path: Path) -> None:
    root = tmp_path / "kstore"
    store1 = DiskKnowledgeStore(str(root))
    await store1.write("/hello.txt", "world")
    assert await store1.read("/hello.txt") == "world"

    # Re-open: data persists
    store2 = DiskKnowledgeStore(str(root))
    assert await store2.read("/hello.txt") == "world"


async def test_knowledge_tool_via_actor(gemini_flash_config) -> None:
    """LLM drives the knowledge tool: write a note, list, read it back."""
    store = MemoryKnowledgeStore()

    agent = Actor(
        "knower",
        prompt=(
            "You have a `knowledge` tool with actions: read, write, list, delete. "
            "Paths are slash-separated. Always use the tool when the user asks you "
            "to remember or recall things."
        ),
        config=gemini_flash_config,
        knowledge=KnowledgeConfig(store=store),
    )

    r1 = await agent.ask(
        "Please remember that my favourite animal is a red panda. "
        "Store this at /preferences/animal.md"
    )
    assert r1.body is not None

    # Direct store inspection — the tool should have written
    content = await store.read("/preferences/animal.md")
    assert content is not None
    assert "red panda" in content.lower()

    # Ask the agent to recall via the tool
    r2 = await agent.ask(
        "What is my favourite animal? Look it up in /preferences/animal.md and tell me."
    )
    assert r2.body is not None
    assert "red panda" in r2.body.lower()


async def test_bootstrap_runs_once(gemini_flash_config) -> None:
    """DefaultBootstrap writes initial store layout on first _execute."""
    store = MemoryKnowledgeStore()

    # Pre-condition: store is empty
    assert await store.exists("/.initialized") is False

    agent = Actor(
        "bootstrapped",
        config=gemini_flash_config,
        knowledge=KnowledgeConfig(store=store, bootstrap=DefaultBootstrap()),
    )
    await agent.ask("Say 'ok'.")

    # Post-condition: sentinel was written
    assert await store.exists("/.initialized") is True


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


async def test_tail_window_compact_triggers(gemini_flash_config) -> None:
    """TailWindowCompact (non-LLM) triggers after enough events."""
    store = MemoryKnowledgeStore()

    compact_events: list = []
    stream = MemoryStream()
    stream.where(CompactionCompleted).subscribe(lambda e: compact_events.append(e))

    agent = Actor(
        "compactor",
        prompt="Reply in one short sentence.",
        config=gemini_flash_config,
        knowledge=KnowledgeConfig(
            store=store,
            compact=TailWindowCompact(target=3),
            compact_trigger=CompactTrigger(max_events=4),
        ),
    )

    # Burn through enough turns to trigger compaction
    r = await agent.ask("Turn one: say 'one'.", stream=stream)
    r = await r.ask("Turn two: say 'two'.")
    r = await r.ask("Turn three: say 'three'.")
    r = await r.ask("Turn four: say 'four'.")

    # After ~4 turns we should have seen at least one compaction
    assert len(compact_events) >= 1
    evt = compact_events[0]
    assert evt.events_before > evt.events_after
    assert evt.strategy == "TailWindowCompact"


async def test_summarize_compact_uses_llm(gemini_flash_config) -> None:
    """SummarizeCompact uses a secondary LLM call to summarize history."""
    store = MemoryKnowledgeStore()

    compact_events: list = []
    stream = MemoryStream()
    stream.where(CompactionCompleted).subscribe(lambda e: compact_events.append(e))

    agent = Actor(
        "summarizer",
        prompt="Very short answers.",
        config=gemini_flash_config,
        knowledge=KnowledgeConfig(
            store=store,
            compact=SummarizeCompact(target=2, config=gemini_flash_config),
            compact_trigger=CompactTrigger(max_events=4),
        ),
    )

    r = await agent.ask("Say 'a'.", stream=stream)
    r = await r.ask("Say 'b'.")
    r = await r.ask("Say 'c'.")
    r = await r.ask("Say 'd'.")

    assert len(compact_events) >= 1
    evt = compact_events[0]
    assert evt.strategy == "SummarizeCompact"
    assert evt.llm_calls >= 1  # Compaction used the LLM


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


async def test_on_end_aggregation(gemini_flash_config) -> None:
    """AggregateTrigger(on_end=True) fires at execute finalisation."""
    store = MemoryKnowledgeStore()

    aggregate_events: list = []
    stream = MemoryStream()
    stream.where(AggregationCompleted).subscribe(lambda e: aggregate_events.append(e))

    agent = Actor(
        "aggregator",
        prompt="Brief answer.",
        config=gemini_flash_config,
        knowledge=KnowledgeConfig(
            store=store,
            aggregate=ConversationSummaryAggregate(config=gemini_flash_config),
            aggregate_trigger=AggregateTrigger(on_end=True),
        ),
    )

    await agent.ask("Tell me one fact about the moon.", stream=stream)

    assert len(aggregate_events) >= 1
    evt = aggregate_events[0]
    assert evt.strategy == "ConversationSummaryAggregate"
    assert evt.event_count > 0

    # The aggregator should have written to the knowledge store
    listing = await store.list("/")
    assert len(listing) > 0


async def test_every_n_turns_aggregation(gemini_flash_config) -> None:
    """AggregateTrigger(every_n_turns=2) fires aggregation every 2 user asks.

    A "turn" is counted as a :class:`ModelRequest` event in stream history
    — one per user ``.ask()``. The trigger is stateless: the middleware
    re-derives turn count from history on every call, so counters survive
    the per-execute middleware rebuild.
    """
    store = MemoryKnowledgeStore()

    aggregate_events: list = []
    stream = MemoryStream()
    stream.where(AggregationCompleted).subscribe(lambda e: aggregate_events.append(e))

    agent = Actor(
        "periodic",
        prompt="One word answers.",
        config=gemini_flash_config,
        knowledge=KnowledgeConfig(
            store=store,
            aggregate=WorkingMemoryAggregate(config=gemini_flash_config),
            aggregate_trigger=AggregateTrigger(every_n_turns=2, on_end=False),
        ),
    )

    r = await agent.ask("Say 'a'.", stream=stream)  # turn 1
    r = await r.ask("Say 'b'.")                      # turn 2 → fire
    r = await r.ask("Say 'c'.")                      # turn 3
    r = await r.ask("Say 'd'.")                      # turn 4 → fire

    assert len(aggregate_events) == 2
    for evt in aggregate_events:
        assert evt.strategy == "WorkingMemoryAggregate"

    # Aggregator must have written to the knowledge store
    assert len(await store.list("/")) > 0


async def test_every_n_events_aggregation(gemini_flash_config) -> None:
    """AggregateTrigger(every_n_events=N) fires when history crosses multiples of N.

    Stateless semantics: each turn we check whether total event count in
    history moved from ``< k*N`` to ``≥ k*N`` for some integer k. With two
    events per simple turn (``ModelRequest`` + ``ModelResponse``) and
    ``every_n_events=3``, the crossings are 3, 6, 9, … — which correspond
    to asks 2, 3, 5, …
    """
    store = MemoryKnowledgeStore()

    aggregate_events: list = []
    stream = MemoryStream()
    stream.where(AggregationCompleted).subscribe(lambda e: aggregate_events.append(e))

    agent = Actor(
        "event-counter",
        prompt="Very short.",
        config=gemini_flash_config,
        knowledge=KnowledgeConfig(
            store=store,
            aggregate=WorkingMemoryAggregate(config=gemini_flash_config),
            aggregate_trigger=AggregateTrigger(every_n_events=3, on_end=False),
        ),
    )

    r = await agent.ask("Say 'w'.", stream=stream)  # 0→2, no crossing
    r = await r.ask("Say 'x'.")                      # 2→4, crosses 3 → fire
    r = await r.ask("Say 'y'.")                      # 4→6, crosses 6 → fire
    r = await r.ask("Say 'z'.")                      # 6→8, no crossing

    assert len(aggregate_events) == 2
