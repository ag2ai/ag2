# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Harness feature smoke: assembly policies, knowledge store, compaction,
aggregation — every opt-in Agent primitive exercised against a real LLM.
"""

import pytest

from autogen.beta import Agent
from autogen.beta.agent import KnowledgeConfig
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
from autogen.beta.events import ModelRequest, TextInput
from autogen.beta.events.lifecycle import (
    AggregationCompleted,
    CompactionCompleted,
)
from autogen.beta.knowledge import (
    DefaultBootstrap,
    MemoryKnowledgeStore,
)
from autogen.beta.policies import (
    ConversationPolicy,
    SlidingWindowPolicy,
    TokenBudgetPolicy,
)
from autogen.beta.stream import MemoryStream

pytestmark = pytest.mark.asyncio


async def test_conversation_policy_basic(provider_config) -> None:
    """ConversationPolicy filters stream to conversation + tool events.

    The agent should still produce a sensible reply when the policy is active.
    """
    agent = Agent(
        "conv",
        prompt="Be concise.",
        config=provider_config,
        assembly=[ConversationPolicy()],
    )
    reply = await agent.ask("Say 'hello'.")
    assert reply.body is not None
    assert "hello" in reply.body.lower()


async def test_sliding_window_trims_long_history(provider_config) -> None:
    """``SlidingWindowPolicy`` caps the events delivered to the LLM.

    Asserts on the *policy's contract* — the trimmed event list sent to
    the LLM — instead of the model's reply, which is unreliable across
    providers (assistant responses can echo trimmed words back).
    """
    sent_payloads: list[list] = []

    class _CapturingClient:
        def __init__(self, inner):
            self._inner = inner

        async def __call__(self, messages, context, **kwargs):
            sent_payloads.append(list(messages))
            return await self._inner(messages, context=context, **kwargs)

    class _CapturingConfig:
        def __init__(self, inner):
            self._inner = inner

        def copy(self):
            return self

        def create(self):
            return _CapturingClient(self._inner.create())

    agent = Agent(
        "sliding",
        prompt="Be concise.",
        config=_CapturingConfig(provider_config),
        assembly=[ConversationPolicy(), SlidingWindowPolicy(max_events=4)],
    )

    r1 = await agent.ask("Remember the word 'elephant'.")
    r2 = await r1.ask("Remember the word 'volcano'.")
    r3 = await r2.ask("Remember the word 'nebula'.")
    r4 = await r3.ask("Which words have I mentioned?")
    assert r4.body is not None

    # Last LLM call is the assembly-trimmed view for r4.
    last = sent_payloads[-1]
    assert len(last) <= 5  # 4 from the window cap + the new ModelRequest
    assert not any(
        isinstance(e, ModelRequest) and any(isinstance(p, TextInput) and "elephant" in p.content for p in e.parts)
        for e in last
    ), f"sliding window did not evict 'elephant'; LLM saw: {last!r}"


async def test_token_budget_policy_clamps_history(provider_config) -> None:
    """TokenBudgetPolicy with a very tight budget evicts older turns.

    Build up a multi-turn history of fruits, then ask the agent to recall
    them. With ``max_tokens=5`` the prompt sent to the LLM cannot include the
    earlier turns, so the model cannot produce the fruit list. The control
    (no policy) is verified separately to keep this test cheap.
    """
    agent = Agent(
        "budget",
        prompt="One-word answers.",
        config=provider_config,
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


async def test_multiple_policies_compose(provider_config) -> None:
    """Composed policies: Conversation → SlidingWindow → TokenBudget."""
    agent = Agent(
        "composed",
        prompt="Brief.",
        config=provider_config,
        assembly=[
            ConversationPolicy(),
            SlidingWindowPolicy(max_events=20),
            TokenBudgetPolicy(max_tokens=2000),
        ],
    )
    reply = await agent.ask("What is 1+1? Number only.")
    assert reply.body is not None
    assert "2" in reply.body


async def test_knowledge_tool_via_actor(provider_config) -> None:
    """LLM drives the knowledge tool: write a note, list, read it back."""
    store = MemoryKnowledgeStore()

    agent = Agent(
        "knower",
        prompt=(
            "You have a `knowledge` tool with actions: read, write, list, delete. "
            "Paths are slash-separated. Always use the tool when the user asks you "
            "to remember or recall things."
        ),
        config=provider_config,
        knowledge=KnowledgeConfig(store=store),
    )

    r1 = await agent.ask(
        "Please remember that my favourite animal is a red panda. Store this at /preferences/animal.md"
    )
    assert r1.body is not None

    # Direct store inspection — the tool should have written
    content = await store.read("/preferences/animal.md")
    assert content is not None
    assert "red panda" in content.lower()

    # Ask the agent to recall via the tool
    r2 = await agent.ask("What is my favourite animal? Look it up in /preferences/animal.md and tell me.")
    assert r2.body is not None
    assert "red panda" in r2.body.lower()


async def test_bootstrap_runs_once(provider_config) -> None:
    """DefaultBootstrap writes initial store layout on first _execute."""
    store = MemoryKnowledgeStore()

    # Pre-condition: store is empty
    assert await store.exists("/.initialized") is False

    agent = Agent(
        "bootstrapped",
        config=provider_config,
        knowledge=KnowledgeConfig(store=store, bootstrap=DefaultBootstrap()),
    )
    await agent.ask("Say 'ok'.")

    # Post-condition: sentinel was written
    assert await store.exists("/.initialized") is True


async def test_tail_window_compact_triggers(provider_config) -> None:
    """TailWindowCompact (non-LLM) triggers after enough events."""
    store = MemoryKnowledgeStore()

    compact_events: list = []
    stream = MemoryStream()
    stream.where(CompactionCompleted).subscribe(lambda e: compact_events.append(e))

    agent = Agent(
        "compactor",
        prompt="Reply in one short sentence.",
        config=provider_config,
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


async def test_summarize_compact_uses_llm(provider_config) -> None:
    """SummarizeCompact uses a secondary LLM call to summarize history."""
    store = MemoryKnowledgeStore()

    compact_events: list = []
    stream = MemoryStream()
    stream.where(CompactionCompleted).subscribe(lambda e: compact_events.append(e))

    agent = Agent(
        "summarizer",
        prompt="Very short answers.",
        config=provider_config,
        knowledge=KnowledgeConfig(
            store=store,
            compact=SummarizeCompact(target=2, config=provider_config),
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


async def test_on_end_aggregation(provider_config) -> None:
    """AggregateTrigger(on_end=True) fires at execute finalisation."""
    store = MemoryKnowledgeStore()

    aggregate_events: list = []
    stream = MemoryStream()
    stream.where(AggregationCompleted).subscribe(lambda e: aggregate_events.append(e))

    agent = Agent(
        "aggregator",
        prompt="Brief answer.",
        config=provider_config,
        knowledge=KnowledgeConfig(
            store=store,
            aggregate=ConversationSummaryAggregate(config=provider_config),
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


async def test_every_n_turns_aggregation(provider_config) -> None:
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

    agent = Agent(
        "periodic",
        prompt="One word answers.",
        config=provider_config,
        knowledge=KnowledgeConfig(
            store=store,
            aggregate=WorkingMemoryAggregate(config=provider_config),
            aggregate_trigger=AggregateTrigger(every_n_turns=2, on_end=False),
        ),
    )

    r = await agent.ask("Say 'a'.", stream=stream)  # turn 1
    r = await r.ask("Say 'b'.")  # turn 2 → fire
    r = await r.ask("Say 'c'.")  # turn 3
    r = await r.ask("Say 'd'.")  # turn 4 → fire

    assert len(aggregate_events) == 2
    for evt in aggregate_events:
        assert evt.strategy == "WorkingMemoryAggregate"

    # Aggregator must have written to the knowledge store
    assert len(await store.list("/")) > 0


async def test_every_n_events_aggregation(provider_config) -> None:
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

    agent = Agent(
        "event-counter",
        prompt="Very short.",
        config=provider_config,
        knowledge=KnowledgeConfig(
            store=store,
            aggregate=WorkingMemoryAggregate(config=provider_config),
            aggregate_trigger=AggregateTrigger(every_n_events=3, on_end=False),
        ),
    )

    r = await agent.ask("Say 'w'.", stream=stream)  # 0→2, no crossing
    r = await r.ask("Say 'x'.")  # 2→4, crosses 3 → fire
    r = await r.ask("Say 'y'.")  # 4→6, crosses 6 → fire
    r = await r.ask("Say 'z'.")  # 6→8, no crossing

    assert len(aggregate_events) == 2
