# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Edge-case smoke: long conversations + compaction, concurrent tools, unicode,
retry middleware on flaky tools, history limiter, response schema retry on
malformed output, very large responses.
"""

import asyncio

import pytest

from autogen.beta import Agent
from autogen.beta.agent import KnowledgeConfig
from autogen.beta.compact import CompactTrigger, TailWindowCompact
from autogen.beta.events.lifecycle import CompactionCompleted
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.middleware import BaseMiddleware
from autogen.beta.middleware.builtin import HistoryLimiter, RetryMiddleware
from autogen.beta.stream import MemoryStream

pytestmark = pytest.mark.asyncio


async def test_long_conversation_with_compaction(provider_config) -> None:
    """A multi-turn conversation triggers compaction without breaking the chat.

    With max_events=8 (a low threshold), compaction must fire after a few
    turns and the agent should still answer coherently.
    """
    store = MemoryKnowledgeStore()
    compactions: list[CompactionCompleted] = []

    stream = MemoryStream()
    stream.where(CompactionCompleted).subscribe(lambda e: compactions.append(e))

    agent = Agent(
        "long-talker",
        prompt="Be very brief — 1 sentence answers.",
        config=provider_config,
        knowledge=KnowledgeConfig(
            store=store,
            compact=TailWindowCompact(target=4),
            compact_trigger=CompactTrigger(max_events=8),
        ),
    )

    questions = [
        "What colour is the sky on a clear day?",
        "What colour is grass?",
        "What colour is fresh snow?",
        "What colour is a ripe tomato?",
        "What colour is coal?",
    ]

    reply = await agent.ask(questions[0], stream=stream)
    for q in questions[1:]:
        reply = await reply.ask(q)

    assert reply.body is not None
    # At least one compaction must have fired
    assert len(compactions) >= 1


async def test_unicode_and_emoji_pass_through(provider_config) -> None:
    """Emoji and non-ASCII characters survive round-trip through the agent."""

    def echo(text: str) -> str:
        """Echo the text back unchanged."""
        return f"echo: {text}"

    agent = Agent(
        "unicode",
        prompt="Use the echo tool for any text the user gives you.",
        config=provider_config,
        tools=[echo],
    )
    reply = await agent.ask("Use echo with text='日本語 🎌 ñoño 🚀'.")
    assert reply.body is not None
    body = reply.body
    # At least one of the unicode tokens should round-trip
    assert any(token in body for token in ["日本語", "🎌", "ñoño", "🚀"])


async def test_concurrent_tool_calls_via_run_subtasks(provider_config) -> None:
    """run_subtasks(parallel=True) dispatches subtasks concurrently.

    Verified by inspecting ``TaskStarted.created_at`` for the three subtasks:
    when dispatched in parallel they are kicked off within the same event-loop
    tick, so the spread between the earliest and latest ``created_at`` must
    be far smaller than a single LLM round trip. Sequential dispatch would
    space them by 1+ seconds each.
    """
    from autogen.beta.events import TaskStarted

    starts: list[TaskStarted] = []
    stream = MemoryStream()
    stream.where(TaskStarted).subscribe(lambda e: starts.append(e))

    agent = Agent(
        "parallel",
        prompt=("Use run_subtasks with parallel=True to dispatch independent jobs. Be concise."),
        config=provider_config,
    )

    reply = await agent.ask(
        "Use run_subtasks with parallel=True to ask three things at once: "
        "(1) what is 2+2, (2) what is 5+5, (3) what is 10+10. "
        "Then list all three results.",
        stream=stream,
    )

    assert reply.body is not None
    body = reply.body
    assert "4" in body
    assert "10" in body
    assert "20" in body
    # 3 subtasks must have started
    assert len(starts) >= 3, f"expected ≥3 TaskStarted events, got {len(starts)}"
    times = sorted(s.created_at for s in starts[:3])
    spread = times[-1] - times[0]
    # Parallel dispatch happens within one event-loop iteration → spread is
    # typically sub-millisecond; sequential would be ≥1s per LLM call.
    assert spread < 0.5, f"subtasks were not dispatched in parallel; spread={spread:.3f}s"


async def test_retry_middleware_happy_path(provider_config) -> None:
    """RetryMiddleware wraps LLM calls; happy-path traffic is unaffected."""
    agent = Agent(
        "retry",
        config=provider_config,
        middleware=[RetryMiddleware(max_retries=2)],
    )
    reply = await agent.ask("Say 'ok'.")
    assert reply.body is not None
    assert "ok" in reply.body.lower()


async def test_history_limiter_caps_context(provider_config) -> None:
    """HistoryLimiter caps the number of events forwarded to the LLM.

    HistoryLimiter preserves the first ``ModelRequest`` and the trailing window;
    we verify the trim happened by recording how many events the LLM call
    actually received via a capture middleware.
    """
    captured_lengths: list[int] = []

    class CaptureLLMCall(BaseMiddleware):
        async def on_llm_call(self, call_next, events, context):
            captured_lengths.append(len(events))
            return await call_next(events, context)

    # Middleware ordering: HistoryLimiter must be OUTSIDE CaptureLLMCall so
    # the capture sees the post-trim event list. Earlier list entries become
    # outer wrappers, so HistoryLimiter goes first.
    agent = Agent(
        "limited",
        prompt="One short word answers.",
        config=provider_config,
        middleware=[
            HistoryLimiter(max_events=3),
            lambda e, c: CaptureLLMCall(e, c),
        ],
    )

    r = await agent.ask("Remember 'apple'.")
    r = await r.ask("Remember 'banana'.")
    r = await r.ask("Remember 'cherry'.")
    r = await r.ask("Remember 'date'.")
    r = await r.ask("Say 'done'.")
    assert r.body is not None

    # Final ask: history has 10 events (5 requests + 5 responses) before the
    # capture middleware runs. CaptureLLMCall sees the *trimmed* output of
    # HistoryLimiter — it must be ≤ max_events=3.
    assert captured_lengths
    assert captured_lengths[-1] <= 3


async def test_response_schema_strict_validation(provider_config) -> None:
    """AgentReply.content() returns a validated instance for strict schemas.

    NOTE: with ``temperature=0`` the model produces a valid response on the
    first try, so this test exercises the happy path of strict-output
    validation — *not* the retry-on-malformed recovery path. The retry
    branch is covered by unit tests; here we assert the typed result and
    that ``retries=0`` succeeds (proving no retry was needed).
    """
    from pydantic import BaseModel, Field

    class StrictAnswer(BaseModel):
        answer: int = Field(..., description="The numeric answer, no text")

    agent = Agent(
        "strict",
        prompt="Reply ONLY with valid JSON matching the schema. The 'answer' must be an integer.",
        config=provider_config,
        response_schema=StrictAnswer,
    )

    reply = await agent.ask("What is 100 / 4? Return the integer answer.")
    result = await reply.content(retries=0)
    assert isinstance(result, StrictAnswer)
    assert result.answer == 25


async def test_large_response(provider_config) -> None:
    """Agent handles a large multi-paragraph response without truncation."""
    agent = Agent(
        "verbose",
        prompt="When asked, write a thorough multi-paragraph response.",
        config=provider_config,
    )
    reply = await agent.ask("Write 5 paragraphs about photosynthesis. Each paragraph at least 50 words.")
    assert reply.body is not None
    assert len(reply.body) > 500  # Roughly 5 * 50 * avg-word-length


async def test_empty_string_user_message(provider_config) -> None:
    """Sending an empty-ish user message should still produce a coherent reply."""
    agent = Agent(
        "empty",
        prompt="If the user sends nothing useful, just say 'ok'.",
        config=provider_config,
    )
    reply = await agent.ask(".")
    assert reply.body is not None
    assert reply.body.strip() != ""


async def test_concurrent_independent_asks(provider_config) -> None:
    """Two .ask() calls on the same agent with independent streams should not interfere."""
    agent = Agent(
        "parallel-asks",
        prompt="Reply with just the requested word.",
        config=provider_config,
    )

    r1, r2 = await asyncio.gather(
        agent.ask("Say 'one'.", stream=MemoryStream()),
        agent.ask("Say 'two'.", stream=MemoryStream()),
    )

    assert "one" in (r1.body or "").lower()
    assert "two" in (r2.body or "").lower()
