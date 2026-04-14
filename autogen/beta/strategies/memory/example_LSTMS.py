"""
example_memory_agent.py — How to use LongShortTermMemoryStorage with an ag2-beta Agent
========================================================================================

This file is a self-contained, runnable tutorial.  Set your API key and run:

    python example_memory_agent.py

What the example demonstrates
------------------------------
PART A — Short-term memory (intra-session L1 compression)
  By using a very low token_threshold (500 tokens) and keeping only 2 raw turns,
  the storage compresses early conversation turns into L1 blocks *mid-session*
  as the rolling context grows.  The agent must call memory_lookup() to retrieve
  exact details from those compressed turns — even though the conversation is
  still in progress (no session boundary has been crossed yet).

  This exercises the intra-session compression path independently of L2
  consolidation.

PART B — Long-term memory (cross-session L2 consolidation)
  1. Creating LongShortTermMemoryStorage with just a model config — no custom
     summarizer or consolidator needed; built-in defaults handle both.
  2. Wiring the storage into a MemoryStream.
  3. Giving the agent a memory_lookup tool so it can drill into stored blocks.
  4. Running a multi-turn conversation inside storage.session() so that
     L1/L2/core consolidation and JSON persistence happen automatically at
     session end.
  5. A second session that introduces new personal facts, showing how the
     core memory block evolves across sessions.
  6. A third session that must call memory_lookup to retrieve specific details.
"""

import asyncio
import os
from pathlib import Path

from autogen.beta.agent import Agent
from autogen.beta.config.openai.config import OpenAIConfig
from autogen.beta.history import LongShortTermMemoryStorage
from autogen.beta.stream import MemoryStream

# ---------------------------------------------------------------------------
# 0.  Configuration
# ---------------------------------------------------------------------------

# Paste your key here or export OPENAI_API_KEY before running.
API_KEY = os.environ.get("OPENAI_API_KEY", "None")

# The model used for both the main agent and memory summarisation.
MODEL = "gpt-5-mini"

config = OpenAIConfig(model=MODEL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# 1.  Create the storage (used for Part B — long-term memory)
#     Passing only `config` is enough — LongShortTermMemoryStorage builds
#     a default summarizer and a smart LLM-driven consolidator automatically.
#     Override them with explicit `summarizer=` / `consolidator=` kwargs only
#     when you need custom behaviour.
# ---------------------------------------------------------------------------

storage = LongShortTermMemoryStorage(
    config=config,
    l2_store_path=Path(__file__).parent / "agent_memory.json",  # always next to this script
    token_threshold=10_000,    # compress to L1 when raw context exceeds ~10k tokens
    max_l2_blocks=20,          # hard cap on long-term blocks
    recent_turns=5,            # keep last 5 turns as raw context
)


# ---------------------------------------------------------------------------
# 2.  Create the agent (used for Part B — long-term memory)
#     We pass storage.memory_lookup_tool() so the agent can call
#     memory_lookup(block_id) to drill into the source text of any L1 block
#     or the sub-blocks of any L2 block.
# ---------------------------------------------------------------------------

agent = Agent(
    "assistant",
    prompt=(
        "You are a helpful assistant with long-term memory. "
        "Your memory context is pinned at the top of each message. "
        "Use memory_lookup(block_id) whenever you want to read the full "
        "detail behind a memory block."
    ),
    config=config,
    tools=[storage.memory_lookup_tool()],
)


# ---------------------------------------------------------------------------
# 3.  Helper utilities
# ---------------------------------------------------------------------------

def print_memory_state(label: str, stor: LongShortTermMemoryStorage | None = None) -> None:
    """Print a formatted snapshot of every memory tier to stdout."""
    s = stor or storage
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")

    # ── Core memory ────────────────────────────────────────────
    if s._core_memory:
        cm = s._core_memory
        print(f"  [Core Memory]  (last updated: {cm.updated_at})")
        for line in cm.content.splitlines():
            print(f"    {line}")
    else:
        print("  [Core Memory]  (none yet — will be created at first session end)")

    # ── L2 long-term blocks ────────────────────────────────────
    if s._l2_blocks:
        print(f"\n  [L2 Blocks]  ({len(s._l2_blocks)} total)")
        for b in s._l2_blocks:
            print(f"    [{b.id}] (created {b.created_at})")
            print(f"      {b.abstract}")
    else:
        print("\n  [L2 Blocks]  (none yet)")

    # ── L1 short-term blocks (current session) ─────────────────
    all_l1 = [b for blocks in s._l1_blocks.values() for b in blocks]
    if all_l1:
        print(f"\n  [L1 Blocks]  ({len(all_l1)} in this session)")
        for b in all_l1:
            print(f"    [{b.id}] (created {b.created_at})")
            print(f"      {b.abstract}")
    else:
        print("\n  [L1 Blocks]  (none yet in this session)")

    print(f"{'─'*60}")


def _show_intra_session_l1(stor: LongShortTermMemoryStorage, stream: "MemoryStream", label: str) -> None:
    """Print the L1 blocks and raw-event count for a specific stream mid-session.

    Because L1 compression is triggered lazily inside get_history(), the block
    counts here reflect what was already compressed on the *previous* agent.ask
    call (not the current pending turn).
    """
    stream_id = stream.id
    l1_list = list(stor._l1_blocks.get(stream_id, []))
    raw_count = len(list(stor._raw_events.get(stream_id, [])))
    print(f"\n  ┌─ {label}")
    print(f"  │  L1 blocks created so far this session : {len(l1_list)}")
    print(f"  │  Raw events still in active window     : {raw_count}")
    for b in l1_list:
        snippet = b.abstract[:90].replace("\n", " ")
        print(f"  │    [{b.id}] {snippet}{'…' if len(b.abstract) > 90 else ''}")
    print(f"  └{'─'*56}")


async def run_diagnostic_session() -> None:
    """Send a single diagnostic prompt that asks the LLM to list every tool
    and memory block visible to it, then discard the conversation entirely.

    This session is intentionally NOT wrapped in ``storage.session()``, so:
      • No L1 compression is triggered (the diagnostic turn is short).
      • No L2 consolidation happens.
      • The raw events are dropped via ``storage.drop_history()`` at the end,
        leaving the persistent memory state completely unchanged.

    Use this between sessions to verify that the memory context looks correct.
    """
    stream = MemoryStream(storage=storage)

    print(f"\n{'='*60}")
    print("Diagnostic session  (not stored in long-term memory)")
    print('='*60)

    print_memory_state("Memory visible to agent")

    prompt = (
        "This is a diagnostic check. Please list:\n"
        "1. Every tool you currently have access to (name and one-line description).\n"
        "2. The full content of your core memory (if any).\n"
        "3. Every L2 and L1 memory block visible in your context — "
        "   for each block state its ID, tier, creation time, and abstract.\n"
        "Be exhaustive and format the output clearly."
    )

    print(f"\nUser: {prompt}")
    reply = await agent.ask(prompt, stream=stream)
    print(f"\nAgent: {await reply.content()}")

    # Discard the raw events for this stream so nothing leaks into L1/L2.
    await storage.drop_history(stream.id)
    print("\n[diagnostic] Raw events discarded — memory state unchanged.")


async def run_session(task: str, follow_ups: list[str] | None = None) -> None:
    """
    Run a user task (and optional follow-up tasks) in a single agent session.

    Each string is a complete, self-contained task description passed to
    agent.ask().  The agent may internally make multiple tool calls and
    LLM turns to complete it — all of that stays on the same stream.

    A new MemoryStream is created for each session so the raw event log
    starts fresh, but L1/L2/core blocks from previous sessions are already
    loaded into `storage` (from disk).
    """
    stream = MemoryStream(storage=storage)

    print(f"\n{'='*60}")
    print("Starting new session")
    print('='*60)
    print_memory_state("Memory state BEFORE this session")

    async with storage.session():           # <-- consolidate() called on exit
        # Primary task
        print(f"\nUser: {task}")
        reply = await agent.ask(task, stream=stream)
        print(f"Agent: {await reply.content()}")

        # Optional follow-up tasks — each continues on the same stream,
        # so the agent retains the full conversation context from above.
        for follow_up in (follow_ups or []):
            print(f"\nUser: {follow_up}")
            reply = await reply.ask(follow_up)   # continues the same context
            print(f"Agent: {await reply.content()}")

    # Consolidation has run; show what changed.
    print_memory_state("Memory state AFTER this session")
    if storage._l2_blocks and storage._l2_store_path:
        print(f"  → Saved to: {storage._l2_store_path}")


# ---------------------------------------------------------------------------
# PART A — Short-term (intra-session) memory test
# ---------------------------------------------------------------------------

async def test_short_term_memory_within_session() -> None:
    """Demonstrate SHORT-TERM (L1) memory operating *within* a single session.

    We create a dedicated storage with a very low token_threshold (500 tokens)
    and recent_turns=2.  This forces the storage to compress early conversation
    turns into L1 blocks mid-session as soon as the rolling raw context exceeds
    ~500 tokens.  Only the last 2 turns remain as raw events; everything older
    is captured in an L1 block.

    The conversation covers a multi-service microservices architecture:
      • Turns 1-2: establish precise technical details (ports, DB host, pool
                   sizes, Redis TTL) that will scroll out of the active window.
      • Turn  3  : introduces new topic (rate limiting), pushing earlier turns
                   into L1 compression.
      • Turn  4  : Kubernetes resources — more new content, more compression.
      • Turn  5  : asks the agent to recall exact details from the earliest
                   turns; the agent must call memory_lookup() because those
                   turns are now only accessible via L1 blocks.

    This session is NOT wrapped in storage.session(), so no L2 consolidation
    occurs and the main storage state is unaffected.
    """
    # Dedicated storage with a tiny threshold so L1 compression fires quickly.
    short_storage = LongShortTermMemoryStorage(
        config=config,
        token_threshold=500,   # compress once raw context exceeds ~500 tokens
        recent_turns=2,        # keep only the 2 most-recent turns as raw events
        max_l2_blocks=5,
        # No l2_store_path — memory is in-process only for this demo
    )
    short_agent = Agent(
        "short_term_assistant",
        prompt=(
            "You are a precise technical assistant. "
            "Use memory_lookup(block_id) to retrieve exact details from any "
            "L1 short-term memory block shown in your context header."
        ),
        config=config,
        tools=[short_storage.memory_lookup_tool()],
    )

    stream = MemoryStream(storage=short_storage)

    print(f"\n{'='*60}")
    print("PART A — SHORT-TERM MEMORY WITHIN A SINGLE SESSION")
    print("        token_threshold=500 tokens, recent_turns=2")
    print('='*60)

    # ── Turn 1 ────────────────────────────────────────────────────────────
    # Introduce concrete technical facts we will query back in Turn 5.
    turn1 = (
        "I'm designing a microservices architecture with three services:\n"
        "• Service A — Python FastAPI gateway on port 8001, handles authentication.\n"
        "• Service B — Go gRPC service on port 50051, handles order processing, "
        "  200ms SLA, max 1000 concurrent goroutines.\n"
        "• Service C — Node.js REST API on port 3000, handles push notifications.\n"
        "Acknowledge each service and briefly note its main responsibility."
    )
    print(f"\nTurn 1 — User:\n{turn1}")
    reply = await short_agent.ask(turn1, stream=stream)
    print(f"\nAgent:\n{await reply.content()}")
    _show_intra_session_l1(short_storage, stream, "After Turn 1")

    # ── Turn 2 ────────────────────────────────────────────────────────────
    # Add precise DB / cache details — these should push Turn 1 toward L1.
    turn2 = (
        "Service B connects to:\n"
        "• PostgreSQL on host db-prod-01:5432, schema 'orders_v2', pool size 50.\n"
        "• Redis cache at cache-prod-01:6379, database index 3, TTL 300 seconds "
        "  for order-status entries.\n"
        "List the two key failure modes if the Redis cache becomes unavailable."
    )
    print(f"\nTurn 2 — User:\n{turn2}")
    reply = await reply.ask(turn2)
    print(f"\nAgent:\n{await reply.content()}")
    _show_intra_session_l1(short_storage, stream, "After Turn 2")

    # ── Turn 3 ────────────────────────────────────────────────────────────
    # New topic — rate limiting — to drive further compression.
    turn3 = (
        "Service A needs a sliding-window rate limiter: 100 requests per user "
        "per 60 seconds, counters stored in the same Redis instance at db index 1. "
        "Show me a minimal Python snippet implementing this with redis-py."
    )
    print(f"\nTurn 3 — User:\n{turn3}")
    reply = await reply.ask(turn3)
    print(f"\nAgent:\n{await reply.content()}")
    _show_intra_session_l1(short_storage, stream, "After Turn 3  ← early turns now in L1")

    # ── Turn 4 ────────────────────────────────────────────────────────────
    # Kubernetes resource quotas — yet more content to keep context growing.
    turn4 = (
        "Now for Kubernetes deployment specs:\n"
        "• Service A: 3 replicas, requests 256Mi/0.5CPU, limits 512Mi/1CPU.\n"
        "• Service B: 5 replicas, requests 512Mi/1CPU,   limits 1Gi/2CPU.\n"
        "• Service C: 2 replicas, requests 128Mi/0.25CPU, limits 256Mi/0.5CPU.\n"
        "Generate a valid Kubernetes Deployment name (RFC-1123) for each service."
    )
    print(f"\nTurn 4 — User:\n{turn4}")
    reply = await reply.ask(turn4)
    print(f"\nAgent:\n{await reply.content()}")
    _show_intra_session_l1(short_storage, stream, "After Turn 4")

    # ── Turn 5 — the recall test ───────────────────────────────────────────
    # Ask for exact numbers from the earliest turns that are now in L1.
    # The agent's active raw window only holds the last 2 turns, so it must
    # call memory_lookup() to answer accurately.
    turn5 = (
        "I need to file an architecture decision record. Please recall precisely:\n"
        "1. The port number and runtime language for each of Services A, B, and C.\n"
        "2. The exact PostgreSQL host, port, schema name, and connection-pool size "
        "   for Service B.\n"
        "3. The Redis host, port, database index, and TTL used for order-status "
        "   caching in Service B.\n"
        "If any of these details have been compressed into L1 memory blocks, "
        "call memory_lookup(block_id) to retrieve the full source text before "
        "answering — do not guess."
    )
    print(f"\nTurn 5 — User (recall test):\n{turn5}")
    reply = await reply.ask(turn5)
    print(f"\nAgent:\n{await reply.content()}")
    _show_intra_session_l1(short_storage, stream, "After Turn 5 — final L1 state")

    # Drop history — no L2 consolidation; main storage unchanged.
    await short_storage.drop_history(stream.id)
    print(
        "\n[Part A] Short-term memory test complete.\n"
        "         History discarded — main storage state is unchanged."
    )


# ---------------------------------------------------------------------------
# PART B — Long-term memory test (cross-session L2 / core memory)
# ---------------------------------------------------------------------------

async def test_long_term_memory() -> None:
    """Run three sessions that demonstrate L2 consolidation and core memory
    creation / evolution across session boundaries.

    Diagnostic (optional) — Verify the full memory context visible to the agent.
    Nothing here is stored; the conversation is dropped on exit.

    Session 1 — Alex introduces his project.
      Expected: core memory is created for the first time (name + project).

    Session 2 — Alex reveals new personal facts.
      Expected: core memory is updated (age, location, employer, tech pref,
      pivot from recommendation to search ranking).

    Session 3 — memory_lookup recall demo.
      The agent must look up the exact Python snippet written in session 1
      from an L1 source block, because it is not in any L2 abstract.
    """
    # ── Diagnostic ───────────────────────────────────────────────────────
    # Ask the agent to enumerate tools and the full memory context it sees.
    # Nothing produced here is saved; the conversation is dropped on exit.
    #await run_diagnostic_session()


    # ── Session 1 ────────────────────────────────────────────────────────
    # Alex introduces himself and his project.
    # Expected core memory outcome: core memory is created for the first
    # time, capturing name + project description.
    # (Core memory shown as "(none yet)" before, then populated after.)
    await run_session(
        task=(
            "I'm Alex. I'm building a Python recommendation system that uses "
            "collaborative filtering with matrix factorisation. The main challenge "
            "is cold-start for new users; I'm exploring content-based fallback "
            "with item embeddings as a solution. "
            "Please give me a concise technical overview of this design and "
            "suggest one concrete improvement I could make."
        ),
        follow_ups=[
            "The cold-start suggestion sounds promising. "
            "Can you show me a minimal Python snippet implementing it?",
        ],
    )

    # ── Session 2 — core memory update demo ──────────────────────────────
    # Alex reveals new personal facts that should update the core memory:
    # age, employer, location, tech preference, and a project pivot.
    # Compare the "BEFORE" and "AFTER" printouts to see the core memory
    # block change — the LLM merges the new facts with the existing profile
    # rather than discarding what it already knows.
    await run_session(
        task=(
            "Just wanted to share a bit more about myself: I'm 28, based in London, "
            "and I work at a pre-seed AI startup. I strongly prefer PyTorch over "
            "TensorFlow for all new code. Also, my project focus has shifted — "
            "I'm now applying the same embedding techniques to search ranking "
            "rather than pure recommendation. "
            "Can you briefly summarise what you now know about me and my project?"
        ),
    )

    # ── Session 3 — memory_lookup demo ───────────────────────────────────
    # A fresh stream with the updated core memory and L2 blocks persisted
    # from sessions 1 & 2.  The task asks for the *exact* Python snippet
    # written in session 1 — only available in an L1 source block, not
    # in any L2 abstract — so the agent must call memory_lookup to find it.
    await run_session(
        task=(
            "Hi again. In our last conversation you showed me a Python snippet "
            "for the content-based cold-start fallback. "
            "Please look it up from your memory (use memory_lookup as needed — "
            "the code lives in an L1 source block) and paste the exact code here."
        ),
        follow_ups=[
            "Good. Now extend that snippet to handle the edge case where "
            "no item embeddings are available at all (completely new item). "
            "Show only the changed/added lines."
        ],
    )


# ---------------------------------------------------------------------------
# 4.  Main entry-point
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 60)
    print("PART A — Short-term memory within a single long-context session")
    print("         (intra-session L1 compression + memory_lookup recall)")
    print("=" * 60)
    await test_short_term_memory_within_session()

    print("\n\n" + "=" * 60)
    print("PART B — Long-term memory across multiple sessions")
    print("         (L2 consolidation + core memory creation/evolution)")
    print("=" * 60)
    await test_long_term_memory()


if __name__ == "__main__":
    asyncio.run(main())
