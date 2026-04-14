"""05 · Research squad — parallel subtasks and sibling delegation

Two patterns for multi-Actor orchestration:

1. **Auto-injected subtask tools.** Every Actor automatically carries
   ``run_subtask`` / ``run_subtasks``. The coordinator uses ``run_subtasks``
   with ``parallel=True`` to fan out three short investigations
   concurrently.

2. **``Actor.as_tool()``.** A second Actor (``math_expert``) is exposed to
   the coordinator as a callable tool with a ``depth_limiter`` middleware
   so it can't recurse into itself indefinitely.

Run::

    .venv-beta/bin/python playground/05_research_squad.py
"""

from __future__ import annotations

import asyncio
import time

from _config import default_config, section

from autogen.beta import Actor
from autogen.beta.events import TaskCompleted, TaskStarted
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.subagents import depth_limiter


async def main() -> None:
    config = default_config()

    section("Parallel subtasks — fan out three lookups in one tool call")

    coordinator = Actor(
        "coordinator",
        prompt=(
            "You answer multi-part questions by dispatching run_subtasks "
            "with parallel=True. Use one tool call with every sub-question "
            "packed into the 'tasks' list. Be concise."
        ),
        config=config,
    )

    # Collect subtask lifecycle events so we can show the fan-out to the user
    starts: list[TaskStarted] = []
    completions: list[TaskCompleted] = []
    stream = MemoryStream()
    stream.where(TaskStarted).subscribe(lambda e: starts.append(e))
    stream.where(TaskCompleted).subscribe(lambda e: completions.append(e))

    start = time.monotonic()
    reply = await coordinator.ask(
        "Use run_subtasks(parallel=True) to answer, in one tool call: "
        "(a) what is the tallest waterfall in the world, "
        "(b) what year was the Eiffel Tower completed, "
        "(c) what is the boiling point of nitrogen in Celsius. "
        "Then list all three answers.",
        stream=stream,
    )
    elapsed = time.monotonic() - start

    print(reply.body)
    print()
    print(f"Subtasks dispatched: {len(starts)}")
    print(f"Subtasks finished:   {len(completions)}")
    print(f"Wall time:           {elapsed:.2f}s (3 concurrent LLM calls)")

    section("Sibling delegation — math_expert is a tool on coordinator2")

    math_expert = Actor(
        "math-expert",
        prompt="You are an arithmetic specialist. Reply with only the number.",
        config=config,
    )

    coordinator2 = Actor(
        "coordinator2",
        prompt=(
            "When arithmetic comes up, delegate to the task_math-expert tool "
            "rather than computing yourself. Then present the answer in a "
            "complete sentence."
        ),
        config=config,
        tools=[
            math_expert.as_tool(
                description="Delegate arithmetic problems to the math expert.",
                middleware=[depth_limiter(max_depth=2)],
            )
        ],
    )

    reply2 = await coordinator2.ask("What is 237 times 19?")
    print(reply2.body)


if __name__ == "__main__":
    asyncio.run(main())
