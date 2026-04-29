"""Research squad — parallel subtasks and sibling delegation.

Mirrors website/docs/beta/code_examples/05_research_squad.mdx. Two patterns
for multi-Agent orchestration:

1. **Opt-in subtask tools.** Pass tasks=TaskConfig(...) and the Agent gains
   run_subtask / run_subtasks. The coordinator uses run_subtasks with
   parallel=True to fan out three short investigations concurrently.
2. **Agent.as_tool().** A second Agent (math_expert) is exposed to the
   coordinator as a callable tool.

Both patterns: spawned subtasks have no run_subtask tools (they default to
tasks=False), so recursion is structurally impossible — no depth limiter
needed.

Run::

    python research_squad.py
"""

import asyncio
import time

from autogen.beta import Agent
from autogen.beta.agent import TaskConfig
from autogen.beta.config import GeminiConfig
from autogen.beta.events import TaskCompleted, TaskStarted
from autogen.beta.stream import MemoryStream


def section(title: str) -> None:
    print(f"\n── {title} ───")


async def main() -> None:
    config = GeminiConfig(model="gemini-3-flash-preview", temperature=0)

    section("Parallel subtasks — fan out three lookups in one tool call")

    coordinator = Agent(
        "coordinator",
        prompt=(
            "You answer multi-part questions by dispatching run_subtasks "
            "with parallel=True. Use one tool call with every sub-question "
            "packed into the 'tasks' list. Be concise."
        ),
        config=config,
        tasks=TaskConfig(),  # Opt in to run_subtask / run_subtasks.
    )

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

    math_expert = Agent(
        "math-expert",
        prompt="You are an arithmetic specialist. Reply with only the number.",
        config=config,
    )

    coordinator2 = Agent(
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
            )
        ],
    )

    reply2 = await coordinator2.ask("What is 237 times 19?")
    print(reply2.body)


if __name__ == "__main__":
    asyncio.run(main())
