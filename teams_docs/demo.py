"""AG2 Teams Demo - see the orchestration system in action.

Run: .venv/bin/python teams_docs/demo.py

Shows real-time streaming events as orchestration progresses.
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.env"))

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat.teams import Orchestrator, Team


def get_llm_config() -> LLMConfig:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in ~/.env")
    return LLMConfig({
        "model": "claude-sonnet-4-20250514",
        "api_key": api_key,
        "api_type": "anthropic",
        "temperature": 0.0,
        "max_tokens": 2048,
    })


async def main() -> None:
    llm_config = get_llm_config()

    # --- Build the team ---
    team = Team("maths-game", description="Build a maths games website")

    leader = ConversableAgent(
        name="leader",
        llm_config=llm_config,
        system_message=(
            "You are a project leader. Break goals into small, concrete tasks "
            "using create_task. Assign each task to the best team member with "
            "assign_task. Use blocked_by when tasks depend on earlier ones. "
            "Keep tasks focused — one clear deliverable each."
        ),
    )
    team.add_agent(leader, is_leader=True)

    designer = ConversableAgent(
        name="designer",
        llm_config=llm_config,
        system_message=(
            "You are a UI/UX designer. When given a task, produce concrete design "
            "specs: layout descriptions, color choices, component structure. "
            "Use complete_task with your design as the result."
        ),
        description="UI/UX designer — layouts, colors, component structure",
    )
    team.add_agent(designer)

    developer = ConversableAgent(
        name="developer",
        llm_config=llm_config,
        system_message=(
            "You are a frontend developer. When given a task, write actual HTML, "
            "CSS, and JavaScript code. Use complete_task with your code as the result."
        ),
        description="Frontend developer — HTML, CSS, JavaScript implementation",
    )
    team.add_agent(developer)

    # --- Run with streaming events ---
    orch = Orchestrator(team, max_rounds=6, max_stalls=2)

    print("=" * 60)
    print("AG2 TEAMS DEMO — Streaming Events")
    print("=" * 60)
    print()
    print("Goal: Create a simple maths quiz web page")
    print("Team: leader, designer, developer")
    print()
    print("-" * 60)

    result_event = None
    async for event in orch.run_stream(
        "Create a simple maths quiz web page with 5 addition questions, a score counter, and a 'check answers' button"
    ):
        # Print each event as it arrives
        event.content.print()

        # Capture the final result event
        if event.type == "team_run_complete":
            result_event = event

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    if result_event:
        r = result_event.content
        print(f"  Success: {r.success}")
        print(f"  Tasks completed: {r.tasks_completed}/{r.tasks_total}")
        print(f"  Total agent turns: {r.total_turns}")
        print()
        print("--- Summary ---")
        print(r.summary)
    else:
        print("  No result event received.")


if __name__ == "__main__":
    asyncio.run(main())
