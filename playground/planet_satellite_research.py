"""Planet-Satellite Architecture: Research Report Generator

Demonstrates the planet-satellite pattern for decomposing a complex
research task into parallel sub-tasks, with real-time streaming progress.

Architecture:
    Planet (gpt-5.4)  -- orchestrates the research, decides sub-topics,
                         spawns satellites, synthesises the final report.
    Satellites (gpt-5.2) -- each researches one sub-topic independently.
    TokenMonitor       -- tracks cumulative token usage across all agents.
    LoopDetector       -- flags repetitive tool-call patterns.

Usage:
    export OPENAI_API_KEY=sk-...
    python playground/planet_satellite_research.py
"""

import asyncio
import os
import sys

from autogen.beta.config.openai import OpenAIConfig
from autogen.beta.events import ModelMessageChunk
from autogen.beta.satellites import (
    LoopDetector,
    PlanetAgent,
    SatelliteFlag,
    SatelliteStarted,
    TaskSatelliteProgress,
    TaskSatelliteRequest,
    TaskSatelliteResult,
    TokenMonitor,
)
from autogen.beta.stream import MemoryStream

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

planet_config = OpenAIConfig(
    model="gpt-5.4",
    api_key=os.environ.get("OPENAI_API_KEY"),
    streaming=True,
)

satellite_config = OpenAIConfig(
    model="gpt-5.2",
    api_key=os.environ.get("OPENAI_API_KEY"),
    streaming=True,
)


# ---------------------------------------------------------------------------
# Planet agent
# ---------------------------------------------------------------------------

PLANET_PROMPT = """\
You are a senior research analyst. When given a research topic you:

1. Identify 3-4 key sub-topics that together cover the subject comprehensively.
2. Use the `spawn_tasks` tool to delegate each sub-topic to a task satellite.
   Pass all sub-topics at once with parallel=true for efficiency.
3. Once the satellite results come back, synthesise them into a cohesive
   executive-summary style report (500-800 words) with clear section headers.

Keep your own token usage low -- delegate research, focus on synthesis.
"""

planet = PlanetAgent(
    "Research Director",
    prompt=PLANET_PROMPT,
    config=planet_config,
    satellite_config=satellite_config,
    satellite_prompt=(
        "You are a research specialist. Provide a thorough, factual analysis "
        "of the assigned topic in 200-400 words. Include specific examples, "
        "data points, or trends where possible. Be concise and informative."
    ),
    satellites=[
        TokenMonitor(warn_threshold=20_000, alert_threshold=50_000),
        LoopDetector(window_size=6, repeat_threshold=3),
    ],
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def main() -> None:
    topic = (
        "The impact of AI on software engineering productivity in 2025: "
        "code generation, testing, deployment, and developer experience"
    )

    print(f"{'=' * 70}")
    print(f"Research Topic: {topic}")
    print(f"  Planet model:    {planet_config.model}")
    print(f"  Satellite model: {satellite_config.model}")
    print(f"{'=' * 70}\n")

    stream = MemoryStream()
    _speaker = ""

    async def _on_event(event: object) -> None:
        nonlocal _speaker
        if isinstance(event, SatelliteStarted):
            print(f"  [satellite] {event.name} attached", flush=True)

        elif isinstance(event, TaskSatelliteRequest):
            _speaker = ""
            print(
                f"\n  \033[32m[spawn]\033[0m {event.satellite_name} "
                f"({satellite_config.model}): {event.task[:60]}...",
                flush=True,
            )

        elif isinstance(event, TaskSatelliteProgress):
            if _speaker != event.satellite_name:
                _speaker = event.satellite_name
                print(
                    f"\n\033[2m  [Satellite: {event.satellite_name} ({satellite_config.model})] >\033[0m\n",
                    flush=True,
                )
            sys.stdout.write(f"\033[2m{event.content}\033[0m")
            sys.stdout.flush()

        elif isinstance(event, TaskSatelliteResult):
            _speaker = ""
            print(
                f"\n  \033[32m[done]\033[0m  {event.satellite_name}: "
                f"{len(event.result)} chars",
                flush=True,
            )

        elif isinstance(event, SatelliteFlag):
            print(
                f"\n  \033[33m[flag]\033[0m  [{event.severity}] {event.message}",
                flush=True,
            )

        elif isinstance(event, ModelMessageChunk):
            if _speaker != "planet":
                _speaker = "planet"
                print(f"\n\033[1;36m  [Planet: Research Director ({planet_config.model})] >\033[0m\n", flush=True)
            sys.stdout.write(event.content)
            sys.stdout.flush()

    stream.subscribe(_on_event)

    print("Thinking...\n", flush=True)
    conversation = await planet.ask(topic, stream=stream)
    print()  # newline after streaming

    # Show token stats
    for sat in planet._satellites:
        if isinstance(sat, TokenMonitor):
            print(f"\n[Token usage: {sat.total_tokens:,} total tokens]")


if __name__ == "__main__":
    asyncio.run(main())
