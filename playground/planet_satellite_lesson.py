"""Planet-Satellite Architecture: Lesson Plan Generator

Demonstrates a pure-LLM planet-satellite workflow (no tools). The planet acts
as a curriculum designer, decomposing a lesson into parallel content generation
tasks: conceptual explanation, practice problems, misconceptions guide, and
a real-world applications section.

A custom "difficulty calibration" satellite monitors the output to flag
content that may be too advanced or too simple for the target audience.

Architecture:
    Planet (main model)      -- designs lesson structure, delegates, assembles
    Satellites (lighter)     -- each creates one section of the lesson
    TokenMonitor             -- tracks token usage
    DifficultyMonitor        -- custom satellite flagging audience mismatch

Usage:
    export OPENAI_API_KEY=sk-...
    python playground/planet_satellite_lesson.py
"""

import asyncio
import os
import sys

from autogen.beta.annotations import Context
from autogen.beta.config.openai import OpenAIConfig
from autogen.beta.events import BaseEvent, ModelMessage, ModelMessageChunk
from autogen.beta.satellites import (
    BaseSatellite,
    OnEvent,
    PlanetAgent,
    SatelliteFlag,
    Severity,
    TaskSatelliteProgress,
    TaskSatelliteRequest,
    TaskSatelliteResult,
    TokenMonitor,
)
from autogen.beta.stream import MemoryStream

# ---------------------------------------------------------------------------
# Custom satellite: difficulty calibration monitor
# ---------------------------------------------------------------------------

# Terms that suggest content is too advanced for an intro CS course
ADVANCED_MARKERS = [
    "dynamic programming",
    "memoization",
    "tail call optimization",
    "continuation passing",
    "y combinator",
    "church encoding",
    "lambda calculus",
    "ackermann",
    "mutual recursion",
    "trampolining",
    "corecursion",
]

# Terms that suggest content is too basic (not challenging enough)
TOO_BASIC_MARKERS = [
    "just like a loop",
    "simply repeat",
    "nothing special",
]


class DifficultyMonitor(BaseSatellite):
    """Monitors lesson content for audience-level mismatches.

    Flags when generated content uses concepts that are too advanced or
    too simplistic for the specified audience level.
    """

    def __init__(self, audience: str = "CS101 undergraduates") -> None:
        super().__init__("difficulty-monitor", trigger=OnEvent(ModelMessage))
        self._audience = audience
        self._flagged_terms: set[str] = set()

    async def process(
        self, events: list[BaseEvent], ctx: Context
    ) -> SatelliteFlag | None:
        for event in events:
            if not isinstance(event, ModelMessage):
                continue
            text = event.content.lower()

            for term in ADVANCED_MARKERS:
                if term in text and term not in self._flagged_terms:
                    self._flagged_terms.add(term)
                    return SatelliteFlag(
                        source=self.name,
                        severity=Severity.WARNING,
                        message=(
                            f"Content mentions '{term}' which may be too "
                            f"advanced for {self._audience}. Consider "
                            f"simplifying or adding prerequisite context."
                        ),
                    )

            for term in TOO_BASIC_MARKERS:
                if term in text and term not in self._flagged_terms:
                    self._flagged_terms.add(term)
                    return SatelliteFlag(
                        source=self.name,
                        severity=Severity.INFO,
                        message=(
                            f"Content uses phrase '{term}' — ensure the lesson "
                            f"conveys that recursion is a distinct paradigm, "
                            f"not just another way to loop."
                        ),
                    )
        return None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

planet_config = OpenAIConfig(
    model="gpt-4o",
    api_key=os.environ.get("OPENAI_API_KEY"),
    streaming=True,
)

satellite_config = OpenAIConfig(
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
    streaming=True,
)

# ---------------------------------------------------------------------------
# Planet agent
# ---------------------------------------------------------------------------

PLANET_PROMPT = """\
You are an experienced CS professor designing a lesson for CS101 undergraduates.
These are students in their first or second programming course — they know
loops, functions, and basic data structures, but have NOT seen recursion before.

Your workflow:
1. Briefly outline what the lesson needs to cover (2-3 sentences).
2. Use `spawn_tasks` to delegate FOUR parallel content sections:

   a) "CONCEPTUAL EXPLANATION: Write a clear, intuitive explanation of recursion
      for students who have never seen it. Use a real-world analogy (not just
      Russian dolls — be creative). Explain base case vs recursive case.
      Show the call stack visually with a simple example. ~400 words."

   b) "PRACTICE PROBLEMS: Create 3 practice problems with increasing difficulty:
      (1) Direct application (e.g., factorial or sum), (2) Moderate challenge
      (e.g., reverse a string or check palindrome), (3) Stretch goal
      (e.g., generate permutations or flatten nested lists). For each: problem
      statement, hint, and complete Python solution with comments. Target: CS101."

   c) "COMMON MISCONCEPTIONS: Identify the top 4-5 misconceptions students have
      when first learning recursion. For each: what students think, why it's
      wrong, a concrete example showing the correct behavior, and a tip for
      avoiding the confusion. Draw from real teaching experience. ~400 words."

   d) "REAL-WORLD APPLICATIONS: Describe 3-4 real-world applications where
      recursion is the natural solution (file system traversal, parsing,
      fractals, divide-and-conquer algorithms). For each: brief description of
      the problem, why recursion fits, and a short Python code sketch (<15 lines).
      Keep it accessible for CS101. ~400 words."

3. Assemble the satellite results into a complete lesson plan with:
   - Learning objectives (3-4 bullet points)
   - The four content sections (edited for flow and consistency)
   - A summary with key takeaways
   - Suggested homework assignment (one sentence)

Keep your own text focused on framing and transitions. Let the satellites
create the detailed content.
"""

planet = PlanetAgent(
    "Professor",
    prompt=PLANET_PROMPT,
    config=planet_config,
    satellite_config=satellite_config,
    satellite_prompt=(
        "You are an experienced CS teaching assistant creating educational "
        "content for CS101 undergraduates learning recursion for the first time. "
        "Be clear, concrete, and encouraging. Use Python for all code examples. "
        "Avoid jargon. Return well-structured content with clear headers."
    ),
    satellites=[
        TokenMonitor(warn_threshold=25_000, alert_threshold=60_000),
        DifficultyMonitor(audience="CS101 undergraduates"),
    ],
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def main() -> None:
    print(f"{'=' * 70}")
    print("Lesson Plan Generator: Introduction to Recursion")
    print(f"{'=' * 70}\n")

    stream = MemoryStream()
    _speaker = ""

    async def _on_event(event: object) -> None:
        nonlocal _speaker
        if isinstance(event, TaskSatelliteRequest):
            _speaker = ""
            label = event.task[:70].replace("\n", " ")
            print(
                f"\n  \033[32m[spawn]\033[0m {event.satellite_name}: {label}...",
                flush=True,
            )
        elif isinstance(event, TaskSatelliteProgress):
            if _speaker != event.satellite_name:
                _speaker = event.satellite_name
                print(
                    f"\n\033[2m  [Satellite: {event.satellite_name}] >\033[0m\n",
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
                print(f"\n\033[1;36m  [Planet: Professor] >\033[0m\n", flush=True)
            sys.stdout.write(event.content)
            sys.stdout.flush()

    stream.subscribe(_on_event)

    print("Designing lesson...\n", flush=True)
    conversation = await planet.ask(
        "Create a complete lesson plan for 'Introduction to Recursion' "
        "for a CS101 undergraduate class.",
        stream=stream,
    )
    print()

    for sat in planet._satellites:
        if isinstance(sat, TokenMonitor):
            print(f"\n[Token usage: {sat.total_tokens:,} total tokens]")


if __name__ == "__main__":
    asyncio.run(main())
