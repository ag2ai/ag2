"""Planet-Satellite Architecture: Competitive Landscape Analysis

Demonstrates real web search (Tavily API) combined with a custom "claim
validator" satellite that flags unsubstantiated numerical claims.

The planet researches a market using web_search, then delegates competitor
deep-dives to satellites. Each satellite receives real search data in its
task description, so its analysis is grounded in current information.

Architecture:
    Planet (main model)     -- researches market, delegates competitor analyses
    Satellites (lighter)    -- each analyses one competitor based on search data
    TokenMonitor            -- tracks token usage
    ClaimValidator          -- custom satellite flagging unsourced statistics

Requirements:
    pip install tavily-python
    export TAVILY_API_KEY=tvly-...
    export OPENAI_API_KEY=sk-...

Usage:
    python playground/planet_satellite_competitive.py
"""

import asyncio
import os
import re
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
    TaskSatelliteRequest,
    TaskSatelliteResult,
    TokenMonitor,
)
from autogen.beta.stream import MemoryStream

# Import shared web search tool
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from tools.web import web_search

# ---------------------------------------------------------------------------
# Custom satellite: claim validator
# ---------------------------------------------------------------------------

# Pattern for numerical claims (percentages, dollar amounts, multiples)
_CLAIM_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(%|percent|billion|million|trillion|x\b|times\b)",
    re.IGNORECASE,
)


class ClaimValidator(BaseSatellite):
    """Flags numerical claims that may need source verification.

    Watches for statistics, market share figures, and financial claims
    in the planet's output. Doesn't block — just advises the planet to
    cite sources or add caveats.
    """

    def __init__(self, threshold: int = 3) -> None:
        super().__init__("claim-validator", trigger=OnEvent(ModelMessage))
        self._claim_count = 0
        self._threshold = threshold

    async def process(
        self, events: list[BaseEvent], ctx: Context
    ) -> SatelliteFlag | None:
        for event in events:
            if not isinstance(event, ModelMessage):
                continue

            claims = _CLAIM_PATTERN.findall(event.content)
            self._claim_count += len(claims)

            if claims and self._claim_count >= self._threshold:
                sample = ", ".join(
                    f"{num}{unit}" for num, unit in claims[:3]
                )
                self._claim_count = 0  # reset after flagging
                return SatelliteFlag(
                    source=self.name,
                    severity=Severity.WARNING,
                    message=(
                        f"Multiple numerical claims detected ({sample}, ...). "
                        "Ensure statistics are sourced from the search results "
                        "or clearly marked as estimates. Add 'according to...' "
                        "attribution where possible."
                    ),
                )
        return None


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
You are a competitive intelligence analyst. You research markets using real
web search and produce grounded, evidence-based competitive analyses.

Your workflow:
1. Use `web_search` to research the overall market landscape (2-3 searches).
2. Identify the key players and recent developments from the search results.
3. Use `spawn_tasks` to delegate competitor deep-dives in parallel (one per
   major competitor, 3-4 total). IMPORTANT: Include the relevant search
   results/snippets in each task description so the satellite's analysis
   is grounded in real data, not hallucinated.
4. Synthesise into a competitive landscape brief:
   - Market overview (size, growth, key trends)
   - Competitor profiles (strengths, weaknesses, recent moves)
   - Competitive dynamics (who's gaining/losing, why)
   - Strategic implications and opportunities
   - Sources (cite URLs from search results)

Always attribute claims to sources. If data is unavailable, say so.
"""

planet = PlanetAgent(
    "Intelligence Analyst",
    prompt=PLANET_PROMPT,
    config=planet_config,
    satellite_config=satellite_config,
    satellite_prompt=(
        "You are a competitive intelligence researcher analysing one specific "
        "company. Base your analysis ONLY on the provided search data — do not "
        "invent statistics. Structure: Overview, Recent Strategy, Strengths, "
        "Weaknesses, Outlook. Cite sources where possible. 300-500 words."
    ),
    satellites=[
        TokenMonitor(warn_threshold=25_000, alert_threshold=60_000),
        ClaimValidator(threshold=3),
    ],
    tools=[web_search],
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def main() -> None:
    topic = "the AI code assistant market (GitHub Copilot, Cursor, Codeium, etc.)"

    print(f"{'=' * 70}")
    print(f"Competitive Landscape Analysis")
    print(f"Topic: {topic}")
    print(f"  Planet model:    {planet_config.model}")
    print(f"  Satellite model: {satellite_config.model}")
    print(f"{'=' * 70}\n")

    stream = MemoryStream()
    _speaker = ""

    async def _on_event(event: object) -> None:
        nonlocal _speaker
        if isinstance(event, TaskSatelliteRequest):
            _speaker = ""
            label = event.task[:70].replace("\n", " ")
            print(
                f"\n  \033[32m[spawn]\033[0m {event.satellite_name} "
                f"({satellite_config.model}): {label}...",
                flush=True,
            )
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
                print(f"\n\033[1;36m  [Planet: Intelligence Analyst ({planet_config.model})] >\033[0m\n", flush=True)
            sys.stdout.write(event.content)
            sys.stdout.flush()

    stream.subscribe(_on_event)

    print("Researching market...\n", flush=True)
    conversation = await planet.ask(
        f"Produce a competitive landscape analysis of {topic}",
        stream=stream,
    )
    print()

    for sat in planet._satellites:
        if isinstance(sat, TokenMonitor):
            print(f"\n[Token usage: {sat.total_tokens:,} total tokens]")


if __name__ == "__main__":
    asyncio.run(main())
