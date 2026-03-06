"""Planet-Satellite Architecture: Scientific Literature Review

Demonstrates real web search (Tavily) and URL browsing (requests + BeautifulSoup)
for conducting a systematic literature review. The planet searches for recent
research, reads key articles, then delegates analysis of different research
themes to satellites.

This is the most tool-heavy example — the planet actively gathers information
from the web before delegating analytical work.

Architecture:
    Planet (main model)     -- searches, reads papers/articles, delegates analysis
    Satellites (lighter)    -- each analyses one research theme with gathered data
    TokenMonitor            -- tracks token usage
    LoopDetector            -- flags circular research patterns

Requirements:
    pip install tavily-python beautifulsoup4 requests
    export TAVILY_API_KEY=tvly-...
    export OPENAI_API_KEY=sk-...

Usage:
    python playground/planet_satellite_scientific.py
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
    TaskSatelliteProgress,
    TaskSatelliteRequest,
    TaskSatelliteResult,
    TokenMonitor,
)
from autogen.beta.stream import MemoryStream

# Import shared web tools
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from tools.web import browse_url, web_search

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
You are a scientific research assistant conducting a literature review.
You have real web search and URL browsing tools at your disposal.

Your systematic workflow:
1. SEARCH PHASE: Use `web_search` to find recent research on the topic.
   Run 3-5 targeted searches with different angles (e.g., "recent advances",
   "clinical trials", "challenges", "review papers").

2. DEEP READ PHASE: Use `browse_url` on the 3-4 most relevant/authoritative
   URLs from the search results (prefer review articles, reputable journals,
   institutional sources). This gives you detailed content beyond snippets.

3. THEME IDENTIFICATION: Based on what you've gathered, identify 3-4 major
   research themes or sub-topics that warrant deeper analysis.

4. DELEGATION PHASE: Use `spawn_tasks` to delegate each theme to a satellite.
   CRITICAL: Include ALL relevant search results and browsed content in each
   task description. Satellites cannot search — they can only work with the
   data you provide.

5. SYNTHESIS: Combine satellite analyses into a structured literature review:
   - Abstract (150 words)
   - Introduction and Background
   - Thematic Analysis (one section per theme)
   - Current Challenges and Open Questions
   - Future Directions
   - Key References (URLs from your searches)

Be rigorous. Cite sources. Distinguish established findings from preliminary
results. Note when evidence is limited or conflicting.
"""

planet = PlanetAgent(
    "Research Lead",
    prompt=PLANET_PROMPT,
    config=planet_config,
    satellite_config=satellite_config,
    satellite_prompt=(
        "You are a scientific researcher analysing one specific theme from a "
        "literature review. Work ONLY with the provided data — do not invent "
        "findings or citations. Structure: Key Findings, Methodology Trends, "
        "Evidence Quality, Gaps in Knowledge. Be precise and cite sources "
        "from the provided data. 400-600 words."
    ),
    satellites=[
        TokenMonitor(warn_threshold=40_000, alert_threshold=90_000),
        LoopDetector(window_size=10, repeat_threshold=3),
    ],
    tools=[web_search, browse_url],
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def main() -> None:
    topic = (
        "Recent advances in CRISPR-based gene therapy for genetic diseases, "
        "focusing on in vivo delivery methods, clinical trial results, and "
        "safety profiles (2024-2025)"
    )

    print(f"{'=' * 70}")
    print("Scientific Literature Review")
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
                print(f"\n\033[1;36m  [Planet: Research Lead ({planet_config.model})] >\033[0m\n", flush=True)
            sys.stdout.write(event.content)
            sys.stdout.flush()

    stream.subscribe(_on_event)

    print("Starting literature review...\n", flush=True)
    conversation = await planet.ask(
        f"Conduct a systematic literature review on: {topic}",
        stream=stream,
    )
    print()

    for sat in planet._satellites:
        if isinstance(sat, TokenMonitor):
            print(f"\n[Token usage: {sat.total_tokens:,} total tokens]")


if __name__ == "__main__":
    asyncio.run(main())
