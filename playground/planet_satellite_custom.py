"""Planet-Satellite Architecture: Custom Satellite Plug-in

Demonstrates how to create a custom satellite that monitors specific
domain logic. This example builds a "fact checker" satellite that flags
when the model makes claims about specific topics.

This shows the plug-in extensibility of the satellite architecture:
any class implementing the Satellite protocol can be attached.

Usage:
    export OPENAI_API_KEY=sk-...
    python playground/planet_satellite_custom.py
"""

import asyncio
import os
from autogen.beta.annotations import Context
from autogen.beta.config.openai import OpenAIConfig
from autogen.beta.events import BaseEvent, ModelMessage
from autogen.beta.satellites import (
    BaseSatellite,
    LoopDetector,
    OnEvent,
    PlanetAgent,
    SatelliteFlag,
    Severity,
    TokenMonitor,
)
from autogen.beta.stream import MemoryStream


# ---------------------------------------------------------------------------
# Custom satellite: keyword watchdog
# ---------------------------------------------------------------------------


class KeywordWatchdog(BaseSatellite):
    """Flags when model output contains watched keywords.

    This is a simple example of a domain-specific satellite. In production
    you might check for compliance terms, competitor mentions, PII patterns,
    or use an LLM for nuanced classification.
    """

    def __init__(
        self,
        keywords: list[str],
        severity: Severity = Severity.WARNING,
        *,
        name: str = "keyword-watchdog",
    ) -> None:
        super().__init__(name, trigger=OnEvent(ModelMessage))
        self._keywords = [kw.lower() for kw in keywords]
        self._severity = severity
        self._flagged: set[str] = set()

    async def process(
        self, events: list[BaseEvent], ctx: Context
    ) -> SatelliteFlag | None:
        for event in events:
            if not isinstance(event, ModelMessage):
                continue

            text = event.content.lower()
            for kw in self._keywords:
                if kw in text and kw not in self._flagged:
                    self._flagged.add(kw)
                    return SatelliteFlag(
                        source=self.name,
                        severity=self._severity,
                        message=(
                            f"Output mentions watched keyword: '{kw}'. "
                            "Please verify accuracy of claims related to "
                            "this topic before including in the final output."
                        ),
                    )
        return None

    def detach(self) -> None:
        super().detach()
        self._flagged.clear()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

planet_config = OpenAIConfig(
    model="gpt-5.4",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

satellite_config = OpenAIConfig(
    model="gpt-5.2",
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# ---------------------------------------------------------------------------
# Planet agent with custom + built-in satellites
# ---------------------------------------------------------------------------

planet = PlanetAgent(
    "Market Analyst",
    prompt=(
        "You are a market analyst writing a brief on AI chip manufacturers. "
        "Use `spawn_tasks` to have satellites research: "
        "(1) NVIDIA's market position, (2) AMD's AI strategy, "
        "(3) emerging competitors. Then synthesise into a market brief."
    ),
    config=planet_config,
    satellite_config=satellite_config,
    satellites=[
        # Built-in satellites
        TokenMonitor(warn_threshold=15_000, alert_threshold=40_000),
        LoopDetector(repeat_threshold=3),
        # Custom satellite -- flags mentions of specific companies
        # so the planet can double-check accuracy
        KeywordWatchdog(
            keywords=["market share", "revenue", "billion", "percent"],
            severity=Severity.WARNING,
        ),
    ],
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def main() -> None:
    print(f"{'=' * 70}")
    print("Market Analysis with Custom Satellite Monitoring")
    print(f"  Planet model:    {planet_config.model}")
    print(f"  Satellite model: {satellite_config.model}")
    print(f"{'=' * 70}\n")

    stream = MemoryStream()

    async def _log(event: object) -> None:
        if isinstance(event, SatelliteFlag):
            print(
                f"  \033[33m[flag]\033[0m  [{event.severity}] {event.message}",
                flush=True,
            )

    stream.subscribe(_log)

    conversation = await planet.ask(
        "Write a market analysis brief on the AI chip industry.",
        stream=stream,
    )

    print(f"\n{'=' * 70}")
    print("MARKET BRIEF")
    print(f"{'=' * 70}\n")

    if conversation.message and conversation.message.message:
        print(conversation.message.message.content)

    for sat in planet._satellites:
        if isinstance(sat, TokenMonitor):
            print(f"\n[Token usage: {sat.total_tokens:,} total tokens]")


if __name__ == "__main__":
    asyncio.run(main())
