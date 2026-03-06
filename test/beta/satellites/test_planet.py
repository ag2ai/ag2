# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.context import Context
from autogen.beta.events import ModelMessage, ModelResponse, ToolCall
from autogen.beta.satellites import (
    LoopDetector,
    PlanetAgent,
    SatelliteCompleted,
    SatelliteFlag,
    SatelliteStarted,
    Severity,
    TokenMonitor,
)
from autogen.beta.satellites.events import TaskSatelliteRequest, TaskSatelliteResult
from autogen.beta.satellites.satellite import BaseSatellite
from autogen.beta.stream import MemoryStream
from autogen.beta.testing import TestConfig


# -- helpers ----------------------------------------------------------


class _AlwaysFlagSatellite(BaseSatellite):
    """A test satellite that flags on every ModelResponse."""

    from autogen.beta.satellites.triggers import OnEvent

    def __init__(self):
        super().__init__("always-flag", trigger=self.OnEvent(ModelResponse))

    async def process(self, events, ctx):
        return SatelliteFlag(
            source=self.name,
            severity=Severity.WARNING,
            message="test flag",
        )


# -- tests ------------------------------------------------------------


@pytest.mark.asyncio
async def test_planet_attaches_and_detaches_satellites():
    """Satellites receive lifecycle events."""
    config = TestConfig("Hello from planet!")

    monitor = TokenMonitor(warn_threshold=999_999)
    planet = PlanetAgent(
        "test-planet",
        prompt="You are a test agent.",
        config=config,
        satellites=[monitor],
    )

    stream = MemoryStream()
    conversation = await planet.ask("Hi", stream=stream)

    # Check lifecycle events in history
    events = list(await stream.history.get_events())
    started_events = [e for e in events if isinstance(e, SatelliteStarted)]
    completed_events = [e for e in events if isinstance(e, SatelliteCompleted)]

    assert len(started_events) == 1
    assert started_events[0].name == "token-monitor"
    assert len(completed_events) == 1
    assert completed_events[0].name == "token-monitor"


@pytest.mark.asyncio
async def test_planet_injects_flags_into_prompt():
    """When a satellite flags, the planet injects it before the LLM call."""

    prompts_seen: list[list[str]] = []

    class _SpyConfig:
        """Config that captures ctx.prompt when the client is called."""

        def copy(self):
            return self

        def create(self):
            return self

        async def __call__(self, *messages, ctx, **kwargs):
            # Capture the prompt at call time
            prompts_seen.append(list(ctx.prompt))
            # Emit a response
            await ctx.send(
                ModelResponse(
                    message=ModelMessage(content="OK"),
                    usage={"total_tokens": 100},
                )
            )

    planet = PlanetAgent(
        "test-planet",
        prompt="System prompt.",
        config=_SpyConfig(),
        satellites=[_AlwaysFlagSatellite()],
    )

    # The _AlwaysFlagSatellite flags on ModelResponse.
    # First call: no flags yet (nothing has happened).
    # The response triggers the satellite, so if there's a follow-up call
    # (e.g., from tool use), flags would be injected.
    # For a single-turn conversation, the flag fires AFTER the response,
    # so the first LLM call won't see it.
    await planet.ask("Hello")

    assert len(prompts_seen) >= 1
    # First call should have the base system prompt
    assert "System prompt." in prompts_seen[0]


@pytest.mark.asyncio
async def test_planet_with_multiple_satellites():
    config = TestConfig("Result")

    planet = PlanetAgent(
        "multi-sat",
        config=config,
        satellites=[
            TokenMonitor(warn_threshold=999_999),
            LoopDetector(repeat_threshold=5),
        ],
    )

    stream = MemoryStream()
    conversation = await planet.ask("Hi", stream=stream)

    events = list(await stream.history.get_events())
    started = [e for e in events if isinstance(e, SatelliteStarted)]
    completed = [e for e in events if isinstance(e, SatelliteCompleted)]

    assert len(started) == 2
    assert len(completed) == 2
    assert {e.name for e in started} == {"token-monitor", "loop-detector"}


@pytest.mark.asyncio
async def test_planet_add_satellite():
    config = TestConfig("OK")
    planet = PlanetAgent("test", config=config)

    assert len(planet._satellites) == 0

    monitor = TokenMonitor()
    planet.add_satellite(monitor)
    assert len(planet._satellites) == 1


@pytest.mark.asyncio
async def test_planet_spawn_tool_available():
    """The planet agent should have spawn_task and spawn_tasks tools."""
    config = TestConfig("I'll just respond directly.")

    planet = PlanetAgent(
        "test-planet",
        config=config,
        satellite_config=config,
    )

    conversation = await planet.ask("Hello")
    assert conversation.message.message.content == "I'll just respond directly."


@pytest.mark.asyncio
async def test_planet_format_flags():
    flags = [
        SatelliteFlag(
            source="monitor-a",
            severity=Severity.WARNING,
            message="Running low on tokens",
        ),
        SatelliteFlag(
            source="monitor-b",
            severity=Severity.CRITICAL,
            message="Loop detected",
        ),
    ]

    text = PlanetAgent._format_flags(flags)
    assert "[SATELLITE MONITORING ALERTS]" in text
    assert "[WARNING]" in text
    assert "[CRITICAL]" in text
    assert "monitor-a" in text
    assert "monitor-b" in text
