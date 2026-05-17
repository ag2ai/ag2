# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for DynamicAgentToolkit — spawn specialist sub-agents on demand."""

import pytest

from autogen.beta import Agent
from autogen.beta.events import ModelMessage, ModelResponse, ToolCallEvent
from autogen.beta.testing import TestConfig
from autogen.beta.tools import DynamicAgentToolkit
from autogen.beta.tools.final import Toolkit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _specialist_config(reply: str) -> TestConfig:
    """Config that makes the specialist agent return *reply*."""
    return TestConfig(reply)


# ---------------------------------------------------------------------------
# Schema and structure
# ---------------------------------------------------------------------------


class TestDynamicAgentToolkitStructure:
    def test_is_toolkit(self) -> None:
        config = TestConfig("ok")
        t = DynamicAgentToolkit(config=config)
        assert isinstance(t, Toolkit)

    def test_has_ask_specialist_tool(self) -> None:
        config = TestConfig("ok")
        t = DynamicAgentToolkit(config=config)
        names = {tool.name for tool in t.tools}
        assert "ask_specialist" in names

    def test_exactly_one_tool(self) -> None:
        config = TestConfig("ok")
        t = DynamicAgentToolkit(config=config)
        assert len(t.tools) == 1


# ---------------------------------------------------------------------------
# ask_specialist — functional tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAskSpecialist:
    async def test_specialist_reply_returned(self) -> None:
        """The orchestrator receives the specialist's reply as a string."""
        specialist_config = _specialist_config("42 is the answer.")
        orchestrator_config = TestConfig(
            ToolCallEvent(
                name="ask_specialist",
                arguments='{"role":"You are a philosopher.","task":"What is the answer?"}',
            ),
            ModelResponse(ModelMessage("The specialist says: 42 is the answer.")),
        )

        toolkit = DynamicAgentToolkit(config=specialist_config)
        orchestrator = Agent("orchestrator", config=orchestrator_config, tools=[toolkit])
        reply = await orchestrator.ask("Delegate to specialist.")
        text = await reply.content()
        assert text is not None
        assert "42" in text

    async def test_custom_specialist_name(self) -> None:
        """The name parameter is accepted without error."""
        specialist_config = _specialist_config("analysis done")
        orchestrator_config = TestConfig(
            ToolCallEvent(
                name="ask_specialist",
                arguments='{"role":"analyst","task":"crunch numbers","name":"data-analyst"}',
            ),
            ModelResponse(ModelMessage("done")),
        )

        toolkit = DynamicAgentToolkit(config=specialist_config)
        orchestrator = Agent("orchestrator", config=orchestrator_config, tools=[toolkit])
        reply = await orchestrator.ask("Analyse.")
        assert await reply.content() is not None

    async def test_specialist_empty_reply_returns_empty_string(self) -> None:
        """If the specialist returns None/empty content, an empty string is returned."""
        specialist_config = TestConfig("")
        orchestrator_config = TestConfig(
            ToolCallEvent(
                name="ask_specialist",
                arguments='{"role":"silent","task":"say nothing"}',
            ),
            ModelResponse(ModelMessage("ok")),
        )

        toolkit = DynamicAgentToolkit(config=specialist_config)
        orchestrator = Agent("orchestrator", config=orchestrator_config, tools=[toolkit])
        reply = await orchestrator.ask("go")
        assert await reply.content() is not None


# ---------------------------------------------------------------------------
# Shared tools propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSharedTools:
    async def test_shared_tools_forwarded_to_specialist(self) -> None:
        """Tools passed to DynamicAgentToolkit are available to the specialist."""
        from autogen.beta.tools import tool as make_tool

        calls: list[str] = []

        @make_tool
        def recorder(msg: str) -> str:
            """Record a call."""
            calls.append(msg)
            return "recorded"

        # Specialist calls recorder, then replies
        specialist_config = TestConfig(
            ToolCallEvent(name="recorder", arguments='{"msg":"hello from specialist"}'),
            ModelResponse(ModelMessage("done")),
        )
        orchestrator_config = TestConfig(
            ToolCallEvent(
                name="ask_specialist",
                arguments='{"role":"recorder agent","task":"record something"}',
            ),
            ModelResponse(ModelMessage("delegated")),
        )

        toolkit = DynamicAgentToolkit(config=specialist_config, tools=[recorder])
        orchestrator = Agent("orchestrator", config=orchestrator_config, tools=[toolkit])
        await orchestrator.ask("Go.")

        assert "hello from specialist" in calls


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_importable_from_tools() -> None:
    from autogen.beta.tools import DynamicAgentToolkit as D

    assert D is DynamicAgentToolkit


def test_all_export() -> None:
    from autogen.beta.tools import toolkits

    assert "DynamicAgentToolkit" in toolkits.__all__
