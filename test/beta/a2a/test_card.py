# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import textwrap
from pathlib import Path

import pytest
from a2a.types import AgentCapabilities, AgentSkill

from autogen.beta import Agent
from autogen.beta.a2a.cards import build_card, url_from_card
from autogen.beta.testing import TestConfig
from autogen.beta.tools.builtin.skills import Skill, SkillsTool
from autogen.beta.tools.skills import LocalRuntime, SkillsToolkit


@pytest.fixture
def agent() -> Agent:
    return Agent("specialist", "You are a helper", config=TestConfig("ok"))


class TestBuildCard:
    def test_uses_agent_name_and_url(self, agent: Agent) -> None:
        card = build_card(agent, url="http://localhost:8000")

        assert card.name == "specialist"
        assert url_from_card(card) == "http://localhost:8000"

    def test_description_is_joined_system_prompt(self, agent: Agent) -> None:
        card = build_card(agent, url="http://x")

        assert card.description == "You are a helper"

    def test_streaming_capability_declared(self, agent: Agent) -> None:
        card = build_card(agent, url="http://x")

        assert card.capabilities.streaming is True

    def test_text_modes_default(self, agent: Agent) -> None:
        card = build_card(agent, url="http://x")

        assert card.default_input_modes == ["text"]
        assert card.default_output_modes == ["text"]

    def test_skills_empty_when_agent_has_no_skills_tools(self, agent: Agent) -> None:
        card = build_card(agent, url="http://x")

        assert card.skills == []

    def test_version_default(self, agent: Agent) -> None:
        card = build_card(agent, url="http://x")

        assert card.version == "0.1.0"

    def test_version_override(self, agent: Agent) -> None:
        card = build_card(agent, url="http://x", version="2.5.0")

        assert card.version == "2.5.0"

    def test_description_override(self, agent: Agent) -> None:
        card = build_card(agent, url="http://x", description="custom-bio")

        assert card.description == "custom-bio"

    def test_capabilities_override(self, agent: Agent) -> None:
        caps = AgentCapabilities(streaming=False, push_notifications=True)
        card = build_card(agent, url="http://x", capabilities=caps)

        assert card.capabilities.streaming is False
        assert card.capabilities.push_notifications is True

    def test_input_output_modes_override(self, agent: Agent) -> None:
        card = build_card(
            agent,
            url="http://x",
            default_input_modes=["text", "image/png"],
            default_output_modes=["text", "application/json"],
        )

        assert card.default_input_modes == ["text", "image/png"]
        assert card.default_output_modes == ["text", "application/json"]

    def test_skills_auto_extracted_from_skills_tool(self) -> None:
        agent = Agent(
            "specialist",
            "p",
            config=TestConfig("ok"),
            tools=[SkillsTool("pptx", Skill("xlsx", version="2024-01"))],
        )
        card = build_card(agent, url="http://x")

        assert card.skills == [
            AgentSkill(
                id="pptx",
                name="pptx",
                description="Provider-managed skill pptx",
                tags=[],
            ),
            AgentSkill(
                id="xlsx",
                name="xlsx",
                description="Provider-managed skill xlsx (v2024-01)",
                tags=[],
            ),
        ]

    def test_skills_auto_extracted_from_skills_toolkit(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "recipe-search"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            textwrap.dedent("""\
            ---
            name: recipe-search
            description: Find recipes by ingredients
            ---
            body
            """)
        )

        agent = Agent(
            "specialist",
            "p",
            config=TestConfig("ok"),
            tools=[SkillsToolkit(runtime=LocalRuntime(str(tmp_path)))],
        )
        card = build_card(agent, url="http://x")

        assert card.skills == [
            AgentSkill(
                id="recipe-search",
                name="recipe-search",
                description="Find recipes by ingredients",
                tags=[],
            ),
        ]

    def test_supports_extended_flag_off_by_default(self, agent: Agent) -> None:
        card = build_card(agent, url="http://x")

        assert not card.capabilities.extended_agent_card

    def test_supports_extended_flag_set_when_requested(self, agent: Agent) -> None:
        card = build_card(agent, url="http://x", supports_extended=True)

        assert card.capabilities.extended_agent_card is True
