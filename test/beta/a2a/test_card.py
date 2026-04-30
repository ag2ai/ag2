# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Agent
from autogen.beta.a2a.card import build_card
from autogen.beta.testing import TestConfig


@pytest.fixture
def agent() -> Agent:
    return Agent("specialist", "You are a helper", config=TestConfig("ok"))


class TestBuildCard:
    def test_uses_agent_name_and_url(self, agent: Agent) -> None:
        card = build_card(agent, url="http://localhost:8000")

        assert card.name == "specialist"
        assert card.url == "http://localhost:8000"

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

    def test_skills_empty_by_default(self, agent: Agent) -> None:
        card = build_card(agent, url="http://x")

        assert card.skills == []

    def test_supports_extended_flag_off_by_default(self, agent: Agent) -> None:
        card = build_card(agent, url="http://x")

        assert not card.supports_authenticated_extended_card

    def test_supports_extended_flag_set_when_requested(self, agent: Agent) -> None:
        card = build_card(agent, url="http://x", supports_extended=True)

        assert card.supports_authenticated_extended_card is True
