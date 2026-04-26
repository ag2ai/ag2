# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta import Agent
from autogen.beta.a2a.card import build_card
from autogen.beta.testing import TestConfig


def test_uses_agent_name() -> None:
    agent = Agent("specialist", "You are a helper", config=TestConfig("ok"))

    card = build_card(agent, url="http://localhost:8000")

    assert card.name == "specialist"
    assert card.url == "http://localhost:8000"


def test_description_is_joined_system_prompt() -> None:
    agent = Agent("specialist", "You are a helper", config=TestConfig("ok"))

    card = build_card(agent, url="http://localhost:8000")

    assert card.description == "You are a helper"


def test_streaming_capability_declared() -> None:
    agent = Agent("a", "p", config=TestConfig("ok"))

    card = build_card(agent, url="http://x")

    assert card.capabilities.streaming is True


def test_text_modes_default() -> None:
    agent = Agent("a", "p", config=TestConfig("ok"))

    card = build_card(agent, url="http://x")

    assert card.default_input_modes == ["text"]
    assert card.default_output_modes == ["text"]


def test_skills_empty_by_default() -> None:
    agent = Agent("a", "p", config=TestConfig("ok"))

    card = build_card(agent, url="http://x")

    assert card.skills == []
