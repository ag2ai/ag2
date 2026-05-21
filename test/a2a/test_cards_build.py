# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen import ConversableAgent
from autogen.a2a import A2aAgentServer, CardSettings


def test_default_card() -> None:
    agent = ConversableAgent(name="test")

    server = A2aAgentServer(agent)

    card = server.card
    assert card.name == "test"
    assert card.url == "http://localhost:8000"
    assert card.description == agent.description
    assert not card.supports_authenticated_extended_card


def test_custom_autocreated_card() -> None:
    agent = ConversableAgent(name="test", description="test-description")

    server = A2aAgentServer(agent, url="http://0.0.0.0:8000")

    card = server.card
    assert card.name == "test"
    assert card.description == "test-description"
    assert card.url == "http://0.0.0.0:8000"


def test_card_settings_overrides_agent() -> None:
    agent = ConversableAgent(name="test")

    server = A2aAgentServer(
        agent,
        agent_card=CardSettings(
            name="another-name",
            description="another-description",
        ),
    )

    card = server.card
    assert card.name == "another-name"
    assert card.description == "another-description"


def test_card_settings_overrides_url() -> None:
    agent = ConversableAgent(name="test")

    with pytest.warns(RuntimeWarning):
        server = A2aAgentServer(
            agent,
            url="http://0.0.0.0:8001",
            agent_card=CardSettings(
                url="http://0.0.0.0:8000",
            ),
        )

    assert server.card.url == "http://0.0.0.0:8000"


def test_extended_card() -> None:
    agent = ConversableAgent(name="test", description="test-description")

    server = A2aAgentServer(
        agent,
        url="http://0.0.0.0:8000",
        extended_agent_card=CardSettings(),
    )

    assert server.card.supports_authenticated_extended_card

    card = server.extended_agent_card
    assert card.name == "test"
    assert card.description == "test-description"
    assert card.url == "http://0.0.0.0:8000"


def test_extended_card_settings_overrides_agent() -> None:
    agent = ConversableAgent(name="test")

    server = A2aAgentServer(
        agent,
        extended_agent_card=CardSettings(
            name="another-name",
            description="another-description",
        ),
    )

    card = server.extended_agent_card
    assert card.name == "another-name"
    assert card.description == "another-description"


def test_extended_card_settings_overrides_url() -> None:
    agent = ConversableAgent(name="test")

    with pytest.warns(RuntimeWarning):
        server = A2aAgentServer(
            agent,
            url="http://0.0.0.0:8001",
            extended_agent_card=CardSettings(
                url="http://0.0.0.0:8000",
            ),
        )

    assert server.extended_agent_card.url == "http://0.0.0.0:8000"
