# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Elicitation over a real subprocess, the way AG2 actually drives a CLI agent.

``test_elicitation.py`` drives the in-process double, which calls the bridge
directly. That leaves two things it cannot prove: that the capability really
reaches the agent through ``initialize``, and that ``elicitation/create`` really
dispatches to the bridge rather than being answered with method-not-found by the
SDK's router. Both are exercised here against a subprocess speaking ACP on stdio.
"""

import sys
from pathlib import Path

import pytest

from ag2 import Agent
from ag2.acp import ACPConfig
from ag2.events import HumanInputRequest

from ._elicit_agent import AUTH_URL

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config(**overrides: str) -> ACPConfig:
    return ACPConfig(
        command=[sys.executable, "-m", "test.acp._elicit_agent"],
        cwd=str(REPO_ROOT),
        **overrides,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_the_agent_can_ask_and_gets_the_answer() -> None:
    prompts: list[str] = []

    def human(event: HumanInputRequest) -> str:
        prompts.append(event.content)
        return "yes"

    cfg = _config()
    agent = Agent("acp", config=cfg, hitl_hook=human)

    try:
        reply = await agent.ask("do the thing")
    finally:
        await cfg.aclose()

    # The agent reports the action AG2 sent back, so this is the response as the
    # agent itself saw it after a full round trip over the wire.
    assert reply.body == "accept"
    [prompt] = prompts
    assert AUTH_URL in prompt


@pytest.mark.asyncio
async def test_declining_the_policy_stops_the_agent_asking_at_all() -> None:
    prompts: list[str] = []

    def human(event: HumanInputRequest) -> str:
        prompts.append(event.content)
        return "yes"

    cfg = _config(elicitation_policy="decline")
    agent = Agent("acp", config=cfg, hitl_hook=human)

    try:
        reply = await agent.ask("do the thing")
    finally:
        await cfg.aclose()

    # The agent saw no elicitation capability in `initialize`, so it never asked.
    assert reply.body == "not advertised"
    assert prompts == []
