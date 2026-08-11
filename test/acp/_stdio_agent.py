# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""A standalone ACP agent process, for the stdio end-to-end test.

Run as ``python -m test.acp._stdio_agent``. An ACP Client launches this the same
way it launches Claude Code or Codex: as a subprocess speaking ACP on stdio.

Nothing may be written to stdout except the protocol itself — stdout *is* the
transport.
"""

import asyncio

from ag2 import Agent
from ag2.acp import ACPAgent
from ag2.events import ToolCallEvent
from ag2.testing import TestConfig


def build_server() -> ACPAgent:
    agent = Agent(
        "workie",
        config=TestConfig(ToolCallEvent(name="add", arguments='{"a": 100, "b": 100}'), "200"),
    )

    @agent.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return ACPAgent(agent, name="workie", version="test")


if __name__ == "__main__":
    asyncio.run(build_server().run_stdio())
