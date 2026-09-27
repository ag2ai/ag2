# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""An agent with a response schema can be served over MCP under the checker.

Checked by ``test/typing/test_fixtures.py``; not imported by the suite. The fixture
carries no expectations: a clean run is the assertion.
"""

from pydantic import BaseModel

from ag2 import Agent
from ag2.mcp import MCPServer


class Weather(BaseModel):
    city: str


# The bridge derives the tool's output schema from ``response_schema``, so an
# ``Agent[Weather]`` is what it is for, not only the default ``Agent[str]``.
# Bound to a name first, as a caller holds one: inline, the argument would be
# inferred from the parameter instead.
agent = Agent("weather", response_schema=Weather)
server = MCPServer(agent)
