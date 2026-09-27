# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``mcp_tool``'s display metadata takes a ``Variable`` under the checker.

Checked by ``test/typing/test_fixtures.py``; not imported by the suite. The fixture
carries no expectations: a clean run is the assertion.
"""

from ag2 import Variable
from ag2.mcp import MCPFunctionTool, mcp_tool


# ``title`` and ``annotations`` are resolved per request, like ``meta``.
@mcp_tool(title=Variable("tool_title"), annotations=Variable("tool_annotations"))
async def read_scope() -> str:
    """Read request-scoped data."""
    return "ok"


def undecorated() -> str:
    """Read request-scoped data."""
    return "ok"


tool: MCPFunctionTool = mcp_tool(undecorated, title=Variable("tool_title"))
