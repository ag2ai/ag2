# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")


@mcp.tool()  # type: ignore[misc]
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()  # type: ignore[misc]
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    mcp.run(transport="stdio")
