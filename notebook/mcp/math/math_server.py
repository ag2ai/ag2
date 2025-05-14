# math_server.py
import argparse

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Math Server")
    parser.add_argument("transport", choices=["stdio", "sse"], help="Transport mode (stdio or sse)")
    args = parser.parse_args()

    mcp.run(transport=args.transport)
