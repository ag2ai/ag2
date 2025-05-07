# mcp_server.py
import argparse
import os
from pathlib import Path
from typing import List

from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("FilesystemServer")

# This will be defined in main
CONTEXT_PATH = None


@mcp.tool()
def list_files(relative_path: str = "") -> List[str]:
    """
    List files and directories under CONTEXT_PATH/relative_path. Pass empty string to list root.
    """
    path = (CONTEXT_PATH / relative_path).resolve()
    if not str(path).startswith(str(CONTEXT_PATH)):
        return [f"Access denied: {relative_path}"]
    if not path.exists() or not path.is_dir():
        return [f"Not a directory: {relative_path}"]
    return os.listdir(path)


@mcp.tool()
def read_file(relative_path: str) -> str:
    """
    Read and return the contents of the file at CONTEXT_PATH/relative_path.
    """
    path = (CONTEXT_PATH / relative_path).resolve()
    if not str(path).startswith(str(CONTEXT_PATH)):
        return f"Access denied: {relative_path}"
    if not path.exists() or not path.is_file():
        return f"Not a file: {relative_path}"
    return path.read_text()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Filesystem Server")
    parser.add_argument(
        "transport",
        choices=["stdio", "sse"],
        help="Transport mode (stdio or sse)",
    )
    parser.add_argument("--context-path", required=True, help="Path to context docs")
    args = parser.parse_args()

    CONTEXT_PATH = Path(args.context_path).resolve()
    CONTEXT_PATH.mkdir(parents=True, exist_ok=True)

    mcp.run(transport=args.transport)
