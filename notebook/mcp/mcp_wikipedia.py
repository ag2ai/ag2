import argparse
from pathlib import Path
from typing import Dict, List

import wikipedia
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("WikipediaServer")

parser = argparse.ArgumentParser(description="Wikipedia MCP Server")
parser.add_argument("--storage-path", required=True, help="Path to store downloaded articles")
args, unknown = parser.parse_known_args()

STORAGE_PATH = Path(args.storage_path).resolve()
STORAGE_PATH.mkdir(parents=True, exist_ok=True)


@mcp.tool()
def search_wikipedia(query: str, max_results: int = 3) -> List[str]:
    """Search Wikipedia and return titles of top articles."""
    return wikipedia.search(query, results=max_results)


@mcp.tool()
def download_article(title: str) -> str:
    """Download a Wikipedia article and store it as a text file."""
    try:
        page = wikipedia.page(title)
        file_path = STORAGE_PATH / f"{title.replace(' ', '_')}.txt"
        file_path.write_text(page.content, encoding="utf-8")
        return f"Downloaded '{title}' to {file_path.name}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous title '{title}'. Possible options: {', '.join(e.options[:5])}..."
    except wikipedia.exceptions.PageError:
        return f"Article '{title}' not found."


@mcp.tool()
def list_articles() -> List[str]:
    """List downloaded Wikipedia articles."""
    return [f.name for f in STORAGE_PATH.glob("*.txt")]


@mcp.tool()
def get_article_summary(title: str) -> Dict[str, str]:
    """Extract and return the title and summary of a Wikipedia article."""
    try:
        summary = wikipedia.summary(title)
        return {"title": title, "summary": summary}
    except wikipedia.exceptions.PageError:
        return {"error": f"Article '{title}' not found"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wikipedia MCP Server")
    parser.add_argument("transport", choices=["stdio", "sse"], help="Transport mode")
    parser.add_argument("--storage-path", required=True, help="Path to store articles")
    args = parser.parse_args()

    mcp.run(transport=args.transport)
