"""Shared web tools for playground examples.

Requires:
    pip install tavily-python beautifulsoup4 requests
    export TAVILY_API_KEY=tvly-...
"""

import os

from autogen.beta.tools import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information on a topic.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (default 5).
    """
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(query, max_results=max_results)

    parts = []
    for r in response.get("results", []):
        parts.append(
            f"**{r['title']}**\n{r['url']}\n{r.get('content', '')[:300]}"
        )
    return "\n\n---\n\n".join(parts) if parts else "No results found."


@tool
def browse_url(url: str) -> str:
    """Fetch a URL and extract its main text content.

    Use this to read full articles, papers, or documentation pages
    discovered via web_search.

    Args:
        url: The URL to fetch.
    """
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"},
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Truncate to avoid context overflow
    if len(text) > 8000:
        text = text[:8000] + "\n\n[... truncated]"
    return text
