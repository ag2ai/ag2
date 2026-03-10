# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""QuickResearchTool for AG2.

Performs parallel web research across multiple queries using Tavily search,
crawls each result with crawl4ai, and optionally summarizes page content via LLM.
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Annotated, Any
from urllib.parse import urlparse

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import
from ....llm_config import LLMConfig
from ....oai.client import OpenAIWrapper
from ... import Tool
from ...dependency_injection import Depends, on

with optional_import_block():
    import tiktoken
    import tldextract
    from bs4 import BeautifulSoup
    from crawl4ai import AsyncWebCrawler
    from tavily import TavilyClient

__all__ = ["QuickResearchTool"]

# ──────────────────────────── constants ────────────────────────────────────────

MAX_QUERIES = 5
CRAWL_TIMEOUT = 10.0
SUMMARIZE_TIMEOUT = 30.0
MAX_INPUT_TOKENS = 15_000
MAX_CHUNK_TOKENS = 15_000
OVERLAP_TOKENS = 200

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".avif"}
MEDIA_EXTS = {".mp4", ".mp3", ".wav", ".ogg", ".webm", ".pdf"}


# ──────────────────────────── HTML cleaning ───────────────────────────────────


def _reg_dom(netloc: str) -> str:
    """Extract registered domain from netloc."""
    ext = tldextract.extract(netloc)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain


def _clean_html(html: str, base_url: str) -> str:
    """Remove non-text elements and off-site / asset links."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["img", "audio", "video", "script", "style"]):
        tag.decompose()

    base_dom = _reg_dom(urlparse(base_url).netloc)
    for a in soup.find_all("a", href=True):
        href = urlparse(str(a["href"]))
        same = _reg_dom(href.netloc) == base_dom if href.netloc else True
        ext = ("." + href.path.rsplit(".", 1)[-1].lower()) if "." in href.path else ""
        if not same or ext in IMAGE_EXTS or ext in MEDIA_EXTS:
            a.unwrap()

    txt = soup.get_text(separator="\n")
    txt = re.sub(r"[ \t]+\n", "\n", txt)
    txt = re.sub(r"\n{2,}", "\n\n", txt)
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    return txt.strip()


# ──────────────────────────── token splitting ─────────────────────────────────


def _split_tokens(text: str, enc: Any) -> list[str]:
    """Split text into chunks with token-based overlap."""
    ids = enc.encode(text)
    out: list[str] = []
    i = 0
    step = MAX_CHUNK_TOKENS - OVERLAP_TOKENS
    while i < len(ids):
        out.append(enc.decode(ids[i : i + MAX_CHUNK_TOKENS]))
        i += step
    return out


# ──────────────────────────── LLM summarization ──────────────────────────────


def _ask_llm(config_list: list[dict[str, Any]], prompt: str, system_message: str) -> str:
    """Call LLM synchronously via OpenAIWrapper (run in thread for async)."""
    try:
        wrapper = OpenAIWrapper(config_list=config_list)
        response = wrapper.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            cache_seed=None,
        )
        texts = wrapper.extract_text_or_completion_object(response)
        if not texts:
            return ""
        result = texts[0]
        # Handle both plain strings and ChatCompletionMessage objects
        if isinstance(result, str):
            return result.strip()
        content = getattr(result, "content", None)
        return content.strip() if content else ""
    except Exception as e:
        print(f"[QuickResearch] LLM call failed: {e}")
        return ""


async def _ask_llm_async(config_list: list[dict[str, Any]], prompt: str, system_message: str) -> str:
    """Call LLM asynchronously by running OpenAIWrapper in a thread."""
    return await asyncio.to_thread(_ask_llm, config_list, prompt, system_message)


async def _summarise_text_async(
    config_list: list[dict[str, Any]],
    enc: Any,
    text: str,
    chunk_prompt: str,
    merger_prompt: str,
    full_prompt: str,
) -> str:
    """Summarise text using chunk + merge strategy with parallel async LLM calls."""
    tokens = enc.encode(text)
    was_truncated = False
    if len(tokens) > MAX_INPUT_TOKENS:
        tokens = tokens[:MAX_INPUT_TOKENS]
        text = enc.decode(tokens)
        was_truncated = True

    if len(tokens) <= MAX_CHUNK_TOKENS:
        summary = await _ask_llm_async(config_list, text, full_prompt)
        if was_truncated:
            summary += (
                "\n\n[Note: Summary based on first portion of content - original page was truncated due to length]"
            )
        return summary

    chunks = _split_tokens(text, enc)
    chunk_tasks = [_ask_llm_async(config_list, chunk, chunk_prompt) for chunk in chunks]
    partials = await asyncio.gather(*chunk_tasks)

    merge_text = "\n\n".join(partials)
    summary = await _ask_llm_async(config_list, merge_text, merger_prompt)

    if was_truncated:
        summary += "\n\n[Note: Summary based on first portion of content - original page was truncated due to length]"

    return summary


# ──────────────────────────── crawl + summarize ──────────────────────────────


async def _crawl_and_summarise(
    url: str,
    config_list: list[dict[str, Any]],
    enc: Any,
    *,
    summarise: bool = True,
    chunk_prompt: str = "Summarise the chunk, preserving all facts.",
    merger_prompt: str = "Merge the partial summaries into one coherent overview.",
    full_prompt: str = "Provide a concise but complete summary.",
) -> dict[str, Any]:
    """Fetch url with Crawl4AI and optionally return an LLM summary."""
    start_total = time.time()

    # Stage 1: Crawl
    start_crawl = time.time()
    try:
        async with AsyncWebCrawler() as crawler:
            html = (await asyncio.wait_for(crawler.arun(url), timeout=CRAWL_TIMEOUT)).html
        crawl_time = time.time() - start_crawl
    except asyncio.TimeoutError:
        crawl_time = time.time() - start_crawl
        return {
            "url": url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "page_content": None,
            "timing": {"crawl": crawl_time, "clean": 0, "summarize": 0, "total": crawl_time},
        }

    # Stage 2: Clean HTML
    start_clean = time.time()
    raw_text = _clean_html(html, url)
    clean_time = time.time() - start_clean

    # Stage 3: Summarize
    summary: str | None = None
    summarize_time = 0.0
    if summarise:
        start_summarize = time.time()
        try:
            summary = await asyncio.wait_for(
                _summarise_text_async(
                    config_list,
                    enc,
                    raw_text,
                    chunk_prompt=chunk_prompt,
                    merger_prompt=merger_prompt,
                    full_prompt=full_prompt,
                ),
                timeout=SUMMARIZE_TIMEOUT,
            )
            summarize_time = time.time() - start_summarize
        except asyncio.TimeoutError:
            summarize_time = time.time() - start_summarize
            summary = raw_text[:500] + "..." if len(raw_text) > 500 else raw_text
            summary += "\n\n[Note: Full summarization timed out - showing snippet only]"

    total_time = time.time() - start_total

    return {
        "url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "page_content": summary if summarise else raw_text,
        "timing": {
            "crawl": round(crawl_time, 2),
            "clean": round(clean_time, 2),
            "summarize": round(summarize_time, 2),
            "total": round(total_time, 2),
        },
    }


# ──────────────────────────── Tavily search ──────────────────────────────────


def _tavily_search(query: str, tavily_api_key: str, num_results: int = 3) -> list[dict[str, str]]:
    """Call Tavily Search API and return a list of result dicts."""
    tavily_client = TavilyClient(api_key=tavily_api_key)
    data = tavily_client.search(
        query=query,
        search_depth="basic",
        max_results=min(num_results, 10),
        include_answer=False,
        include_raw_content=False,
    )

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
        }
        for item in data.get("results", [])
    ]


# ──────────────────────────── single query research ──────────────────────────


async def _research_single_query(
    query: str,
    tavily_api_key: str,
    config_list: list[dict[str, Any]],
    enc: Any,
    num_results: int = 3,
    chunk_prompt: str = "Summarise the chunk, preserving all facts.",
    merger_prompt: str = "Merge the partial summaries into one coherent overview.",
    full_prompt: str = "Provide a concise but complete summary.",
) -> dict[str, Any]:
    """Research a single query: search + crawl + summarize."""
    search_results = _tavily_search(query, tavily_api_key, num_results)

    crawl_tasks = [
        _crawl_and_summarise(
            result["url"],
            config_list,
            enc,
            summarise=True,
            chunk_prompt=chunk_prompt,
            merger_prompt=merger_prompt,
            full_prompt=full_prompt,
        )
        for result in search_results
    ]

    crawled = await asyncio.gather(*crawl_tasks, return_exceptions=True)

    sources = []
    for search_result, page in zip(search_results, crawled):
        if isinstance(page, BaseException):
            continue
        content = page.get("page_content")
        if not content:
            continue
        sources.append({
            "title": search_result["title"],
            "url": search_result["url"],
            "summary": content,
        })

    return {
        "query": query,
        "sources": sources,
    }


# ──────────────────────────── Tool class ─────────────────────────────────────


@require_optional_import(["tavily", "crawl4ai", "tiktoken", "tldextract", "bs4"], "quick-research")
@export_module("autogen.tools.experimental")
class QuickResearchTool(Tool):
    """Performs parallel web research across multiple queries.

    For each query, the tool:
    1. Searches the web using Tavily to get top results
    2. Crawls each result URL using crawl4ai
    3. Summarizes page content via LLM (any provider supported by AG2)

    This is a lightweight alternative to DeepResearchTool for quick fact-finding
    across multiple topics in parallel.

    Attributes:
        tavily_api_key: The Tavily API key for web search.
        llm_config: LLM configuration for summarization.
    """

    def __init__(
        self,
        *,
        llm_config: LLMConfig,
        tavily_api_key: str | None = None,
        num_results_per_query: int = 3,
    ) -> None:
        """Initialize the QuickResearchTool.

        Args:
            llm_config: LLM configuration for summarization. Supports any AG2-compatible provider.
            tavily_api_key: Tavily API key. Falls back to TAVILY_API_KEY env var.
            num_results_per_query: Number of search results to crawl per query. Defaults to 3.

        Raises:
            ValueError: If tavily_api_key is not provided or set in env vars.
        """
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.llm_config = llm_config
        self.num_results_per_query = num_results_per_query

        # Extract config_list as list of dicts for OpenAIWrapper
        self._config_list = [c.model_dump() for c in llm_config.config_list]

        if self.tavily_api_key is None:
            raise ValueError("tavily_api_key must be provided either as an argument or via TAVILY_API_KEY env var")

        async def quick_research(
            queries: Annotated[list[str], "List of search queries to research (max 5)."],
            tavily_api_key: Annotated[str, Depends(on(self.tavily_api_key))],
            config_list: Annotated[list[dict[str, Any]], Depends(on(self._config_list))],
            num_results_per_query: Annotated[int, Depends(on(self.num_results_per_query))],
            chunk_prompt: Annotated[
                str, "Prompt for summarizing individual text chunks."
            ] = "Summarise the chunk, preserving all facts.",
            merger_prompt: Annotated[
                str, "Prompt for merging chunk summaries."
            ] = "Merge the partial summaries into one coherent overview.",
            full_prompt: Annotated[
                str, "Prompt for summarizing short texts in one pass."
            ] = "Provide a concise but complete summary.",
        ) -> str:
            """Research multiple queries in parallel and return summarized sources.

            For each query:
            1. Performs Tavily search to get top N results
            2. Crawls each result URL using crawl4ai
            3. Summarizes page content via LLM

            Args:
                queries: List of search queries (max 5).
                tavily_api_key: Tavily API key (injected dependency).
                config_list: LLM config list (injected dependency).
                num_results_per_query: Number of results per query (injected dependency).
                chunk_prompt: Prompt for summarizing individual text chunks.
                merger_prompt: Prompt for merging chunk summaries.
                full_prompt: Prompt for summarizing short texts in one pass.

            Returns:
                JSON string: list of objects with query and sources (title, url, summary).
            """
            if not queries:
                return "[]"

            queries = queries[:MAX_QUERIES]

            enc = tiktoken.get_encoding("cl100k_base")

            research_tasks = [
                _research_single_query(
                    query,
                    tavily_api_key=tavily_api_key,
                    config_list=config_list,
                    enc=enc,
                    num_results=num_results_per_query,
                    chunk_prompt=chunk_prompt,
                    merger_prompt=merger_prompt,
                    full_prompt=full_prompt,
                )
                for query in queries
            ]

            results = await asyncio.gather(*research_tasks, return_exceptions=True)

            output = []
            for result in results:
                if isinstance(result, Exception):
                    continue
                output.append(result)

            return json.dumps(output)

        super().__init__(
            name="quick_research",
            description="Research multiple queries in parallel using web search, crawling, and LLM summarization.",
            func_or_tool=quick_research,
        )
