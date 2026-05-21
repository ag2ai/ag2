---
title: Firecrawl Tool
description: Web scraping and crawling with Firecrawl
---

# Firecrawl Tool

The `FirecrawlTool` provides powerful web scraping and crawling capabilities using the [Firecrawl API](https://firecrawl.dev/). It can scrape single pages, crawl entire websites, and map website structures to discover URLs.

## Features

- **Single Page Scraping**: Extract clean content from individual web pages
- **Website Crawling**: Recursively crawl multiple pages from a website
- **URL Mapping**: Discover URLs from a website without scraping content
- **Multiple Formats**: Get content in markdown, HTML, or other formats
- **Advanced Filtering**: Include/exclude specific HTML tags and paths
- **JavaScript Support**: Handles JavaScript-rendered pages
- **Rate Limiting**: Built-in timeout and rate limiting controls

## Installation

Install the required dependency:

```bash
pip install firecrawl-py
```

## Setup

1. Get an API key from [firecrawl.dev](https://firecrawl.dev/)
2. Set it as an environment variable or pass it directly to the tool:

```python
import os
from autogen.tools.experimental.firecrawl import FirecrawlTool

# Option 1: Environment variable
os.environ["FIRECRAWL_API_KEY"] = "fc-your-api-key-here"
firecrawl_tool = FirecrawlTool()

# Option 2: Direct parameter
firecrawl_tool = FirecrawlTool(firecrawl_api_key="fc-your-api-key-here")
```

## Usage

### Basic Scraping

The default functionality scrapes a single URL:

```python
# Scrape a single page
results = firecrawl_tool(url="https://example.com")

print(f"Title: {results[0]['title']}")
print(f"Content: {results[0]['content']}")
print(f"Metadata: {results[0]['metadata']}")
```

### Advanced Scraping Options

Customize the scraping process:

```python
results = firecrawl_tool(
    url="https://example.com",
    formats=["markdown", "html"],  # Output formats
    include_tags=["h1", "h2", "p"],  # Only these HTML tags
    exclude_tags=["script", "style"],  # Exclude these tags
    headers={"User-Agent": "Custom Agent"},  # Custom headers
    wait_for=2000,  # Wait 2 seconds for page load
    timeout=10000   # 10 second timeout
)
```

### Website Crawling

Crawl multiple pages from a website:

```python
crawl_results = firecrawl_tool.crawl(
    url="https://example.com",
    limit=5,  # Maximum pages to crawl
    formats=["markdown"],
    max_depth=2,  # Maximum crawl depth
    include_paths=["/docs/*"],  # Only crawl documentation
    exclude_paths=["/admin/*"],  # Exclude admin pages
    allow_backward_crawling=False,
    allow_external_content_links=False
)

for page in crawl_results:
    print(f"Title: {page['title']}")
    print(f"URL: {page['url']}")
    print(f"Content: {page['content'][:200]}...")
```

### URL Mapping

Discover URLs from a website:

```python
map_results = firecrawl_tool.map(
    url="https://example.com",
    search="docs",  # Filter URLs containing "docs"
    include_subdomains=False,
    ignore_sitemap=False,
    limit=100  # Maximum URLs to return
)

for url_info in map_results:
    print(f"Found URL: {url_info['url']}")
```

## AG2 Agent Integration

Use the FirecrawlTool with AG2 agents:

```python
from autogen import AssistantAgent

# Create assistant with Firecrawl capabilities
assistant = AssistantAgent(
    name="web_scraper",
    system_message="You are a web scraping assistant. Use Firecrawl to extract content from websites.",
    llm_config={"model": "gpt-5-nano"},
)

# Register the tool
firecrawl_tool.register_for_llm(assistant)

# The assistant can now use Firecrawl in conversations
response = assistant.run(
    message="Please scrape the content from https://example.com and summarize it",
    tools=assistant.tools
)
```

## Tool Methods

The `FirecrawlTool` provides three main methods:

### `scrape()` (Default)
Scrapes a single URL and returns structured content.

**Parameters:**
- `url` (str): The URL to scrape
- `formats` (list[str], optional): Output formats (default: ["markdown"])
- `include_tags` (list[str], optional): HTML tags to include
- `exclude_tags` (list[str], optional): HTML tags to exclude
- `headers` (dict[str, str], optional): HTTP headers
- `wait_for` (int, optional): Page load wait time in milliseconds
- `timeout` (int, optional): Request timeout in milliseconds

### `crawl()`
Recursively crawls a website starting from a URL.

**Parameters:**
- `url` (str): Starting URL to crawl
- `limit` (int): Maximum pages to crawl (default: 5)
- `formats` (list[str], optional): Output formats
- `include_paths` (list[str], optional): URL patterns to include
- `exclude_paths` (list[str], optional): URL patterns to exclude
- `max_depth` (int, optional): Maximum crawl depth
- `allow_backward_crawling` (bool): Allow crawling backward links
- `allow_external_content_links` (bool): Allow external links

### `map()`
Discovers URLs from a website without scraping content.

**Parameters:**
- `url` (str): Website URL to map
- `search` (str, optional): Filter URLs by search term
- `ignore_sitemap` (bool): Whether to ignore sitemap
- `include_subdomains` (bool): Include subdomain URLs
- `limit` (int): Maximum URLs to return (default: 5000)

## Response Format

All methods return a list of dictionaries with the following structure:

```python
{
    "title": "Page Title",
    "url": "https://example.com/page",
    "content": "Page content in requested format",
    "metadata": {
        "title": "Page Title",
        "sourceURL": "https://example.com/page",
        # Additional metadata from Firecrawl
    }
}
```

## Use Cases

- **Documentation Extraction**: Scrape and process documentation sites
- **Content Research**: Gather information from multiple web sources
- **Website Analysis**: Map and analyze website structures
- **Data Collection**: Automated content extraction for analysis
- **Knowledge Base Creation**: Build knowledge bases from web content

## Error Handling

The tool includes proper error handling and logging:

```python
import logging

# Enable logging to see detailed error information
logging.basicConfig(level=logging.INFO)

# Failed operations return empty lists
results = firecrawl_tool(url="https://invalid-url.com")
if not results:
    print("Scraping failed - check logs for details")
```

## Example Notebook

For a complete working example, see the [Firecrawl Tool notebook](https://github.com/ag2ai/ag2/blob/main/notebook/tools_firecrawl.ipynb).
