---
title: TinyFish Tool
description: "Integrate TinyFish with AG2 agents for goal-directed web scraping, web search, and page content extraction. Learn how to set up TINYFISH_API_KEY and use TinyFish tools in your multi-agent workflows."
---

The [TinyFish](https://tinyfish.ai/) integration allows AG2 agents to use TinyFish's Agent, Search, and Fetch APIs. Use the Agent API when TinyFish should decide the browser actions from a natural-language goal, Search when the agent needs ranked web results, and Fetch when the agent already has URLs and needs clean extracted page content.

## Configuring Your TinyFish API Key

1. **Create a TinyFish Account**:
   - Visit [TinyFish](https://tinyfish.ai/)
   - Click **Sign Up** and create an account

2. **Get Your API Key**:
   - Navigate to the TinyFish dashboard
   - Generate an API key under **API Keys**

3. **Set the `TINYFISH_API_KEY` Environment Variable**:
   ```bash
   export TINYFISH_API_KEY="your_api_key_here"
   ```

## Package Installation

Install AG2 (with the `openai` extra for the example below) and the `tinyfish` package:

```bash
pip install -U "ag2[openai]" "tinyfish>=0.2.3"
```

> **Note:** `autogen` and `ag2` are aliases for the same PyPI package:
> ```bash
> pip install -U "autogen[openai]" "tinyfish>=0.2.3"
> ```

## Implementation

### Available Tools

| Tool | TinyFish API | Use it when |
|---|---|---|
| `TinyFishTool` | Agent | TinyFish should automate a website from a natural-language goal |
| `TinyFishSearchTool` | Search | The agent needs ranked search results with titles, snippets, and URLs |
| `TinyFishFetchTool` | Fetch | The agent already has URLs and needs extracted page content |

### Imports

```python
import asyncio
import os
from autogen import ConversableAgent, LLMConfig
from autogen.tools.experimental import TinyFishFetchTool, TinyFishSearchTool, TinyFishTool
```

### Agent Configuration

```python
llm_config = LLMConfig({"api_type": "openai", "model": "gpt-4o"})

assistant = ConversableAgent(
    name="assistant",
    system_message="You are a helpful assistant that can scrape web pages using the TinyFish tool. Use the tool to extract the requested information.",
    llm_config=llm_config,
)

user_proxy = ConversableAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    llm_config=False,
)
```

### Agent Tool Setup

```python
tinyfish_tool = TinyFishTool(tinyfish_api_key=os.getenv("TINYFISH_API_KEY"))

# Register the tool for LLM recommendation and execution.
tinyfish_tool.register_for_llm(assistant)
tinyfish_tool.register_for_execution(user_proxy)
```

### Agent Usage Example

```python
async def main():
    response = await user_proxy.a_run(
        assistant,
        message="Scrape https://example.com and extract the main product offerings and pricing information.",
        max_turns=2,
        summary_method="last_msg",
    )
    await response.process()
    print(f"Final Answer: {await response.summary}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Search and Fetch Tool Setup

Use `TinyFishSearchTool` to discover relevant pages and `TinyFishFetchTool` to extract clean content from known URLs:

```python
search_tool = TinyFishSearchTool(tinyfish_api_key=os.getenv("TINYFISH_API_KEY"))
fetch_tool = TinyFishFetchTool(tinyfish_api_key=os.getenv("TINYFISH_API_KEY"))

search_tool.register_for_llm(assistant)
search_tool.register_for_execution(user_proxy)

fetch_tool.register_for_llm(assistant)
fetch_tool.register_for_execution(user_proxy)
```

### Search and Fetch Usage Example

```python
async def main():
    response = await user_proxy.a_run(
        assistant,
        message=(
            "Search for AG2 multi-agent framework, pick the most relevant result, "
            "then fetch the page content and summarize it."
        ),
        max_turns=4,
        summary_method="last_msg",
    )
    await response.process()
    print(f"Final Answer: {await response.summary}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Parameters

`TinyFishTool` accepts the following parameters at call time:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | *required* | The URL to scrape |
| `goal` | `str` | *required* | A natural language description of what information to extract from the page |

`TinyFishSearchTool` accepts the following parameters at call time:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | *required* | The search query string |
| `location` | `str \| None` | `None` | Optional country or location for geo-targeted results |
| `language` | `str \| None` | `None` | Optional language code for result language |

`TinyFishFetchTool` accepts the following parameters at call time:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `urls` | `list[str]` | *required* | URLs to fetch and extract. TinyFish supports 1-10 URLs per request |
| `format` | `str \| None` | `None` | Output format: `markdown`, `html`, or `json` |
| `links` | `bool \| None` | `None` | Whether to include page links in results |
| `image_links` | `bool \| None` | `None` | Whether to include image links in results |

`TinyFishFetchTool` accepts only `http` and `https` URLs.

## Output

Each Agent scrape returns a dictionary with:

- `url` — the scraped URL
- `goal` — the extraction goal that was used
- `data` — the structured data extracted by TinyFish

Each Search call returns a dictionary with:

- `query` — the search query TinyFish executed
- `total_results` — the number of returned results
- `results` — ranked results containing `position`, `site_name`, `title`, `snippet`, and `url`

Each Fetch call returns a dictionary with:

- `results` — successfully fetched pages with metadata, extracted `text`, links, and image links
- `errors` — per-URL failures containing `url` and `error`

## Error Handling

The tool handles errors gracefully and returns them in the response:

```python
# Failed operations return a dict with an error field
result = tinyfish_tool(
    url="https://invalid-url.com",
    goal="Extract company info"
)
if "error" in result:
    print(f"Scraping failed: {result['error']}")
```

Search and Fetch tools also return error information instead of raising it to the agent:

```python
search_result = search_tool(query="AG2", tinyfish_api_key=os.getenv("TINYFISH_API_KEY"))
if "error" in search_result:
    print(f"Search failed: {search_result['error']}")

fetch_result = fetch_tool(urls=["https://invalid-url.com"], tinyfish_api_key=os.getenv("TINYFISH_API_KEY"))
for error in fetch_result["errors"]:
    print(f"Fetch failed for {error['url']}: {error['error']}")
```

## Use Cases

- **Due Diligence**: Extract company information, team details, and financials from corporate websites - [Code on Build with AG2](https://github.com/ag2ai/build-with-ag2/tree/main/due-diligence-with-tinyfish)
- **Competitive Analysis**: Gather product and pricing data from competitor sites
- **Lead Enrichment**: Scrape company profiles for sales intelligence
- **Content Research**: Extract specific data points from articles and reports
- **Market Research**: Collect structured data from industry publications
- **Web Research**: Search for candidate pages and fetch selected pages for summarization or extraction

## See Also

- [Beta TinyFish Search Extension](../../beta/extensions/tools/search/tinyfish.md)
- [Crawl4AI Tool](crawl4ai.md)
- [Firecrawl Tool](firecrawl.md)
- [Build with AG2 - Due Diligence Example](https://github.com/ag2ai/build-with-ag2/tree/main/due-diligence-with-tinyfish)
