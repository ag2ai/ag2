---
title: TinyFish Search
sidebarTitle: TinyFish Search
---

`TinyFishSearchToolkit` gives an agent two related tools powered by [TinyFish](https://www.tinyfish.ai/){.external-link target="_blank"}: web search and browser-rendered page fetch. Use it when an agent needs current search results and then needs to read the full content of selected URLs.

!!! note
    Requires the `tinyfish` package and an API key: `pip install "tinyfish>=0.2.3"`

```python linenums="1"
import os
from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.extensions.tools.search import TinyFishSearchToolkit

agent = Agent(
    "researcher",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
    tools=[TinyFishSearchToolkit(api_key=os.environ["TINYFISH_API_KEY"])],
)
```

If `api_key` is omitted, the TinyFish SDK reads the `TINYFISH_API_KEY` environment variable automatically.

## Tools

| Tool | Description |
| :--- | :--- |
| `tinyfish_search` | Search the web and return ranked results with position, site name, title, snippet, and URL |
| `tinyfish_fetch` | Fetch and extract clean content from up to 10 URLs, with per-URL errors returned separately |

## Shared defaults

Constructor defaults are applied to the toolkit's default tools:

```python linenums="1"
toolkit = TinyFishSearchToolkit(
    api_key=...,
    location="US",       # default location for tinyfish_search
    language="en",       # default language for tinyfish_search
    format="markdown",   # default output format for tinyfish_fetch
    links=True,          # include page links in fetch results
    image_links=False,   # include image links in fetch results
)
```

## Picking a subset of tools

Each tool is exposed as a factory method on the toolkit (`toolkit.search()`, `toolkit.fetch()`). Call the method to get a ready-to-use tool, then pass only the ones you need to the agent:

```python linenums="1"
toolkit = TinyFishSearchToolkit(api_key=...)

agent = Agent(
    "researcher",
    config=config,
    tools=[toolkit.search(location="US", language="en")],
)
```

## Search configuration

`tinyfish_search` accepts a required `query` at execution time. Configure the optional location and language defaults on `toolkit.search()`:

```python linenums="1"
toolkit = TinyFishSearchToolkit(api_key=...)

search_tool = toolkit.search(
    location="US",
    language="en",
)

agent = Agent("researcher", config=config, tools=[search_tool])
```

## Fetch configuration

`tinyfish_fetch` accepts a list of URLs at execution time. Configure the output format and extracted link fields on `toolkit.fetch()`:

```python linenums="1"
toolkit = TinyFishSearchToolkit(api_key=...)

fetch_tool = toolkit.fetch(
    format="markdown",  # "markdown" | "html" | "json"
    links=True,
    image_links=True,
)

agent = Agent("researcher", config=config, tools=[fetch_tool])
```

All configurable defaults on `TinyFishSearchToolkit`, `toolkit.search()`, and `toolkit.fetch()` accept a `Variable` for deferred context resolution.
`tinyfish_fetch` accepts only `http` and `https` URLs.

## Result

`tinyfish_search` returns a `TinyFishSearchResponse`:

| Field | Description |
| :--- | :--- |
| `query` | The search query TinyFish executed |
| `results` | List of `TinyFishSearchResult` (`position`, `site_name`, `title`, `snippet`, `url`) |
| `total_results` | Number of results returned |

`tinyfish_fetch` returns a `TinyFishFetchResponse`:

| Field | Description |
| :--- | :--- |
| `results` | List of successfully fetched pages with metadata, extracted `text`, links, and image links |
| `errors` | List of per-URL fetch failures (`url`, `error`) |

!!! tip
    Use `tinyfish_search` first to discover candidate pages, then `tinyfish_fetch` to read the pages that look relevant. TinyFish Search and Fetch are separate from the goal-directed TinyFish Agent API.
