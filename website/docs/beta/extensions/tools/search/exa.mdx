---
title: Exa Search
sidebarTitle: Exa Search
---

`ExaToolkit` gives an agent four related tools powered by the [Exa](https://exa.ai){.external-link target="_blank"} neural search engine: web search, find-similar, content retrieval, and AI-powered answers — all sharing a single client.

!!! note
    Requires the `exa-py` package and an API key: `pip install "exa-py>=2.12.1,<3"`

```python linenums="1"
import os
from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.extensions.tools.search import ExaToolkit

agent = Agent(
    "researcher",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
    tools=[ExaToolkit(api_key=os.environ["EXA_API_KEY"])],
)
```

If `api_key` is omitted, the Exa SDK reads `EXA_API_KEY` from the environment automatically.

## Tools

| Tool | Description |
| :--- | :--- |
| `exa_search` | Neural web search with filters (domains, dates, type, category) |
| `exa_find_similar` | Find pages similar to a given URL |
| `exa_get_contents` | Fetch full text content for specific URLs |
| `exa_answer` | Get an AI-generated answer with citations |

## Shared defaults

`num_results` and `max_characters` on the constructor are applied to the default `exa_search` and `exa_find_similar` tools:

```python linenums="1"
toolkit = ExaToolkit(
    api_key=...,
    num_results=10,         # applies to search & find_similar
    max_characters=2000,    # per-result text cap for search; None = metadata-only
)
```

## Picking a subset of tools

Each tool is exposed as a factory method on the toolkit (`toolkit.search()`, `toolkit.find_similar()`, `toolkit.get_contents()`, `toolkit.answer()`). Call the method to get a ready-to-use tool, then pass only the ones you need to the agent:

```python linenums="1"
toolkit = ExaToolkit(api_key=...)

agent = Agent(
    "researcher",
    config=config,
    tools=[toolkit.search(), toolkit.answer()],
)
```

## Per-tool configuration

Per-call parameters (filters, domains, dates, `num_results`, `max_characters`, etc.) live on the factory methods, not on the toolkit itself:

```python linenums="1"
toolkit = ExaToolkit(api_key=...)

search_tool = toolkit.search(
    num_results=5,
    max_characters=2000,           # triggers search_and_contents for full text
    search_type="neural",          # "neural" | "keyword" | "hybrid" | "auto" | "fast" | "deep"
    category="research paper",     # e.g. "news", "github", "pdf", ...
    include_domains=["arxiv.org"],
    exclude_domains=["medium.com"],
    start_published_date="2024-01-01",
    end_published_date="2024-12-31",
    use_autoprompt=True,
    livecrawl="always",            # "never" | "fallback" | "always" | "preferred"
)

agent = Agent("researcher", config=config, tools=[search_tool, toolkit.answer()])
```

When `max_characters` is set, `exa_search` calls Exa's `search_and_contents` endpoint so each result carries `text`. When `max_characters` is `None`, only metadata is returned (cheaper and faster).

All runtime parameters accept `Variable` for deferred context resolution.
