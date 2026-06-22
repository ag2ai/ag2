---
title: WebSurferAgent
sidebarTitle: WebSurferAgent
---

### Introduction to WebSurferAgent

If you need an agent that can browse, extract, or interact with the web, `WebSurferAgent` is a good choice. The agent actions the request(s) given to it by determining what to do on the web and browsing and crawling it, returning the details of what it finds.

The `WebSurferAgent` has two in-built web tools to choose from:

1. [browser-use](https://github.com/browser-use/browser-use) - uses an actual browser instance (visible or headless), interacting with the web pages in realtime
2. [Crawl4AI](https://github.com/unclecode/crawl4ai) - crawls without a visual browser instance

!!! tip
    If you want to add browsing capabilities to your existing agents, see [this notebook for browser-use](../../use-cases/notebooks/notebooks/agentchat_tools_browser_use.md) and [this notebook for Crawl4AI](../../use-cases/notebooks/notebooks/agentchat_tools_crawl4ai.md).

!!! warning
    [`Browser Use`](https://github.com/browser-use/browser-use) requires **Python 3.11 or higher**.

### Installation and Setup

To get started with `WebSurferAgent`, install AG2 with the `browser-use` and/or `crawl4ai` extras.
```bash
pip install ag2[openai,browser-use]
```
and/or
```bash
pip install ag2[openai,crawl4ai]
```

!!! tip
    If you have been using `autogen` or `ag2`, all you need to do is upgrade it using:
    ```bash
    pip install -U autogen[openai,browser-use]
    ```
    or
    ```bash
    pip install -U ag2[openai,browser-use]
    ```
    as `autogen` and `ag2` are aliases for the same PyPI package.

And then setup Playwright:
```bash
# Installs Playwright and browsers for all OS
playwright install
# Additional command, mandatory for Linux only
playwright install-deps
```

Now, you can create an agent, nominating the web tool:
!!! tip
    [`Browser Use`](https://github.com/browser-use/browser-use) supports the following models: [Supported Models](https://docs.browser-use.com/customize/supported-models#supported-models)

    We had great experience with `OpenAI`, `Anthropic`, and `Gemini`. However, `DeepSeek` and `Ollama` haven't performed as well.
!!! tip
    [`Crawl4AI`](https://github.com/unclecode/crawl4ai) is built on top of [LiteLLM](https://github.com/BerriAI/litellm) and supports the same models as LiteLLM.

    We had great experience with `OpenAI`, `Anthropic`, `Gemini` and `Ollama`. However, as of this writing, `DeepSeek` is encountering some issues.

### Using WebSurferAgent

=== "browser_use"
    ```python
        from autogen import LLMConfig
        from autogen.agents.experimental import WebSurferAgent
        from dotenv import load_dotenv
        import os
        load_dotenv()

        # Put your key in the OPENAI_API_KEY environment variable
        llm_config = LLMConfig({"api_type": "openai", "model": "gpt-5-nano","api_key":os.getenv("OPENAI_API_KEY")})

        # Create our agent
        websurfer = WebSurferAgent(
            name="WebSurfer",
            web_tool="browser_use",
            llm_config=llm_config,
        )
        ```

=== "crawl4ai"
    ```python
        from autogen import LLMConfig
        from autogen.agents.experimental import WebSurferAgent
        from dotenv import load_dotenv
        import os
        load_dotenv()

        # Put your key in the OPENAI_API_KEY environment variable
        llm_config = LLMConfig({"api_type": "openai", "model": "gpt-5-nano","api_key":os.getenv("OPENAI_API_KEY")})

        # Create our agent
        websurfer = WebSurferAgent(
            name="WebSurfer",
            web_tool="crawl4ai",
            llm_config=llm_config,
        )
        ```

!!! tip
    Crawl4AI doesn't always require an LLM configuration, see [this notebook](../../use-cases/notebooks/notebooks/agentchat_tools_crawl4ai.md) for examples with and without one.

Let's browse the web for news on AG2.

```python
# 1. Import our WebSurferAgent
from autogen import LLMConfig
from autogen.agents.experimental import WebSurferAgent
from dotenv import load_dotenv
import os
load_dotenv()

# Put your key in the OPENAI_API_KEY environment variable
llm_config = LLMConfig(config_list=[{"api_type": "openai", "model": "gpt-5-nano", "api_key": os.getenv("OPENAI_API_KEY")}])

# 2. Add additional browser configurations for our browser-use tool
browser_use_browser_config = {"browser_config": {"headless": False}, "agent_kwargs": {"generate_gif": True}}

# 3. Create the agent, nominating the tool and tool config
web_researcher = WebSurferAgent(
    name="researcher",
    web_tool="browser_use",
    web_tool_kwargs=browser_use_browser_config,
    llm_config=llm_config,
)

# 4. Run our agent, passing in the tools that our WebSurferAgent has so they can be executed
ag2_news_result = web_researcher.run(
    "Search for the latest news on AG2 AI",
    tools=web_researcher.tools,
)

print(ag2_news_result.summary)
```

<Example/>

Let's break it down:

1. Import `WebSurferAgent` and create an LLM configuration for the browser-use tool to use.

2. We create a configuration dictionary turning off the headless mode (so we can see what's happening) and saving an animated GIF of the process (shown below).

3. Create the agent, nominating the web tool and passing in the LLM and tool configurations.

4. Run the agent, ensuring we pass the agent's tools through to the `run` method so it can add them to the internal executor agent to execute.

![WebSurferAgent in action](./assets/websurferagent_animated.gif)
