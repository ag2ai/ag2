---
description: Correct import paths for AG2 modules
globs: "**/*.py"
alwaysApply: true
---

# AG2 Import Guide

The package is installed as `ag2` but imported as `ag2`:

```bash
pip install ag2
```

```python
import ag2  # Correct
# import ag2  # Wrong — this is the package name, not the module
```

## Core Imports

```python
from ag2 import (
    ConversableAgent,
    UserProxyAgent,
    LLMConfig,
    ChatResult,
    UpdateSystemMessage,
    register_function,
)
from ag2.agentchat import run_group_chat, a_run_group_chat
```

## Tools

```python
from ag2.tools import Tool, Toolkit, tool, ChatContext, Depends
```

## Group Chat Patterns (Modern API)

```python
from ag2.agentchat.group import (
    OnCondition,
    OnContextCondition,
    ContextVariables,
)
from ag2.agentchat.group.patterns import (
    AutoPattern,
    RoundRobinPattern,
    ManualPattern,
    RandomPattern,
    DefaultPattern,
)
from ag2.agentchat.group.guardrails import LLMGuardrail, RegexGuardrail
```

## Transition Targets

```python
from ag2.agentchat.group import (
    AgentTarget,
    AgentNameTarget,
    TerminateTarget,
    StayTarget,
    RevertToUserTarget,
    NestedChatTarget,
    GroupChatTarget,
    FunctionTarget,
)
```

## Context Conditions

```python
from ag2.agentchat.group import (
    StringContextCondition,
    ExpressionContextCondition,
    StringAvailableCondition,
    ExpressionAvailableCondition,
    StringLLMCondition,
    ContextStrLLMCondition,
    ContextStr,
    ContextExpression,
    Handoffs,
    ReplyResult,
    AskUserTarget,
)
```

## MCP

```python
from ag2.mcp import create_toolkit
from ag2.mcp.mcp_client import StdioConfig, SseConfig, MCPConfig
```

Requires: `pip install ag2[mcp]`

## Code Executors

```python
from ag2.coding import LocalCommandLineCodeExecutor, DockerCommandLineCodeExecutor
```

## Experimental Tools

Pre-built tools for common tasks — install the relevant extra and import from `ag2.tools.experimental`:

```python
# Web search — pip install ag2[duckduckgo_search]
from ag2.tools.experimental import DuckDuckGoSearchTool

# Tavily search — pip install ag2[tavily]
from ag2.tools.experimental import TavilySearchTool

# Parallel web research — pip install ag2[quick-research]
from ag2.tools.experimental import QuickResearchTool

# Browser automation — pip install ag2[browser-use]
from ag2.tools.experimental import BrowserUseTool

# Web crawling — pip install ag2[crawl4ai]
from ag2.tools.experimental import Crawl4AITool

# Deep research
from ag2.tools.experimental import DeepResearchTool

# Wikipedia — pip install ag2[wikipedia]
from ag2.tools.experimental import WikipediaQueryRunTool, WikipediaPageLoadTool

# Messaging — pip install ag2[slack] / ag2[discord] / ag2[telegram]
from ag2.tools.experimental import SlackSendTool, DiscordSendTool, TelegramSendTool

# Google search — pip install ag2[google-search]
from ag2.tools.experimental import GoogleSearchTool

# YouTube search — pip install ag2[google-search]
from ag2.tools.experimental import YoutubeSearchTool

# Perplexity search — pip install ag2[perplexity]
from ag2.tools.experimental import PerplexitySearchTool

# Reliable tool wrapper
from ag2.tools.experimental import ReliableTool

# Apply patch tool
from ag2.tools.experimental import ApplyPatchTool

# Messaging — retrieve tools
from ag2.tools.experimental import SlackRetrieveTool, DiscordRetrieveTool, TelegramRetrieveTool
```

## Experimental Agents

```python
from ag2.agents.experimental import (
    ReasoningAgent,       # Tree-of-thought reasoning
    DeepResearchAgent,    # Multi-step deep research
    WebSurferAgent,       # Web browsing agent
    DocAgent,             # Document processing
    WikipediaAgent,       # Wikipedia querying
    DiscordAgent,         # Discord bot
    SlackAgent,           # Slack bot
    TelegramAgent,        # Telegram bot
)
```

## Contrib Modules

Contrib modules require extra dependencies:

```python
# RAG — pip install ag2[rag]
from ag2.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

# Interop — pip install ag2[interop]
from ag2.interop import Interoperability
from ag2.interop.langchain import LangChainInteroperability
from ag2.interop.pydantic_ai import PydanticAIInteroperability

# Caching
from ag2.cache import Cache
```

## Events

```python
from ag2.events import BaseEvent, wrap_event
from ag2.events.agent_events import TextEvent, ToolCallEvent
```

## Optional Extras

Install optional features with extras:

```bash
pip install ag2[openai]            # OpenAI support
pip install ag2[anthropic]         # Anthropic support
pip install ag2[gemini]            # Google Gemini
pip install ag2[mcp]               # Model Context Protocol
pip install ag2[rag]               # RAG capabilities
pip install ag2[interop]           # Framework interop (LangChain, PydanticAI)
pip install ag2[tracing]           # OpenTelemetry tracing
pip install ag2[duckduckgo_search] # DuckDuckGo search (no API key)
pip install ag2[tavily]            # Tavily search
pip install ag2[quick-research]    # Parallel web research
pip install ag2[browser-use]       # Browser automation
pip install ag2[crawl4ai]          # Web crawling
pip install ag2[wikipedia]         # Wikipedia tools
pip install ag2[google-search]     # Google Search
pip install ag2[perplexity]        # Perplexity search
pip install ag2[searxng]           # SearxNG search
pip install ag2[firecrawl]         # Firecrawl web scraping
pip install ag2[a2a]               # Agent-to-agent protocol
```
