---
title: Built-in Provider Tools
sidebarTitle: Built-in Tools
---

# Built-in Provider Tools

AG2 includes built-in tools that map to server-side capabilities offered by LLM providers. These tools are executed by the provider's API — not locally — and require no function implementation on your side.

| Tool | Anthropic | OpenAI | Gemini |
| :--- | :---: | :---: | :---: |
| `CodeExecutionTool` | ✓ | ✓ | ✓ |
| `WebSearchTool` | ✓ | ✓ | ✓ |
| `WebFetchTool` | ✓ | ✗ | ✓ |
| `ShellTool` | ✓ | ✓ | ✗ |
| `MCPServerTool` | ✓ | ✓ | ✗ |
| `ImageGenerationTool` | ✗ | ✓ | ✗ |
| `MemoryTool` | ✓ | ✗ | ✗ |

## Web Search

Gives the model access to real-time web search results.

```python linenums="1"
from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.tools import WebSearchTool, UserLocation

agent = Agent(
    "researcher",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
    tools=[
        WebSearchTool(
            max_uses=5,
            user_location=UserLocation(country="US"),
            allowed_domains=["github.com", "pypi.org"],
            blocked_domains=["pinterest.com"],
        ),
    ],
)
```

Not all parameters are supported by every provider. Unsupported parameters are silently ignored.

| Parameter | Anthropic | OpenAI | Gemini |
| :--- | :---: | :---: | :---: |
| `max_uses` | ✓ | ✓ | ✗ |
| `user_location` | ✓ | ✓ | ✗ |
| `search_context_size` | ✗ | ✓ | ✗ |
| `allowed_domains` | ✓ | ✓ | ✗ |
| `blocked_domains` | ✓ | ✗ | ✓ |

## Web Fetch

Fetches full content from specific URLs. Useful for reading documentation, articles, or PDFs.

```python linenums="1"
from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.tools import WebFetchTool

agent = Agent(
    "researcher",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
    tools=[
        WebFetchTool(
            max_uses=3,
            max_content_tokens=50000,
            citations=True,
        ),
    ],
)
```

| Parameter | Anthropic | Gemini |
| :--- | :---: | :---: |
| `max_uses` | ✓ | ✗ |
| `allowed_domains` | ✓ | ✗ |
| `blocked_domains` | ✓ | ✗ |
| `citations` | ✓ | ✗ |
| `max_content_tokens` | ✓ | ✗ |

!!! note
    OpenAI does not support web fetch. Using `WebFetchTool` with an OpenAI config will raise an error.

## Code Execution

Lets the model write and run code inline during a conversation.

```python linenums="1"
from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.tools import CodeExecutionTool

agent = Agent(
    "analyst",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
    tools=[CodeExecutionTool()],
)
```

The tool accepts a `version` parameter for provider version pinning:

```python
CodeExecutionTool(version="code_execution_20250825")
```

## Memory

Enables Claude to store and retrieve information across conversations. Claude can create, read, update, and delete files in a `/memories` directory.

```python linenums="1"
from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.tools import MemoryTool

agent = Agent(
    "assistant",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
    tools=[MemoryTool()],
)
```

!!! note
    `MemoryTool` is currently only supported by Anthropic.

## Shell

Gives the model the ability to run shell commands. The execution environment depends on the provider.

```python linenums="1"
from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.tools import ShellTool

agent = Agent(
    "devops",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
    tools=[ShellTool()],
)
```

OpenAI supports configuring the execution environment:

```python linenums="1"
from autogen.beta import Agent
from autogen.beta.config import OpenAIResponsesConfig
from autogen.beta.tools import ShellTool
from autogen.beta.tools.builtin.shell import ContainerAutoEnvironment, NetworkPolicy

agent = Agent(
    "devops",
    config=OpenAIResponsesConfig(model="gpt-4.1"),
    tools=[
        ShellTool(
            environment=ContainerAutoEnvironment(
                network_policy=NetworkPolicy(allowed_domains=["pypi.org"]),
            ),
        ),
    ],
)
```

| Environment | Description |
| :--- | :--- |
| `ContainerAutoEnvironment` | Provider-managed container with optional network policy |
| `ContainerReferenceEnvironment` | Reference an existing container by ID |

!!! warning
    `ShellTool` gives the model direct shell access. Use it only with trusted prompts and consider restricting the environment.

## MCP Server

Integrates external [MCP (Model Context Protocol)](https://modelcontextprotocol.io/){.external-link target="_blank"} servers, giving the model access to remote tools.

```python linenums="1"
from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.tools import MCPServerTool

agent = Agent(
    "assistant",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
    tools=[
        MCPServerTool(
            server_url="https://mcp.example.com/sse",
            server_label="my-tools",
            allowed_tools=["search", "summarize"],
        ),
    ],
)
```

| Parameter | Anthropic | OpenAI |
| :--- | :---: | :---: |
| `server_url` | ✓ | ✓ |
| `server_label` | ✓ | ✓ |
| `authorization_token` | ✓ | ✗ |
| `description` | ✓ | ✗ |
| `allowed_tools` | ✓ | ✓ |
| `blocked_tools` | ✓ | ✗ |
| `headers` | ✗ | ✓ |

## Image Generation

Instructs the model to generate images inline during a conversation. Generated images are returned via `#!python reply.images`.

```python linenums="1"
from autogen.beta import Agent
from autogen.beta.config import OpenAIResponsesConfig
from autogen.beta.tools import ImageGenerationTool

agent = Agent(
    "designer",
    config=OpenAIResponsesConfig(model="gpt-4.1"),
    tools=[
        ImageGenerationTool(
            quality="high",
            size="1024x1024",
            output_format="png",
            background="transparent",
        ),
    ],
)

reply = await agent.ask("Generate a logo for a coffee shop.")
images: list[bytes] = reply.images
```

| Parameter | Description |
| :--- | :--- |
| `quality` | `"low"`, `"medium"`, `"high"`, or `"auto"` |
| `size` | e.g. `"1024x1024"`, `"1536x1024"`, or `"auto"` |
| `background` | `"transparent"`, `"opaque"`, or `"auto"` |
| `output_format` | `"png"`, `"jpeg"`, or `"webp"` |
| `output_compression` | 0–100, for jpeg/webp only |
| `partial_images` | 1–3, number of partial images to stream |

!!! note
    `ImageGenerationTool` is only supported by OpenAI (Responses API). Using it with other providers will raise an error.

## Anthropic Tool Versions

Anthropic versions their server-side tools. Newer versions support dynamic filtering (Claude writes code to filter results before loading into context), but require Opus 4.6 or Sonnet 4.6.

Set the version on each built-in tool (defaults match the older Anthropic tool revisions):

```python linenums="1"
from autogen.beta.tools import WebFetchTool, WebSearchTool

tools = [
    WebSearchTool(version="web_search_20260209"),  # default: web_search_20250305
    WebFetchTool(version="web_fetch_20260209"),    # default: web_fetch_20250910
]
```

The default versions are compatible with all Claude models including Haiku.
