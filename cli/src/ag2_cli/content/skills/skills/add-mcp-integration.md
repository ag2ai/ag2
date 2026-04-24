---
name: add-mcp-integration
description: Wire a Model Context Protocol (MCP) server into AG2 agents using create_toolkit. Use when the user wants to connect external tools via MCP.
---

# Add MCP Integration to AG2

You are an expert at integrating MCP servers with AG2 agents.

## 1. Understand the MCP Server

- What MCP server does the user want to connect?
- Is it a stdio-based server (runs as subprocess) or SSE-based (HTTP endpoint)?
- What tools does the server provide?

## 2. Generate the Integration

### Stdio Server (most common)

```python
import asyncio
from autogen import ConversableAgent, LLMConfig
from autogen.mcp import create_toolkit
from autogen.mcp.mcp_client import StdioConfig

async def main():
    llm_config = LLMConfig(
        {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
    )

    agent = ConversableAgent(
        name="assistant",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # Configure the MCP server
    stdio_config = StdioConfig(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
    )

    # Create toolkit from MCP server and register with agent
    async with stdio_config.connect() as session:
        toolkit = await create_toolkit(session)
        toolkit.register_for_llm(agent)
        toolkit.register_for_execution(agent)

        # Now use the agent — it has all MCP tools available
        user_proxy = ConversableAgent(
            name="user",
            llm_config=False,
            human_input_mode="NEVER",
        )
        result = await user_proxy.a_run(
            agent,
            message="List files in the directory",
        )
        await result.process()

asyncio.run(main())
```

### SSE Server

```python
from autogen.mcp.mcp_client import SseConfig

sse_config = SseConfig(
    url="http://localhost:8080/sse",
    headers={"Authorization": f"Bearer {os.environ['MCP_TOKEN']}"},
    timeout=5,
    sse_read_timeout=300,
)

async with sse_config.connect() as session:
    toolkit = await create_toolkit(session)
    toolkit.register_for_llm(agent)
    toolkit.register_for_execution(agent)
```

## 3. StdioConfig Parameters

```python
StdioConfig(
    command="npx",                    # Command to run
    args=["-y", "server-package"],    # Arguments
    environment={"KEY": "value"},     # Optional env vars
    working_dir="/path/to/dir",       # Optional working directory
    encoding="utf-8",                 # Default
)
```

## 4. create_toolkit Options

```python
toolkit = await create_toolkit(
    session,
    use_mcp_tools=True,              # Import tools from MCP server
    use_mcp_resources=True,           # Import resources from MCP server
    resource_download_folder="./mcp_resources",  # Where to save resources
)
```

## 5. Requirements

- Install MCP support: `pip install ag2[mcp]`
- MCP integration is async — use `async/await` and `a_run`
- The MCP server must be running during the agent conversation
- Use context manager (`async with`) to properly manage server lifecycle
