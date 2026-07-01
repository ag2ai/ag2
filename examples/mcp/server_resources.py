import uvicorn

from ag2 import Agent
from ag2.config import AnthropicConfig
from ag2.mcp import MCPServer, Prompt, PromptArgument, Resource, ResourceTemplate, SessionConfig

agent = Agent(
    name="claude",
    prompt="You are a concise assistant.",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
)

# A static resource (fixed URI) and a templated one ({city} is matched from the URI).
readme = Resource(
    uri="docs://readme",
    name="readme",
    description="Project overview.",
    read=lambda: "AG2 beta MCP server example.",
    mime_type="text/markdown",
)

weather = ResourceTemplate(
    uri_template="weather://{city}",
    name="weather",
    description="Current (stubbed) weather for a city.",
    read=lambda vars: f"It is sunny in {vars['city']}.",
)

translate = Prompt(
    name="translate",
    description="Ask the agent to translate text into a target language.",
    arguments=(
        PromptArgument(name="text", description="Text to translate.", required=True),
        PromptArgument(name="language", description="Target language.", required=True),
    ),
    render=lambda a: f"Translate the following into {a['language']}:\n\n{a['text']}",
)

# Multi-turn history is on by default; SessionConfig tunes the bound / idle TTL.
app = MCPServer(
    agent,
    resources=[readme],
    resource_templates=[weather],
    prompts=[translate],
    sessions=SessionConfig(max_sessions=256, ttl=3600),
    path="/mcp",
)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
