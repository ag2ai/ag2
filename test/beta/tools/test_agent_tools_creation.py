from unittest.mock import MagicMock

from dirty_equals import IsPartialDict

from autogen.beta import Agent, tool

DEFAULT_SCHEMA = {
    "function": {
        "description": "Tool description.",
        "name": "my_tool",
        "parameters": {
            "properties": {
                "a": {
                    "title": "A",
                    "type": "string",
                },
                "b": {
                    "title": "B",
                    "type": "integer",
                },
            },
            "required": [
                "a",
                "b",
            ],
            "type": "object",
        },
    },
    "type": "function",
}


def test_agent_with_function(mock: MagicMock) -> None:
    def my_tool(a: str, b: int) -> str:
        """Tool description."""
        return ""

    agent = Agent(
        "",
        client=mock,
        tools=[my_tool],
    )

    assert agent.tools[0].schema.to_api() == DEFAULT_SCHEMA


def test_agent_with_tool(mock: MagicMock) -> None:
    @tool
    def my_tool(a: str, b: int) -> str:
        """Tool description."""
        return ""

    agent = Agent(
        "",
        client=mock,
        tools=[my_tool],
    )

    assert agent.tools[0].schema.to_api() == DEFAULT_SCHEMA


def test_agent_with_tool_decorator(mock: MagicMock) -> None:
    agent = Agent("", client=mock)

    @agent.tool
    def my_tool(a: str, b: int) -> str:
        """Tool description."""
        return ""

    assert agent.tools[0].schema.to_api() == DEFAULT_SCHEMA


def test_agent_with_tool_decorator_options_override(mock: MagicMock) -> None:
    agent = Agent("", client=mock)

    @agent.tool(name="another_name", description="another_description")
    def my_tool(a: str, b: int) -> str:
        """Tool description."""
        return ""

    assert agent.tools[0].schema.to_api() == {
        "function": IsPartialDict({
            "description": "another_description",
            "name": "another_name",
        }),
        "type": "function",
    }
