from dirty_equals import IsPartialDict

from autogen.beta.tools import Tool, tool

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


def test_simple_tool() -> None:
    @tool
    def my_tool(a: str, b: int) -> str:
        """Tool description."""
        return ""

    assert my_tool.schema.to_api() == DEFAULT_SCHEMA


def test_override_options() -> None:
    @tool(name="another_name", description="another_description")
    def my_tool(a: str, b: int) -> str:
        """Tool description."""
        return ""

    assert my_tool.schema.to_api() == {
        "function": IsPartialDict({
            "description": "another_description",
            "name": "another_name",
        }),
        "type": "function",
    }


def test_ensure_tools() -> None:
    def my_tool(a: str, b: int) -> str:
        """Tool description."""
        return ""

    assert Tool.ensure_tool(my_tool).schema.to_api() == DEFAULT_SCHEMA


def test_ensure_tool_from_tool() -> None:
    @tool
    def my_tool(a: str, b: int) -> str:
        """Tool description."""
        return ""

    assert Tool.ensure_tool(my_tool).schema.to_api() == DEFAULT_SCHEMA
