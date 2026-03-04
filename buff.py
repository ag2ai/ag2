import pytest

from autogen.beta import Agent
from autogen.beta.events import ToolCall
from autogen.beta.testing import TestConfig


@pytest.mark.asyncio
async def test_tool_raise_exc():
    # Define a tool that raises an error
    def failing_tool() -> str:
        raise ValueError("Something went wrong")

    test_config = TestConfig(
        ToolCall(name="failing_tool"),
        "result",
    )

    agent = Agent(
        "test_agent",
        config=test_config,
        tools=[failing_tool],
    )

    with pytest.raises(ValueError, match="Something went wrong"):
        await agent.ask("Hi!")
