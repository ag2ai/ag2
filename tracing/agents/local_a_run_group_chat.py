"""Test file for a_run_group_chat tracing (async version)."""

import asyncio
import os

from dotenv import load_dotenv

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import a_run_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from autogen.instrumentation import instrument_pattern, setup_instrumentation

load_dotenv()

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Create specialized agents
analyst = ConversableAgent(
    name="analyst",
    system_message="You analyze data and identify trends. Be concise.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

reporter = ConversableAgent(
    name="reporter",
    system_message="You create clear reports from analysis. Be concise. Say TERMINATE when done.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Create the pattern (no user_agent for non-interactive test)
pattern = AutoPattern(
    initial_agent=analyst,
    agents=[analyst, reporter],
    group_manager_args={"llm_config": llm_config},
    user_agent=None,  # No human interaction needed
)


async def main():
    """Run using async a_run_group_chat function."""
    tracer = setup_instrumentation("local-a-run-group-chat", "http://127.0.0.1:14317")
    instrument_pattern(pattern, tracer)

    # Use async a_run_group_chat
    response = await a_run_group_chat(
        pattern=pattern,
        messages="Analyze the trend of remote work adoption since 2020. Provide a brief report.",
        max_rounds=5,
    )

    # Process events asynchronously - this waits for completion and prints output
    await response.process()

    print(f"\nFinal result: {response.summary}")


if __name__ == "__main__":
    asyncio.run(main())
