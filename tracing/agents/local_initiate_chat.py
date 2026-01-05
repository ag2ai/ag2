"""Test file for sync initiate_chat tracing."""

import os

from dotenv import load_dotenv

from autogen import ConversableAgent, LLMConfig
from autogen.opentelemetry import instrument_agent, instrument_llm_wrapper, setup_instrumentation

load_dotenv()

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Create a simple assistant agent
assistant = ConversableAgent(
    name="assistant",
    system_message="You are a helpful assistant. Keep responses concise.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Create a user proxy agent
user_proxy = ConversableAgent(
    name="user_proxy",
    llm_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
)


def main():
    """Run a sync two-agent chat using initiate_chat."""
    tracer = setup_instrumentation("local-initiate-chat", "http://127.0.0.1:14317")
    instrument_llm_wrapper(tracer)
    instrument_agent(assistant, tracer)
    instrument_agent(user_proxy, tracer)

    # Use sync initiate_chat
    chat_result = user_proxy.initiate_chat(
        assistant,
        message="What is 2 + 2? Reply with just the answer and then say TERMINATE.",
        max_turns=2,
    )

    print(f"Summary: {chat_result.summary}")
    print(f"Cost: {chat_result.cost}")


if __name__ == "__main__":
    main()
