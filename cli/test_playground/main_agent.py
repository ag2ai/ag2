"""Agent with main() entry point for testing ag2 run."""

from autogen import AssistantAgent, LLMConfig


async def main(message="Hello!"):
    config = LLMConfig(
        {"model": "gemini-3.1-flash-lite-preview", "api_type": "google"},
    )

    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant. Keep answers short (1-2 sentences).",
        llm_config=config,
    )

    user = AssistantAgent(
        name="user",
        llm_config=config,
    )

    run_process = user.run(assistant, message=message, max_turns=1)

    run_process.process()

    return run_process.summary
