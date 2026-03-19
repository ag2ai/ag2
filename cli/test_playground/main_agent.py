"""Agent with main() entry point for testing ag2 run."""

import os

from autogen import AssistantAgent, LLMConfig, UserProxyAgent


async def main(message="Hello!"):
    config = LLMConfig(
        {"model": "gemini-3-flash-preview", "api_type": "google", "api_key": os.environ["GEMINI_API_KEY"]},
    )

    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant. Keep answers short (1-2 sentences).",
        llm_config=config,
    )

    user = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    result = user.initiate_chat(assistant, message=message)
    return result
