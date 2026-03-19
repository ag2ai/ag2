"""Single agent for testing ag2 run / ag2 chat."""

import os

from autogen import AssistantAgent, LLMConfig

config = LLMConfig(
    {"model": "gemini-3-flash-preview", "api_type": "google", "api_key": os.environ["GEMINI_API_KEY"]},
)

agent = AssistantAgent(
    name="helper",
    system_message="You are a helpful assistant. Keep answers short (1-2 sentences).",
    llm_config=config,
)
