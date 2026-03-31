"""Single agent for testing ag2 run / ag2 chat."""

from autogen import AssistantAgent, LLMConfig

config = LLMConfig(
    {"model": "gemini-3.1-flash-lite-preview", "api_type": "google"},
)

agent = AssistantAgent(
    name="helper",
    system_message="You are a helpful assistant. Keep answers short (1-2 sentences).",
    llm_config=config,
)
