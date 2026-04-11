"""Coding agent — specializes in code answers. For arena comparisons."""

from autogen import AssistantAgent, LLMConfig

config = LLMConfig(
    {"model": "gemini-3.1-flash-lite-preview", "api_type": "google"},
)

agent = AssistantAgent(
    name="coder",
    system_message=(
        "You are a coding assistant. When asked a question, always include a "
        "code example in Python. Keep explanations brief but code should be "
        "runnable and well-commented. Always end with TERMINATE."
    ),
    llm_config=config,
)
