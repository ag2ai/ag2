"""Detailed agent — gives thorough answers. For arena comparisons."""

from autogen import AssistantAgent, LLMConfig

config = LLMConfig(
    {"model": "gemini-3.1-flash-lite-preview", "api_type": "google"},
)

agent = AssistantAgent(
    name="detailed_helper",
    system_message=(
        "You are a thorough assistant. For every question, provide a detailed answer "
        "with context, examples, and nuance. Aim for 3-5 sentences minimum. "
        "Structure your response clearly. Always end with TERMINATE."
    ),
    llm_config=config,
)
