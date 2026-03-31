"""Concise agent — answers in 1-2 sentences. For arena comparisons."""

from autogen import AssistantAgent, LLMConfig

config = LLMConfig(
    {"model": "gemini-3.1-flash-lite-preview", "api_type": "google"},
)

agent = AssistantAgent(
    name="concise_helper",
    system_message=(
        "You are a concise assistant. Answer every question in 1-2 sentences maximum. "
        "Be direct and factual. No fluff. Always end with TERMINATE."
    ),
    llm_config=config,
)
