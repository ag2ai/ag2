"""Detailed agent — gives thorough answers. For arena comparisons."""

import os

from autogen import AssistantAgent, LLMConfig

config = LLMConfig(
    {"model": "gemini-3-flash-preview", "api_type": "google", "api_key": os.environ["GEMINI_API_KEY"]},
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
