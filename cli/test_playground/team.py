"""Two-agent team for testing ag2 run with agents list."""

import os

from autogen import AssistantAgent, LLMConfig

config = LLMConfig(
    {"model": "gemini-3-flash-preview", "api_type": "google", "api_key": os.environ["GEMINI_API_KEY"]},
)

researcher = AssistantAgent(
    name="researcher",
    system_message=(
        "You are a researcher. When asked a question, provide 2-3 key facts. "
        "Keep it brief. After sharing facts, say TERMINATE."
    ),
    llm_config=config,
)
writer = AssistantAgent(
    name="writer",
    system_message=(
        "You are a writer. Take the researcher's facts and write a single "
        "concise paragraph summarizing them. Then say TERMINATE."
    ),
    llm_config=config,
)

agents = [researcher, writer]
