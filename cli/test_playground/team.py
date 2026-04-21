"""Two-agent team for testing ag2 run with agents list."""

from autogen import AssistantAgent, LLMConfig

config = LLMConfig(
    {"model": "gemini-3.1-flash-lite-preview", "api_type": "google"},
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
