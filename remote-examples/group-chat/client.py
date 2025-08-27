import os

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from autogen.remote import HTTPRemoteAgent

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

triage_agent = HTTPRemoteAgent(
    "http://localhost:8000",
    name="triage_agent",
)

tech_agent = HTTPRemoteAgent(
    "http://localhost:8000",
    name="tech_agent",
)

general_agent = HTTPRemoteAgent(
    "http://localhost:8000",
    name="general_agent",
)

user = ConversableAgent(name="user", human_input_mode="ALWAYS")

pattern = AutoPattern(
    initial_agent=triage_agent,
    agents=[triage_agent, tech_agent, general_agent],
    user_agent=user,
    group_manager_args={"llm_config": llm_config},
)

result, context, last_agent = initiate_group_chat(
    pattern=pattern, messages="My laptop keeps shutting down randomly. Can you help?", max_rounds=10
)
