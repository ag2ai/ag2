from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from autogen.remote import RemoteAgent

llm_config = LLMConfig(
    model="Qwen3-Coder",
    base_url="http://172.29.24.150:3003/v1",
    api_key="NotRequired",
    api_type="openai",
)

triage_agent = RemoteAgent(
    "http://localhost:8000",
    name="triage_agent",
)

tech_agent = RemoteAgent(
    "http://localhost:8000",
    name="tech_agent",
)

general_agent = RemoteAgent(
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
