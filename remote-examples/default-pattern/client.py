import os

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget, OnCondition, RevertToUserTarget, StringLLMCondition
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.remote import HTTPRemoteAgent

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

user = ConversableAgent(
    name="user",
    human_input_mode="ALWAYS",
)

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

pattern = DefaultPattern(
    initial_agent=triage_agent,
    agents=[triage_agent, tech_agent, general_agent],
    user_agent=user,
    group_manager_args={"llm_config": llm_config},
)

triage_agent.handoffs.add_llm_conditions([
    OnCondition(
        target=AgentTarget(tech_agent),
        condition=StringLLMCondition(prompt="When the user query is related to technical issues."),
    ),
    OnCondition(
        target=AgentTarget(agent=general_agent),
        condition=StringLLMCondition(prompt="When the user query is related to general questions."),
    ),
])
tech_agent.handoffs.set_after_work(RevertToUserTarget())
general_agent.handoffs.set_after_work(RevertToUserTarget())

result, context, last_agent = initiate_group_chat(
    pattern=pattern,
    messages="My laptop keeps shutting down randomly. Can you help?",
    max_rounds=10,
)
