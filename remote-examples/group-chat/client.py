import asyncio
import os

from autogen import ConversableAgent, LLMConfig
from autogen.a2a import A2aRemoteAgent
from autogen.agentchat import a_initiate_group_chat
from autogen.agentchat.group import ContextVariables
from autogen.agentchat.group.patterns import AutoPattern

triage_agent = A2aRemoteAgent(
    "http://localhost:8000/triage/",
    name="triage_agent",
)

tech_agent = A2aRemoteAgent(
    "http://localhost:8000/tech/",
    name="tech_agent",
)

general_agent = A2aRemoteAgent(
    "http://localhost:8000/general/",
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

context = ContextVariables({
    "user_name": "Alex",
    "issue_count": 0,
})

pattern = AutoPattern(
    initial_agent=triage_agent,
    agents=[triage_agent, tech_agent, general_agent],
    user_agent=user,
    context_variables=context,
    group_manager_args={"llm_config": llm_config},
)


async def main():
    result, context, last_agent = await a_initiate_group_chat(
        pattern=pattern,
        messages="My laptop keeps shutting down randomly. Can you help?",
        max_rounds=10,
    )
    print(context)


asyncio.run(main())
