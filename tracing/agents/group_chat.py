import asyncio
import os

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import a_initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from autogen.instrumentation import instrument_pattern, setup_instrumentation

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

triage_agent = ConversableAgent(
    name="triage_agent",
    system_message="""You are a triage agent. For each user query,
    identify whether it is a technical issue or a general question. Route
    technical issues to the tech agent and general questions to the general agent.
    Do not provide suggestions or answers, only route the query.""",
    llm_config=llm_config,
)

tech_agent = ConversableAgent(
    name="tech_agent",
    system_message="""You solve technical problems like software bugs
    and hardware issues.""",
    llm_config=llm_config,
)

general_agent = ConversableAgent(
    name="general_agent",
    system_message="You handle general, non-technical support questions.",
    llm_config=llm_config,
)

user = ConversableAgent(
    name="user",
    human_input_mode="ALWAYS",
)

pattern = AutoPattern(
    initial_agent=triage_agent,
    agents=[triage_agent, tech_agent, general_agent],
    user_agent=user,
    group_manager_args={"llm_config": llm_config},
)


async def main():
    tracer, _ = setup_instrumentation("local-group-chat")
    instrument_pattern(pattern, tracer)

    result, _, _ = await a_initiate_group_chat(
        pattern=pattern,
        messages="",
        max_rounds=10,
    )
    print(result)


asyncio.run(main())
