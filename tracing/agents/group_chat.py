import asyncio
import os
from typing import Annotated

from dotenv import load_dotenv

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import a_initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from autogen.instrumentation import instrument_llm_wrapper, instrument_pattern, setup_instrumentation

load_dotenv()

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


def record_tech_fault(fault: Annotated[str, "Description of the fault"]) -> str:
    """Mocks the recording of a technical fault."""
    # raise Exception("Simulated failure in get_weather function")  # Simulate a failure for testing, will add error.type to span with "ExecutionError" as value
    return f"Successfully recorded fault into the system: {fault}"


tech_agent = ConversableAgent(
    name="tech_agent",
    system_message="""You solve technical problems like software bugs and hardware issues.
You must always record faults into the fault management system using the record_tech_fault tool.""",
    functions=[record_tech_fault],
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
    tracer = setup_instrumentation("local-group-chat", "http://127.0.0.1:14317")
    instrument_llm_wrapper(tracer)
    instrument_pattern(pattern, tracer)

    result, _, _ = await a_initiate_group_chat(
        pattern=pattern,
        messages="",
        max_rounds=10,
    )
    print(result)


asyncio.run(main())
