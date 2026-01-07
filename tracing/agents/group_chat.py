import asyncio
import os
from typing import Annotated

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import a_initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from autogen.opentelemetry import instrument_llm_wrapper, instrument_pattern

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
    resource = Resource.create(attributes={"service.name": "local-group-chat"})
    tracer_provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:14317")
    processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)
    instrument_llm_wrapper(tracer_provider=tracer_provider)
    instrument_pattern(pattern, tracer_provider=tracer_provider)

    result, _, _ = await a_initiate_group_chat(
        pattern=pattern,
        messages="",
        max_rounds=10,
    )
    print(result)


asyncio.run(main())
