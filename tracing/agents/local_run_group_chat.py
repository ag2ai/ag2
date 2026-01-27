"""Test file for run_group_chat tracing."""

import os

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import run_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from autogen.opentelemetry import instrument_llm_wrapper, instrument_pattern

load_dotenv()

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Create specialized agents
researcher = ConversableAgent(
    name="researcher",
    system_message="You research topics and provide factual information. Be concise.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

writer = ConversableAgent(
    name="writer",
    system_message="You take research and write clear summaries. Be concise. Say TERMINATE when done.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Create the pattern (no user_agent for non-interactive test)
pattern = AutoPattern(
    initial_agent=researcher,
    agents=[researcher, writer],
    group_manager_args={"llm_config": llm_config},
    user_agent=None,  # No human interaction needed
)


def main():
    """Run using sync run_group_chat function."""
    resource = Resource.create(attributes={"service.name": "local-run-group-chat"})
    tracer_provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:14317")
    processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)

    instrument_llm_wrapper(tracer_provider=tracer_provider)
    instrument_pattern(pattern, tracer_provider=tracer_provider)

    # Use sync run_group_chat
    response = run_group_chat(
        pattern=pattern,
        messages="What are the three laws of thermodynamics? Summarize briefly.",
        max_rounds=5,
    )

    # Process events - this waits for completion and prints output
    response.process()

    print(f"\nFinal result: {response.summary}")


if __name__ == "__main__":
    main()
