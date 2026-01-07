"""Test file for sync run() method tracing (single agent mode)."""

import os
from typing import Annotated

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from autogen import ConversableAgent, LLMConfig
from autogen.opentelemetry import instrument_agent, instrument_llm_wrapper

load_dotenv()

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)


def calculate(expression: Annotated[str, "A math expression to evaluate"]) -> str:
    """Safely evaluate a simple math expression."""
    try:
        # Only allow basic math operations
        allowed = set("0123456789+-*/.(). ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Create a calculator agent with a tool
calculator = ConversableAgent(
    name="calculator",
    system_message="You are a calculator assistant. Use the calculate tool to evaluate expressions. Reply TERMINATE when done.",
    llm_config=llm_config,
    functions=[calculate],
    human_input_mode="NEVER",
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
)


def main():
    """Run using sync run() method in single-agent mode."""
    resource = Resource.create(attributes={"service.name": "local-run"})
    tracer_provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:14317")
    processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)

    instrument_llm_wrapper(tracer_provider=tracer_provider)
    instrument_agent(calculator, tracer_provider=tracer_provider)

    # Use sync run() in single-agent mode (creates temporary executor)
    response = calculator.run(
        message="What is 15 * 7 + 23?",
        max_turns=3,
    )

    # Process events - this waits for completion and prints output
    response.process()

    print(f"\nFinal result: {response.summary}")


if __name__ == "__main__":
    main()
