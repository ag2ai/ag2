# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Beta TelemetryMiddleware example with real OpenTelemetry stack.

Uses TestConfig for deterministic results — no API key needed.
Exercises all span types: agent, llm, tool, tool-error, human_input.

Prerequisites:
    1. Start the local OTEL stack:
       cd examples/tracing && docker-compose up -d

    2. Install dependencies:
       pip install "ag2[tracing]"

    3. Run this script:
       python examples/tracing/beta_telemetry_example.py

    4. Open Grafana at http://localhost:3333
       Go to Explore > Tempo > Search > Service Name = "ag2-beta-tracing"
"""

import asyncio

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from autogen.beta import Agent
from autogen.beta.annotations import Context
from autogen.beta.events import HumanInputRequest, HumanMessage, ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent
from autogen.beta.middleware.builtin import TelemetryMiddleware
from autogen.beta.testing import TestConfig
from autogen.beta.tools import tool

# --- OpenTelemetry setup ---
resource = Resource.create(attributes={"service.name": "ag2-beta-tracing"})
tracer_provider = TracerProvider(resource=resource)
exporter = OTLPSpanExporter(endpoint="http://localhost:14317", insecure=True)
tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(tracer_provider)

telemetry = TelemetryMiddleware(tracer_provider=tracer_provider, capture_content=True)


# --- Tools ---
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "new york": "Sunny, 72F",
        "london": "Cloudy, 15C",
        "tokyo": "Rainy, 18C",
        "paris": "Partly cloudy, 20C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
async def confirm_action(action: str, ctx: Context) -> str:
    """Ask the user to confirm an action before proceeding."""
    answer = await ctx.input(f"Please confirm: {action} (yes/no)")
    if answer.strip().lower() == "yes":
        return f"User confirmed: {action}"
    return f"User declined: {action}"


@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    prices = {"AAPL": "$195.23", "GOOGL": "$142.67", "MSFT": "$378.91"}
    if ticker.upper() not in prices:
        raise ValueError(f"Unknown ticker: {ticker}")
    return f"{ticker.upper()}: {prices[ticker.upper()]}"


# --- HITL hook (auto-confirms for demo) ---
def hitl_hook(event: HumanInputRequest) -> HumanMessage:
    print(f"  [HITL] Agent asks: {event.content}")
    answer = "yes"
    print(f"  [HITL] Auto-responding: {answer}")
    return HumanMessage(content=answer)


# --- Helper to build a canned ModelResponse ---
def _resp(
    content: str | None = None,
    tool_calls: list[ToolCallEvent] | None = None,
    finish_reason: str = "stop",
) -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(content=content) if content else None,
        tool_calls=ToolCallsEvent(calls=tool_calls) if tool_calls else ToolCallsEvent(),
        usage={"prompt_tokens": 50, "completion_tokens": 20},
        model="gpt-4o-mini-2024-07-18",
        provider="openai",
        finish_reason=finish_reason,
    )


async def main():
    # --- Turn 1: Tool execution (agent → llm → tool → llm) ---
    print("--- Turn 1: Tool execution (weather) ---")
    weather_agent = Agent(
        "weather_assistant",
        config=TestConfig(
            _resp(
                tool_calls=[ToolCallEvent(id="c1", name="get_weather", arguments='{"city": "Tokyo"}')],
                finish_reason="tool_calls",
            ),
            _resp(content="The weather in Tokyo is Rainy, 18C."),
        ),
        tools=[get_weather],
        middleware=[
            TelemetryMiddleware(tracer_provider=tracer_provider, agent_name="weather_assistant", capture_content=True)
        ],
    )
    reply = await weather_agent.ask("What is the weather in Tokyo?")
    print(f"Response: {reply.content}\n")

    # --- Turn 2: Plain LLM call (agent → llm, no tools) ---
    print("--- Turn 2: Plain LLM call ---")
    general_agent = Agent(
        "general_assistant",
        config=TestConfig(_resp(content="The capital of France is Paris.")),
        middleware=[TelemetryMiddleware(tracer_provider=tracer_provider, agent_name="general_assistant")],
    )
    reply2 = await general_agent.ask("What is the capital of France?")
    print(f"Response: {reply2.content}\n")

    # --- Turn 3: Human-in-the-loop (agent → llm → tool → HITL → llm) ---
    print("--- Turn 3: Human-in-the-loop ---")
    hitl_agent = Agent(
        "hitl_assistant",
        config=TestConfig(
            _resp(
                tool_calls=[ToolCallEvent(id="c2", name="confirm_action", arguments='{"action": "deploy to production"}')],
                finish_reason="tool_calls",
            ),
            _resp(content="Deployment confirmed and proceeding!"),
        ),
        tools=[confirm_action],
        hitl_hook=hitl_hook,
        middleware=[
            TelemetryMiddleware(tracer_provider=tracer_provider, agent_name="hitl_assistant", capture_content=True)
        ],
    )
    reply3 = await hitl_agent.ask("Deploy the latest build to production")
    print(f"Response: {reply3.content}\n")

    # --- Turn 4: Tool error (agent → llm → tool[ERROR]) ---
    print("--- Turn 4: Tool error ---")
    stock_agent = Agent(
        "stock_assistant",
        config=TestConfig(
            _resp(
                tool_calls=[ToolCallEvent(id="c3", name="get_stock_price", arguments='{"ticker": "INVALID"}')],
                finish_reason="tool_calls",
            ),
            _resp(content="Sorry, that ticker is not available."),
        ),
        tools=[get_stock_price],
        middleware=[
            TelemetryMiddleware(tracer_provider=tracer_provider, agent_name="stock_assistant", capture_content=True)
        ],
    )
    try:
        reply4 = await stock_agent.ask("What is the stock price of INVALID?")
        print(f"Response: {reply4.content}\n")
    except Exception as e:
        print(f"Expected error: {e}\n")

    # Flush spans
    tracer_provider.force_flush()
    print("Traces sent! Open http://localhost:3333 to view in Grafana.")
    print("\nSpan types exercised:")
    print("  Turn 1: agent, llm, tool, llm")
    print("  Turn 2: agent, llm")
    print("  Turn 3: agent, llm, tool + human_input, llm")
    print("  Turn 4: agent, llm, tool (error)")


if __name__ == "__main__":
    asyncio.run(main())
