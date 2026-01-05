import asyncio
import os
from typing import Annotated

from dotenv import load_dotenv

from autogen import ConversableAgent, LLMConfig
from autogen.opentelemetry import instrument_agent, instrument_llm_wrapper, setup_instrumentation

load_dotenv()

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

WEATHER_PROMPT = (
    "You are a weather information agent. Use the get_weather tool to provide weather information for requested cities."
)


def get_weather(city: Annotated[str, "The city name"]) -> str:
    """Get weather information for a city (mock function)."""
    weather_data = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
        "paris": "Partly cloudy, 20°C",
    }
    result = weather_data.get(city.lower(), f"Weather data not available for {city}")
    # raise Exception("Simulated failure in get_weather function")  # Simulate a failure for testing, will add error.type to span with "ExecutionError" as value
    return result


weather_agent = ConversableAgent(
    name="weather",
    system_message=WEATHER_PROMPT,
    human_input_mode="NEVER",
    functions=[get_weather],
    llm_config=llm_config,
)


async def get_new_york_weather(prompt: str) -> str:
    response = await weather_agent.a_run(message=prompt, max_turns=2)

    await response.process()

    weather_agent_messages = [
        y for x in weather_agent.chat_messages.values() for y in x if y.get("name", "") == weather_agent.name
    ]
    if not weather_agent_messages:
        return "No messages from weather agent."
    last_message = weather_agent_messages[-1]["content"]
    lines = last_message.splitlines()
    return "\n".join(lines[1:-1])


if __name__ == "__main__":
    tracer = setup_instrumentation("local-tools", "http://127.0.0.1:14317")
    instrument_llm_wrapper(tracer)
    instrument_agent(weather_agent, tracer)

    code = asyncio.run(get_new_york_weather("What's the weather like today in New York?"))

    print(code)
