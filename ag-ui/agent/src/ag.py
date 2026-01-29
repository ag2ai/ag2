from datetime import datetime
from textwrap import dedent
from typing import Annotated

from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig({"model": "gpt-5-mini"})


agent = ConversableAgent(
    name="agent",
    system_message=dedent("""
        You are a helpful assistant that helps manage and discuss proverbs.

        The user has a list of proverbs that you can help them manage.
        You have tools available to add, set, or retrieve proverbs from the list.

        When discussing proverbs, ALWAYS use the get_proverbs tool to see the current list before
        mentioning, updating, or discussing proverbs with the user.
    """).strip(),
    llm_config=llm_config,
)


@agent.register_for_llm(
    name="get_weekday",
    description="Get the day of the week for a given date",
)
def get_weekday(
    date_string: Annotated[str, "Format: YYYY-MM-DD"],
) -> str:
    return datetime.strptime(date_string, "%Y-%m-%d").strftime("%A")


@agent.register_for_llm(
    name="get_weather",
    description="Get the weather for a given location. Ensure location is fully spelled out.",
)
def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny."
