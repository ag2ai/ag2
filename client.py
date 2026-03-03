import asyncio
from datetime import datetime
from typing import Annotated

import httpx
from pydantic import Field

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import a_initiate_group_chat
from autogen.agentchat.group import ContextVariables
from autogen.agentchat.group.patterns import AutoPattern
from autogen.beta import Agent, Context, Variable
from autogen.beta.config import OpenAIConfig

CONFIG_DICT = {
    "model": "gpt-5-nano",
    "reasoning_effort": "low",
    "streaming": True,
}

new_agent = Agent(
    "triage_agent",
    config=OpenAIConfig(**(CONFIG_DICT | {"http_client": httpx.AsyncClient(verify=False)})),
)


@new_agent.tool
def get_weekday(
    ctx: Context,
    date_string: Annotated[str, Field(description="Format: YYYY-MM-DD")],
    issue_count: Annotated[int, Variable()],
) -> str:
    """Get the day of the week for a given date."""
    print(ctx.variables)
    ctx.variables["issue_count"] = issue_count + 1
    return datetime.strptime(date_string, "%Y-%m-%d").strftime("%A")


llm_config = LLMConfig(CONFIG_DICT | {"http_client": httpx.Client(verify=False)})

agent = ConversableAgent(
    name="local",
    human_input_mode="NEVER",
    llm_config=llm_config,
    context_variables=ContextVariables({"issue_count": 1}),
)

user = ConversableAgent(
    name="user",
    human_input_mode="ALWAYS",
)

conv = new_agent.as_conversable()

pattern = AutoPattern(
    initial_agent=conv,
    agents=[conv, agent],
    user_agent=user,
    group_manager_args={"llm_config": llm_config},
    context_variables=ContextVariables({"issue_count": 1}),
)


async def main(message: str) -> None:
    result, context, last_agent = await a_initiate_group_chat(
        pattern=pattern,
        messages=message,
        max_rounds=10,
    )
    print(context.data)


if __name__ == "__main__":
    asyncio.run(main("I was born on 25th of March 1995, what day was it?"))
