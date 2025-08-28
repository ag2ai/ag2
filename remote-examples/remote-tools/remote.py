import os
from datetime import datetime
from typing import Annotated

from autogen import ConversableAgent, LLMConfig
from autogen.remote import HTTPAgentBus
from autogen.tools.dependency_injection import ChatContext

llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

agent = ConversableAgent(
    name="remote",
    llm_config=llm_config,
    human_input_mode="NEVER",
    silent=True,
)


@agent.register_for_llm(name="get_weekday", description="Get the day of the week for a given date")
@agent.register_for_execution(name="get_weekday")
def get_weekday(
    date_string: Annotated[str, "Format: YYYY-MM-DD"],
    context: ChatContext,
) -> str:
    print(context.chat_messages)
    date = datetime.strptime(date_string, "%Y-%m-%d")
    return date.strftime("%A")


app = HTTPAgentBus(agents=[agent])
