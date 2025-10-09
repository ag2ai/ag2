import os
from datetime import datetime
from typing import Annotated

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat.group import ContextVariables
from autogen.agentchat.group.reply_result import ReplyResult
from autogen.remote import HTTPAgentBus

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
    # context: ChatContext,
    context_variables: ContextVariables,
) -> str:
    # print(context.chat_messages)
    print(context_variables)
    date = datetime.strptime(date_string, "%Y-%m-%d")

    return ReplyResult(
        # context_variables=context_variables,
        message=date.strftime("%A"),
    )


app = HTTPAgentBus(agents=[agent])
