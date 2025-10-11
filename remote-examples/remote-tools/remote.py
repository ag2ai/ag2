import os
from datetime import datetime
from typing import Annotated

from autogen import ConversableAgent, LLMConfig
from autogen.a2a import A2aAgentServer
from autogen.agentchat import ContextVariables, ReplyResult

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
    context_variables: ContextVariables,
) -> str:
    context_variables["issue_count"] = context_variables.get("issue_count", 0) + 1
    return ReplyResult(
        message=datetime.strptime(date_string, "%Y-%m-%d").strftime("%A"),
        context_variables=context_variables,
    )


app = A2aAgentServer(agent, url="http://0.0.0.0:8000").build()
