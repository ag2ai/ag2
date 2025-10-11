import os

from starlette.applications import Starlette
from starlette.routing import Mount

from autogen import ConversableAgent, LLMConfig
from autogen.a2a import A2aAgentServer
from autogen.agentchat import ContextVariables, ReplyResult

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


@triage_agent.register_for_llm(
    name="get_user_context",
    description="Use `get_user_context` tool to understand the current session",
)
@triage_agent.register_for_execution(name="get_user_context")
def get_user_context(
    context_variables: ContextVariables,
    # chat_ctx: ChatContext,
) -> str:
    context_variables["issue_count"] = context_variables.get("issue_count", 0) + 1
    return ReplyResult(
        context_variables=context_variables,
        message=f"""
        Current session information:
        - User Name: {context_variables.get("user_name")}
        """,
    )


tech_agent = ConversableAgent(
    name="tech_agent",
    system_message="""You solve technical problems like software bugs
    and hardware issues.""",
    llm_config=llm_config,
)

general_agent = ConversableAgent(
    name="general_agent",
    system_message="You handle general, non-technical support questions.",
    llm_config=llm_config,
)


app = Starlette(
    routes=[
        Mount("/triage", A2aAgentServer(triage_agent, url="http://0.0.0.0:8000/triage/").build()),
        Mount("/tech", A2aAgentServer(tech_agent, url="http://0.0.0.0:8000/tech/").build()),
        Mount("/general", A2aAgentServer(general_agent, url="http://0.0.0.0:8000/general/").build()),
    ]
)
