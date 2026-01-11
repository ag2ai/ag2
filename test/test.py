import os
import pprint
from typing import Annotated

from dotenv import load_dotenv

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import run_group_chat_iter
from autogen.agentchat.group import AgentNameTarget, ReplyResult
from autogen.agentchat.group.patterns import AutoPattern
from autogen.events.agent_events import GroupChatRunChatEvent, InputRequestEvent, ToolResponseEvent

load_dotenv()


# Define the query classification tool
def classify_query(query: Annotated[str, "The user query to classify"]) -> ReplyResult:
    """Classify a user query as technical or general."""
    technical_keywords = ["error", "bug", "broken", "crash", "not working", "shutting down"]

    if any(keyword in query.lower() for keyword in technical_keywords):
        return ReplyResult(
            message="This appears to be a technical issue. Routing to technical support...",
            target=AgentNameTarget("tech_agent"),
        )
    else:
        return ReplyResult(
            message="This appears to be a general question. Routing to general support...",
            target=AgentNameTarget("general_agent"),
        )


# Create the agents
llm_config = LLMConfig(
    config_list={"api_type": "openai", "model": "gpt-5-nano", "api_key": os.getenv("OPENAI_API_KEY")}
)

triage_agent = ConversableAgent(
    name="triage_agent",
    system_message="""You are a triage agent. For each user query,
    identify whether it is a technical issue or a general question.
    Use the classify_query tool to categorize queries and route them appropriately.
    Do not provide suggestions or answers, only route the query.""",
    functions=[classify_query],
    llm_config=llm_config,
)

tech_agent = ConversableAgent(
    name="tech_agent",
    system_message="""You solve technical problems like software bugs
    and hardware issues. After analyzing the problem, use the provide_technical_solution
    tool to format your response consistently.""",
    llm_config=llm_config,
)

general_agent = ConversableAgent(
    name="general_agent", system_message="You handle general, non-technical support questions.", llm_config=llm_config
)

# User agent
user = ConversableAgent(name="user", human_input_mode="ALWAYS")

# Set up the conversation pattern
pattern = AutoPattern(
    initial_agent=triage_agent,
    agents=[triage_agent, tech_agent, general_agent],
    user_agent=user,
    group_manager_args={"llm_config": llm_config},
)

import os
from typing import Annotated

from dotenv import load_dotenv

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat.group import ReplyResult
from autogen.agentchat.group.patterns import AutoPattern

load_dotenv()


# Define the query classification tool
def classify_query(query: Annotated[str, "The user query to classify"]) -> ReplyResult:
    """Classify a user query as technical or general."""
    technical_keywords = ["error", "bug", "broken", "crash", "not working", "shutting down"]

    if any(keyword in query.lower() for keyword in technical_keywords):
        return ReplyResult(
            message="This appears to be a technical issue. Routing to technical support...",
            target=AgentNameTarget("tech_agent"),
        )
    else:
        return ReplyResult(
            message="This appears to be a general question. Routing to general support...",
            target=AgentNameTarget("general_agent"),
        )


# Create the agents
llm_config = LLMConfig(
    config_list={"api_type": "openai", "model": "gpt-5-nano", "api_key": os.getenv("OPENAI_API_KEY")}
)

triage_agent = ConversableAgent(
    name="triage_agent",
    system_message="""You are a triage agent. For each user query,
    identify whether it is a technical issue or a general question.
    Use the classify_query tool to categorize queries and route them appropriately.
    Do not provide suggestions or answers, only route the query.""",
    functions=[classify_query],
    llm_config=llm_config,
)

tech_agent = ConversableAgent(
    name="tech_agent",
    system_message="""You solve technical problems like software bugs
    and hardware issues. After analyzing the problem, use the provide_technical_solution
    tool to format your response consistently.""",
    llm_config=llm_config,
)

general_agent = ConversableAgent(
    name="general_agent", system_message="You handle general, non-technical support questions.", llm_config=llm_config
)

# User agent
user = ConversableAgent(name="user", human_input_mode="ALWAYS")

# Set up the conversation pattern
pattern = AutoPattern(
    initial_agent=triage_agent,
    agents=[triage_agent, tech_agent, general_agent],
    user_agent=user,
    group_manager_args={"llm_config": llm_config},
)

# Run the chat
for event in run_group_chat_iter(
    pattern=pattern,
    messages="My laptop keeps shutting down randomly. Can you help?",
    max_rounds=20,
):
    if isinstance(event, InputRequestEvent):
        user_input = input("  Input requested: ")
        event.content.respond(user_input)
        continue

    if isinstance(event, GroupChatRunChatEvent):
        print(f"\n=== {event.content.speaker}'s turn ===")
        pprint.pprint(event.content)
        print("-" * 80)

    if isinstance(event, ToolResponseEvent):
        print(f"\n=== {event.content} tool response ===")
        print("-" * 80)

    if hasattr(event, "content") and hasattr(event.content, "content"):
        content = str(event.content.content)
        pprint.pprint(content)
        print("-" * 80)
    pass
