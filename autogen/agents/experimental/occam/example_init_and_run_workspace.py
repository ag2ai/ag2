import os

from occam_agent_params.instance_params_models import MultiAgentWorkspaceParamsModel, ChatMode
from occamai.api_client import OccamClient

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agents.experimental.occam.occam import OccamAgent



if __name__ == "__main__":
    api_key = os.getenv("OCCAM_API_KEY")
    base_url = os.getenv("OCCAM_BASE_URL", "https://api.occam.ai")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    occam_client = OccamClient(api_key=api_key, base_url=base_url)
    agents_catalogue = occam_client.agents.get_agents_catalogue()
    print(f"Found {len(agents_catalogue)} agents in the catalogue.")

    agent_params = MultiAgentWorkspaceParamsModel(
        agents={
            "OpenAI: GPT-4": None,
            "mo": None,
        },
        max_chat_steps=30,
        agent_selection_rule=ChatMode.GROUP_CHAT
    )
    workspace_agent_key = "Multi-agent Workspaces"

    # Initiator agent.
    first_agent = ConversableAgent(name="first-agent", llm_config={"model": "gpt-4o-mini", "api_type": "openai"})
    # Occam workspace agent
    occam_agent = OccamAgent(
        name="occam-agent",
        occam_client=occam_client,
        agent_name=workspace_agent_key,
        agent_params=agent_params,
    )
    # Summariser agent.
    third_agent = ConversableAgent(name="third-agent", llm_config={"model": "gpt-4o-mini", "api_type": "openai"})

    # Kick off a flow from first_agent -> occam_agent -> third_agent
    first_agent.initiate_chats(
        [
            {
                "recipient": occam_agent,
                "message": "Hello, Start a chat about a random scientific topic.",
                "max_turns": 2,
            },
            {
                "recipient": third_agent,
                "message": "Summarise the result of the chat in no more than 100 words.",
                "max_turns": 2,
            },
        ]
    )
