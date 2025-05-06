# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os

from occamai.api_client import AgentInstanceParamsModel, OccamClient

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agents.experimental.occam.occam import OccamAgent

if __name__ == "__main__":
    api_key = os.getenv("OCCAM_API_KEY")
    base_url = os.getenv("OCCAM_BASE_URL", "https://api.occam.ai")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    occam_client = OccamClient(api_key=api_key, base_url=base_url)
    agents_catalogue = occam_client.agents.get_agents_catalogue()
    print(f"Found {len(agents_catalogue)} agents in the catalogue.")

    # TODO: Agent selection and param building
    ...
    agent_name = "DeepSeek: R1 Distill Llama 70B"
    agent_params = AgentInstanceParamsModel()

    occam_agent = OccamAgent(
        name="occam-agent",
        occam_client=occam_client,
        agent_name=agent_name,
        agent_params=agent_params,
    )

    other_agent = ConversableAgent(name="other-agent", llm_config={"model": "gpt-4o-mini", "api_type": "openai"})
    other_agent.initiate_chat(
        recipient=occam_agent,
        message="Hello, tell me an interesting fact about the moon.",
        max_turns=2,
    )
