# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

from autogen import ConversableAgent, UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.capabilities.teachability import Teachability
from autogen.formatting_utils import colored

from ....conftest import KEY_LOC, OAI_CONFIG_LIST

# Specify the model to use. GPT-3.5 is less reliable than GPT-4 at learning from user input.
filter_dict = {"model": ["gpt-4o-mini"]}
# filter_dict = {"model": ["gpt-4-0125-preview"]}


def create_teachable_agent(reset_db=False):
    """Instantiates a teachable agent using the settings from the top of this file."""
    # Load LLM inference endpoints from an env variable or a file
    # See https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/llm-configuration-deep-dive#llm-configuration
    # and OAI_CONFIG_LIST_sample
    config_list = config_list_from_json(env_or_file=OAI_CONFIG_LIST, filter_dict=filter_dict, file_location=KEY_LOC)

    # Start by instantiating any agent that inherits from ConversableAgent.
    teachable_agent = ConversableAgent(
        name="teachable_agent",
        llm_config={"config_list": config_list, "timeout": 120, "cache_seed": None},  # Disable caching.
    )

    # Instantiate the Teachability capability. Its parameters are all optional.
    teachability = Teachability(
        verbosity=0,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
        reset_db=reset_db,
        path_to_db_dir="./tmp/interactive/teachability_db",
        recall_threshold=1.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
    )

    # Now add the Teachability capability to the agent.
    teachability.add_to_agent(teachable_agent)

    return teachable_agent


def interact_freely_with_user():
    """Starts a free-form chat between the user and a teachable agent."""
    # Create the agents.
    print(colored("\nLoading previous memory (if any) from disk.", "light_cyan"))
    teachable_agent = create_teachable_agent(reset_db=False)
    user = UserProxyAgent("user", human_input_mode="ALWAYS", code_execution_config={})

    # Start the chat.
    teachable_agent.initiate_chat(user, message="Greetings, I'm a teachable user assistant! What's on your mind today?")


if __name__ == "__main__":
    """Lets the user test a teachable agent interactively."""
    interact_freely_with_user()
