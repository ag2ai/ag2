# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os

from dotenv import load_dotenv

from autogen import ConversableAgent, LLMConfig

load_dotenv()


llm_config = LLMConfig(
    config_list={
        "api_type": "responses",
        "model": "gpt-5.1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "built_in_tools": ["shell"],
    },
)

# Create the assistant agent
assistant = ConversableAgent(
    name="Assistant",
    system_message="""You are a helpful assistant with access to shell commands.
    You can use the shell tool to execute commands and interact with the filesystem.
    The local shell environment is on Mac/Linux.
    Keep your responses concise and include command output when helpful.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
)


# Example 1: Find the largest PDF and show processes
result = assistant.initiate_chat(
    recipient=assistant,
    message="""
    Please help me with the following tasks:
    1. use ls command to list the files in the current directory
    """,
    max_turns=2,
)
