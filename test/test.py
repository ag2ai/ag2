# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os

from autogen import AssistantAgent, ConversableAgent, LLMConfig, UserProxyAgent
from dotenv import load_dotenv

load_dotenv()

def main():
    """Example using shell tool with Responses API to refactor calculator code."""
    
    # Create LLM config with Responses API and shell tool
    llm_config = LLMConfig(
        config_list={
            "api_type": "responses",
            "model": "gpt-5.1",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "built_in_tools": ["shell"],
        },
    )

    # Create the coding agent
    coding_agent = ConversableAgent(
        name="CodingAgent",
        system_message="""You are a helpful coding assistant. 
        You can use shell commands to inspect files, run tests, and refactor code.
        When you need to execute shell commands, use the shell tool.
        The local shell environment is on Mac/Linux.
        Keep your responses concise and include command output when helpful.
        """,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # Initiate the chat to refactor calculator code
    result = coding_agent.initiate_chat(
        recipient=coding_agent,
        message="""
        use shell tool to create a new folder calculator and create a file calculator.py in it with the following content:
        """,
        max_turns=2,
    )
    
    print("\n" + "="*80)
    print("Chat Summary:")
    print("="*80)
    print(result.summary)
    print("\n" + "="*80)


if __name__ == "__main__":
    main()