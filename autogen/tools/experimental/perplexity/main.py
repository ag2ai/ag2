# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import os

from autogen import AssistantAgent, LLMConfig, UserProxyAgent

# Import the Tool
from autogen.tools.experimental import PerplexitySearchTool

llm_config = LLMConfig(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])

# Agent for LLM tool recommendation
with llm_config:
    assistant = AssistantAgent(name="assistant")

# Agent for tool execution
user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER")

ppx_api_key = os.getenv("PERPLEXITY_API_KEY")

# Create the tool, with defaults
perplexity_search_tool = PerplexitySearchTool(ppx_api_key=ppx_api_key, max_tokens=1000)

# Register it for LLM recommendation and execution
perplexity_search_tool.register_for_llm(assistant)
perplexity_search_tool.register_for_execution(user_proxy)

result = user_proxy.initiate_chat(
    recipient=assistant,
    message="What is perplexity?",
    max_turns=2,
)
