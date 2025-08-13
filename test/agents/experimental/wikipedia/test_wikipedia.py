# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.agents.experimental import WikipediaAgent
from autogen.import_utils import run_for_optional_imports
from autogen.llm_config import LLMConfig
from autogen.tools.experimental import WikipediaPageLoadTool, WikipediaQueryRunTool

from ....conftest import Credentials


@run_for_optional_imports("wikipediaapi", "wikipedia")
class TestWikipediaAgent:
    def test_init(self, mock_credentials: Credentials) -> None:
        """
        Initialization should set name, llm_config, default system_message, and register tools.
        """
        agent = WikipediaAgent(
            name="wiki_agent",
            llm_config=mock_credentials.llm_config,
        )

        # Agent instance and attributes
        assert isinstance(agent, WikipediaAgent)
        assert agent.name == "wiki_agent"
        assert isinstance(agent.llm_config, (dict, LLMConfig)), "llm_config should be a dict or LLMConfig"

        # Tools registered for LLM usage
        tool_names = {tool.name for tool in agent.tools}
        expected_names = {
            WikipediaQueryRunTool().name,
            WikipediaPageLoadTool().name,
        }
        assert tool_names == expected_names, f"Expected tools {expected_names}, got {tool_names}"

        # llm_config should include these tool definitions
        tools_def = agent.llm_config.get("tools")
        assert isinstance(tools_def, list), "llm_config['tools'] should be a list"
        names_in_llm = {entry.get("function", {}).get("name") for entry in tools_def}
        assert names_in_llm == expected_names, f"Expected llm_config tools names {expected_names}, got {names_in_llm}"

        # Default system message
        assert agent.system_message == WikipediaAgent.DEFAULT_SYSTEM_MESSAGE

    def test_format_instructions(self, mock_credentials: Credentials) -> None:
        """
        Providing format_instructions should append to the default system_message.
        """
        fmt = "Answer with bullet points."
        agent = WikipediaAgent(
            name="wiki_agent",
            llm_config=mock_credentials.llm_config,
            format_instructions=fmt,
        )

        expected = WikipediaAgent.DEFAULT_SYSTEM_MESSAGE + "\n\nFollow this format:\n\n" + fmt
        assert agent.system_message == expected

    def test_extra_kwargs_pass_through(self, mock_credentials: Credentials) -> None:
        """
        Extra kwargs should be forwarded to the base ConversableAgent.
        """
        agent = WikipediaAgent(
            system_message="my system message",
            name="wiki_agent",
            llm_config=mock_credentials.llm_config,
            description="you are an agent",
        )
        assert agent.description == "you are an agent"
