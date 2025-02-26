# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from autogen.agents.experimental import DeepResearchAgent
from autogen.import_utils import skip_on_missing_imports
from autogen.tools.experimental import DeepResearchTool

from ....conftest import Credentials


@skip_on_missing_imports(
    ["langchain_openai", "browser_use"],
    "browser-use",
)
class TestDeepResearchAgent:
    def test__init__(self, mock_credentials: Credentials) -> None:
        agent = DeepResearchAgent(
            name="deep_research_agent",
            llm_config=mock_credentials.llm_config,
        )

        assert isinstance(agent, DeepResearchAgent)
        assert agent.name == "deep_research_agent"
        expected_tools = [
            {
                "function": {
                    "description": "Delegate a research task to the deep research agent.",
                    "name": "delegate_research_task",
                    "parameters": {
                        "properties": {"task": {"description": "The task to perform a research on.", "type": "string"}},
                        "required": ["task"],
                        "type": "object",
                    },
                },
                "type": "function",
            }
        ]
        assert isinstance(agent.llm_config, dict), "llm_config should be a dictionary"
        assert agent.llm_config["tools"] == expected_tools

    @pytest.mark.skip(reason="The test takes too long to run.")
    @pytest.mark.openai
    def test_end2end(self, credentials_gpt_4o: Credentials) -> None:
        def is_termination_msg(message: Any) -> bool:
            content = message.get("content", "")
            if not content:
                return False
            return bool(content.startswith(DeepResearchTool.ANSWER_CONFIRMED_PREFIX))

        agent = DeepResearchAgent(
            name="deep_research_agent",
            llm_config=credentials_gpt_4o.llm_config,
            is_termination_msg=is_termination_msg,
            human_input_mode="NEVER",
        )

        result = agent.run(
            message="Who are the founders of the AG2 framework?",
            user_input=False,
            tools=agent.tools[0],
        ).summary

        assert isinstance(result, str)
        assert result.startswith("Answer confirmed:")
        result = result.lower()
        assert "wang" in result or "wu" in result
