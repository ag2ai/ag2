# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any, Callable
from unittest.mock import patch

import pytest

from autogen.agents.experimental.deep_research.deep_research_tool import DeepResearchTool
from autogen.import_utils import skip_on_missing_imports
from autogen.tools.dependency_injection import Depends, on

from ....conftest import Credentials


@skip_on_missing_imports(
    ["langchain_openai", "browser_use"],
    "browser-use",
)
class TestDeepResearchTool:
    def test__init__(self, mock_credentials: Credentials) -> None:
        tool = DeepResearchTool(
            llm_config=mock_credentials.llm_config,
        )

        assert isinstance(tool, DeepResearchTool)
        assert tool.name == "delegate_research_task"
        expected_schema = {
            "description": "Delegate a research task to the deep research agent.",
            "name": "delegate_research_task",
            "parameters": {
                "properties": {"task": {"description": "The tash to perform a research on.", "type": "string"}},
                "required": ["task"],
                "type": "object",
            },
        }
        assert tool.function_schema == expected_schema

    # gpt-4o-mini isn't good enough to answer this question
    @pytest.mark.openai
    def test_answer_question(self, credentials_gpt_4o: Credentials) -> None:
        result = DeepResearchTool._answer_question(
            question="Who are the founders of the AG2 framework?",
            llm_config=credentials_gpt_4o.llm_config,
        )

        assert isinstance(result, str)
        assert result.startswith("Answer confirmed:")
        result = result.lower()
        assert "wang" in result or "wu" in result

    @pytest.mark.openai
    def test_get_split_question_and_answer_subquestions(self, credentials_gpt_4o_mini: Credentials) -> None:
        split_question_and_answer_subquestions = DeepResearchTool._get_split_question_and_answer_subquestions(
            llm_config=credentials_gpt_4o_mini.llm_config,
        )

        with patch(
            "autogen.agents.experimental.deep_research.deep_research_tool.DeepResearchTool._answer_question",
            return_value="Answer confirmed: Some answer",
        ) as mock_answer_question:
            result = split_question_and_answer_subquestions(
                question="Who are the founders of the AG2 framework?",
                # When we register the function to the agents, llm_config will be injected
                llm_config=credentials_gpt_4o_mini.llm_config,
            )
        assert isinstance(result, str)
        assert result.startswith("Subquestions answered:")

        mock_answer_question.assert_called()

    @pytest.mark.openai
    def test_delegate_research_task(self, credentials_gpt_4o_mini: Credentials) -> None:
        def _get_split_question_and_answer_subquestions(llm_config: dict[str, Any]) -> Callable[..., Any]:
            def split_question_and_answer_subquestions(
                question: Annotated[str, "The question to split and answer."],
                llm_config: Annotated[dict[str, Any], Depends(on(llm_config))],
            ) -> str:
                assert llm_config == credentials_gpt_4o_mini.llm_config
                return (
                    "Subquestions answered:\n"
                    "Task: Who are the founders of the AG2 framework?\n\n"
                    "Subquestion 1:\n"
                    "Question: What is the AG2 framework?\n"
                    "Answer confirmed: AG2 (formerly AutoGen) is an open-source AgentOS for building AI agents and facilitating cooperation among multiple agents to solve tasks. AG2 provides fundamental building blocks needed to create, deploy, and manage AI agents that can work together to solve complex problems.\n\n"
                    "Subquestion 2:\n"
                    "Question: Who are the founders of the AG2 framework?\n"
                    "Answer confirmed: Chi Wang and Qingyun Wu are the founders of the AG2 framework.\n"
                )

            return split_question_and_answer_subquestions

        with patch(
            "autogen.agents.experimental.deep_research.deep_research_tool.DeepResearchTool._get_split_question_and_answer_subquestions",
            return_value=_get_split_question_and_answer_subquestions(credentials_gpt_4o_mini.llm_config),
        ):
            tool = DeepResearchTool(
                llm_config=credentials_gpt_4o_mini.llm_config,
            )
            result = tool.func(task="Who are the founders of the AG2 framework?")
        assert isinstance(result, str)
        assert result.startswith("Answer confirmed:")
        result = result.lower()
        assert "wang" in result or "wu" in result
