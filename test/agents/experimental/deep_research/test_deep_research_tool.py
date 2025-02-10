# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from autogen.agents.experimental.deep_research.deep_research_tool import DeepResearchTool

from ....conftest import Credentials


class TestDeepResearchTool:
    def test___init__(self, mock_credentials: Credentials) -> None:
        tool = DeepResearchTool(
            llm_config=mock_credentials.llm_config,
        )

        assert isinstance(tool, DeepResearchTool)
        assert tool.name == "delegate_research_task"

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
