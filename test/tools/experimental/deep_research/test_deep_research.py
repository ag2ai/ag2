# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any, Callable
from unittest.mock import patch

from autogen.agentchat import AssistantAgent
from autogen.import_utils import run_for_optional_imports
from autogen.tools.dependency_injection import Depends, on
from autogen.tools.experimental import DeepResearchTool

from ....conftest import Credentials


@run_for_optional_imports(
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
                "properties": {"task": {"description": "The task to perform a research on.", "type": "string"}},
                "required": ["task"],
                "type": "object",
            },
        }
        assert tool.function_schema == expected_schema

    def test_get_generate_subquestions(self, mock_credentials: Credentials) -> None:
        generate_subquestions = DeepResearchTool._get_generate_subquestions(
            llm_config=mock_credentials.llm_config,
            max_web_steps=30,
        )

        assistant = AssistantAgent(
            name="assistant",
            llm_config=mock_credentials.llm_config,
        )
        assistant.register_for_llm(description="Generate subquestions for a given question.")(generate_subquestions)
        expected_tools = [
            {
                "type": "function",
                "function": {
                    "description": "Generate subquestions for a given question.",
                    "name": "generate_subquestions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "$defs": {
                                    "Subquestion": {
                                        "properties": {
                                            "question": {
                                                "description": "The original question.",
                                                "title": "Question",
                                                "type": "string",
                                            }
                                        },
                                        "required": ["question"],
                                        "title": "Subquestion",
                                        "type": "object",
                                    }
                                },
                                "properties": {
                                    "question": {
                                        "description": "The original question.",
                                        "title": "Question",
                                        "type": "string",
                                    },
                                    "subquestions": {
                                        "description": "The subquestions that need to be answered.",
                                        "items": {"$ref": "#/$defs/Subquestion"},
                                        "title": "Subquestions",
                                        "type": "array",
                                    },
                                },
                                "required": ["question", "subquestions"],
                                "title": "Task",
                                "type": "object",
                                "description": "task",
                            }
                        },
                        "required": ["task"],
                    },
                },
            }
        ]
        assert assistant.llm_config["tools"] == expected_tools, assistant.llm_config["tools"]  # type: ignore[index]

    # gpt-4o-mini isn't good enough to answer this question
    @run_for_optional_imports("openai", "openai")
    def test_answer_question(self, credentials_gpt_4o: Credentials) -> None:
        result = DeepResearchTool._answer_question(
            question="Who are the founders of the AG2 framework?",
            llm_config=credentials_gpt_4o.llm_config,
            max_web_steps=30,
        )

        assert isinstance(result, str)
        assert result.startswith("Answer confirmed:")
        result = result.lower()
        assert "wang" in result or "wu" in result

    @run_for_optional_imports("openai", "openai")
    def test_get_split_question_and_answer_subquestions(self, credentials_gpt_4o_mini: Credentials) -> None:
        max_web_steps = 30
        split_question_and_answer_subquestions = DeepResearchTool._get_split_question_and_answer_subquestions(
            llm_config=credentials_gpt_4o_mini.llm_config,
            max_web_steps=max_web_steps,
        )

        with patch(
            "autogen.agents.experimental.deep_research.deep_research.DeepResearchTool._answer_question",
            return_value="Answer confirmed: Some answer",
        ) as mock_answer_question:
            result = split_question_and_answer_subquestions(
                question="Who are the founders of the AG2 framework?",
                # When we register the function to the agents, llm_config will be injected
                llm_config=credentials_gpt_4o_mini.llm_config,
                max_web_steps=max_web_steps,
            )
        assert isinstance(result, str)
        assert result.startswith("Subquestions answered:")

        mock_answer_question.assert_called()

    @run_for_optional_imports("openai", "openai")
    def test_delegate_research_task(self, credentials_gpt_4o_mini: Credentials) -> None:
        expected_max_web_steps = 30

        def _get_split_question_and_answer_subquestions(
            llm_config: dict[str, Any], max_web_steps: int
        ) -> Callable[..., Any]:
            def split_question_and_answer_subquestions(
                question: Annotated[str, "The question to split and answer."],
                llm_config: Annotated[dict[str, Any], Depends(on(llm_config))],
                max_web_steps: Annotated[int, Depends(on(max_web_steps))],
            ) -> str:
                assert llm_config == credentials_gpt_4o_mini.llm_config
                assert max_web_steps == expected_max_web_steps
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
            "autogen.agents.experimental.deep_research.deep_research.DeepResearchTool._get_split_question_and_answer_subquestions",
            return_value=_get_split_question_and_answer_subquestions(
                credentials_gpt_4o_mini.llm_config, max_web_steps=expected_max_web_steps
            ),
        ):
            tool = DeepResearchTool(
                llm_config=credentials_gpt_4o_mini.llm_config,
            )
            # The second task is for testing if the chat history was preserved
            for task in ["Who are the founders of the AG2 framework?", "Can you please repeat the answer?"]:
                result = tool.func(task=task)
                assert isinstance(result, str)
                assert result.startswith("Answer confirmed:")
                result = result.lower()
                assert "wang" in result or "wu" in result
