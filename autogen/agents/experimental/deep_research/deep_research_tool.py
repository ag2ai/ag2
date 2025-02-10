# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any

from ....agentchat import ConversableAgent
from ....doc_utils import export_module
from ....tools import Tool

__all__ = ["DeepResearchTool"]


@export_module("autogen.tools.experimental")
class DeepResearchTool(Tool):
    ANSWER_CONFIRMED_PREFIX = "Answer confirmed:"

    def __init__(
        self,
        llm_config: dict[str, Any],
    ):
        self.llm_config = llm_config

        self.summarizer_agent = ConversableAgent(
            name="SummarizerAgent",
            system_message=(
                "You are an agent with a task of answering the question provided by the user."
                "First you need to split the question into subquestions by calling the 'split_question_and_answer_subquestions' method."
                "Then you need to sintesize the answers the original question by combining the answers to the subquestions."
            ),
            is_termination_msg=lambda x: x.get("content", "")
            and x.get("content", "").startswith(self.ANSWER_CONFIRMED_PREFIX),
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

        self.critic_agent = ConversableAgent(
            name="CriticAgent",
            system_message=(
                "You are a critic agent responsible for evaluating the answer provided by the summarizer agent.\n"
                "Your task is to assess the quality of the answer based on its coherence, relevance, and completeness.\n"
                "Provide constructive feedback on how the answer can be improved.\n"
                "If the answer is satisfactory, call the 'confirm_answer' method to end the task.\n"
            ),
            is_termination_msg=lambda x: x.get("content", "")
            and x.get("content", "").startswith(self.ANSWER_CONFIRMED_PREFIX),
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

        def delegate_research_task(
            task: Annotated[str, "The tash to perform a research on."],
        ):
            """
            Delegate a research task to the agent.
            """

            @self.summarizer_agent.register_for_execution()
            @self.critic_agent.register_for_llm(description="Call this method to confirm the final answer.")
            def confirm_summary(answer: str, reasoning: str) -> str:
                return f"{self.ANSWER_CONFIRMED_PREFIX}" + answer + "\nReasoning: " + reasoning

            # summarizer_agent.register_for_llm(description="Split the question into subquestions and get answers.")(
            #     split_question_and_answer_subquestions
            # )
            # critic_agent.register_for_execution()(split_question_and_answer_subquestions)

            result = self.critic_agent.initiate_chat(
                self.summarizer_agent,
                message="How many more blocks (also denoted as layers) in BERT base encoder than the encoder from the architecture proposed in Attention is All You Need?????",
            )

            result.summary

        super().__init__(
            name=delegate_research_task.__name__,
            description="Delegate a research task to the deep research agent.",
            func_or_tool=delegate_research_task,
        )
