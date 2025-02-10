# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Annotated, Any, Callable, List, Optional

from pydantic import BaseModel

from ....agentchat import ConversableAgent
from ....doc_utils import export_module
from ....tools import Depends, Tool
from ....tools.dependency_injection import on
from ..websurfer import WebSurferAgent

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
            llm_config: Annotated[dict[str, Any], Depends(on(llm_config))],
        ):
            """
            Delegate a research task to the agent.
            """

            @self.summarizer_agent.register_for_execution()
            @self.critic_agent.register_for_llm(description="Call this method to confirm the final answer.")
            def confirm_summary(answer: str, reasoning: str) -> str:
                return f"{self.ANSWER_CONFIRMED_PREFIX}" + answer + "\nReasoning: " + reasoning

            split_question_and_answer_subquestions = DeepResearchTool._get_split_question_and_answer_subquestions(
                llm_config=llm_config
            )

            self.summarizer_agent.register_for_llm(description="Split the question into subquestions and get answers.")(
                split_question_and_answer_subquestions
            )
            self.critic_agent.register_for_execution()(split_question_and_answer_subquestions)

            result = self.critic_agent.initiate_chat(
                self.summarizer_agent,
                message="Please answer the following question: " + task,
            )

            return result.summary

        super().__init__(
            name=delegate_research_task.__name__,
            description="Delegate a research task to the deep research agent.",
            func_or_tool=delegate_research_task,
        )

    @staticmethod
    def _get_split_question_and_answer_subquestions(llm_config: dict[str, Any]) -> Callable[..., Any]:
        class Subquestion(BaseModel):
            question: Annotated[str, "The original question."]
            answer: Annotated[Optional[str], "The answer to the question."] = None

            def format(self) -> str:
                return f"Question: {self.question}\n{self.answer}\n"

        class Task(BaseModel):
            question: Annotated[str, "The original question."]
            subquestions: Annotated[List[Subquestion], "The subquestions that need to be answered."]

            def format(self) -> str:
                return f"Task: {self.question}\n\n" + "\n".join(
                    "Subquestion " + str(i + 1) + ":\n" + subquestion.format()
                    for i, subquestion in enumerate(self.subquestions)
                )

        def split_question_and_answer_subquestions(
            question: Annotated[str, "The question to split and answer."],
            llm_config: Annotated[dict[str, Any], Depends(on(llm_config))],
        ) -> str:
            subquestions_answered_prefix = "Subquestions answered:"
            decomposition_agent = ConversableAgent(
                name="DecompositionAgent",
                system_message=(
                    "You are an expert at breaking down complex questions into smaller, focused subquestions.\n"
                    "Your task is to take any question provided and divide it into clear, actionable subquestions that can be individually answered.\n"
                    "Ensure the subquestions are logical, non-redundant, and cover all key aspects of the original question.\n"
                    "Avoid providing answers or interpretations—focus solely on decomposition.\n"
                    "Do not include banal, general knowledge questions\n"
                    "Do not include questions that go into unnecessary detail that is not relevant to the original question\n"
                    "Do not include question that require knowledge of the original or other subquestions to answer\n"
                ),
                llm_config=llm_config,
                is_termination_msg=lambda x: x.get("content", "")
                and x.get("content", "").startswith(subquestions_answered_prefix),
                human_input_mode="NEVER",
            )

            example_task = Task(
                question="What is the capital of France?",
                subquestions=[Subquestion(question="What is the capital of France?")],
            )
            decomposition_critic = ConversableAgent(
                name="DecompositionCritic",
                system_message=(
                    "You are a critic agent responsible for evaluating the subquestions provided by the initial analysis agent.\n"
                    "You need to confirm whether the subquestions are clear, actionable, and cover all key aspects of the original question.\n"
                    "Do not accept redundant or unnecessary subquestions, focus solely on the minimal viable subset of subqestions necessary to answer the original question. \n"
                    "Do not accept banal, general knowledge questions\n"
                    "Do not accept questions that go into unnecessary detail that is not relevant to the original question\n"
                    "Remove questions that can be answered with combinig knowledge from other questions\n"
                    "After you are satisfied with the subquestions, call the 'generate_subquestions' method to answer each subquestion.\n"
                    "This is an example of an argument that can be passed to the 'generate_subquestions' method:\n"
                    f"{{'task': {example_task.model_dump()}}}\n"
                ),
                llm_config=llm_config,
                is_termination_msg=lambda x: x.get("content", "")
                and x.get("content", "").startswith(subquestions_answered_prefix),
                human_input_mode="NEVER",
            )

            @decomposition_agent.register_for_execution()
            @decomposition_critic.register_for_llm(
                name="generate_subquestions",
                description="Generates subquestions for a task.",
            )
            def generate_subquestions(
                task: Task,
                llm_config: Annotated[dict[str, Any], Depends(on(llm_config))],
            ) -> Task:
                if not task.subquestions:
                    task.subquestions = [Subquestion(question=task.question)]

                for subquestion in task.subquestions:
                    subquestion.answer = DeepResearchTool._answer_question(subquestion.question, llm_config=llm_config)

                return f"{subquestions_answered_prefix} \n" + task.format()

            result = decomposition_critic.initiate_chat(
                decomposition_agent,
                message="Analyse and gather subqestions for the following question: " + question,
            )

            return result.summary

        return split_question_and_answer_subquestions

    @staticmethod
    def _answer_question(
        question: str,
        llm_config: dict[str, Any],
        max_web_steps: int = 30,
    ) -> str:
        class InformationCrumb(BaseModel):
            source_url: str
            source_title: str
            source_summary: str
            relevant_info: str

        class GatheredInformation(BaseModel):
            information: List[InformationCrumb]

            def format(self) -> str:
                return "Here is the gathered information: \n" + "\n".join(
                    f"URL: {info.source_url}\nTitle: {info.source_title}\nSummary: {info.source_summary}\nRelevant Information: {info.relevant_info}\n\n"
                    for info in self.information
                )

        websurfer_config = copy.deepcopy(llm_config)

        websurfer_config["config_list"][0]["response_format"] = GatheredInformation

        def is_termination_msg(x):
            return x.get("content", "") and x.get("content", "").startswith("Answer confirmed:")

        websurfer_agent = WebSurferAgent(
            llm_config=websurfer_config,
            name="WebSurferAgent",
            system_message=(
                "You are a web surfer agent responsible for gathering information from the web to provide information for answering a question\n"
                "You will be asked to find information related to the question and provide a summary of the information gathered.\n"
                "The summary should include the URL, title, summary, and relevant information for each piece of information gathered.\n"
            ),
            is_termination_msg=is_termination_msg,
            human_input_mode="NEVER",
            web_tool_kwargs={
                "agent_kwargs": {"max_steps": max_web_steps},
            },
        )

        websurfer_critic = ConversableAgent(
            name="WebSurferCritic",
            system_message=(
                "You are a critic agent responsible for evaluating the answer provided by the web surfer agent.\n"
                "You need to confirm whether the information provided by the websurfer is correct and sufficient to answer the question.\n"
                "You can ask the web surfer to provide more information or provide and confirm the answer.\n"
            ),
            llm_config=llm_config,
            is_termination_msg=is_termination_msg,
            human_input_mode="NEVER",
        )

        @websurfer_agent.register_for_execution()
        @websurfer_critic.register_for_llm(
            description="Call this method when you agree that the original question can be answered with the gathered information and provide the answer."
        )
        def confirm_answer(answer: str) -> str:
            return "Answer confirmed: " + answer

        websurfer_critic.register_for_execution()(websurfer_agent.tool)

        result = websurfer_critic.initiate_chat(
            websurfer_agent,
            message="Please find the answer to this question: " + question,
        )

        return result.summary
