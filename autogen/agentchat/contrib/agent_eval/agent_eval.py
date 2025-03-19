# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from typing import Any, Literal, Optional, Union

from .... import GroupChat, GroupChatManager, UserProxyAgent
from ....llm_config import LLMConfig
from .criterion import Criterion
from .critic_agent import CriticAgent
from .quantifier_agent import QuantifierAgent
from .subcritic_agent import SubCriticAgent
from .task import Task


def generate_criteria(
    llm_config: Optional[Union[LLMConfig, dict[str, Any], Literal[False]]] = None,
    task: Task = None,
    additional_instructions: str = "",
    max_round=2,
    use_subcritic: bool = False,
):
    """Creates a list of criteria for evaluating the utility of a given task.

    Args:
        llm_config (LLMConfig or dict or bool): llm inference configuration.
        task (Task): The task to evaluate.
        additional_instructions (str): Additional instructions for the criteria agent.
        max_round (int): The maximum number of rounds to run the conversation.
        use_subcritic (bool): Whether to use the subcritic agent to generate subcriteria.

    Returns:
        list: A list of Criterion objects for evaluating the utility of the given task.
    """
    critic = CriticAgent(
        system_message=CriticAgent.DEFAULT_SYSTEM_MESSAGE + "\n" + additional_instructions,
        llm_config=llm_config,
    )

    critic_user = UserProxyAgent(
        name="critic_user",
        max_consecutive_auto_reply=0,  # terminate without auto-reply
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
    )

    agents = [critic_user, critic]

    if use_subcritic:
        subcritic = SubCriticAgent(
            llm_config=llm_config,
        )
        agents.append(subcritic)

    groupchat = GroupChat(agents=agents, messages=[], max_round=max_round, speaker_selection_method="round_robin")
    critic_manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    critic_user.initiate_chat(critic_manager, message=task.get_sys_message())
    criteria = critic_user.last_message()
    content = criteria["content"]
    # need to strip out any extra code around the returned json
    content = content[content.find("[") : content.rfind("]") + 1]
    criteria = Criterion.parse_json_str(content)
    return criteria


def quantify_criteria(
    llm_config: Optional[Union[LLMConfig, dict[str, Any], Literal[False]]] = None,
    criteria: list[Criterion] = None,
    task: Task = None,
    test_case: str = "",
    ground_truth: str = "",
):
    """Quantifies the performance of a system using the provided criteria.

    Args:
        llm_config (LLMConfig or dict or bool): llm inference configuration.
        criteria ([Criterion]): A list of criteria for evaluating the utility of a given task.
        task (Task): The task to evaluate.
        test_case (str): The test case to evaluate.
        ground_truth (str): The ground truth for the test case.

    Returns:
        dict: A dictionary where the keys are the criteria and the values are the assessed performance based on accepted values for each criteria.
    """
    quantifier = QuantifierAgent(
        llm_config=llm_config,
    )

    quantifier_user = UserProxyAgent(
        name="quantifier_user",
        max_consecutive_auto_reply=0,  # terminate without auto-reply
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
    )

    quantifier_user.initiate_chat(
        quantifier,
        message=task.get_sys_message()
        + "Evaluation dictionary: "
        + Criterion.write_json(criteria)
        + "actual test case to evaluate: "
        + test_case,
    )
    quantified_results = quantifier_user.last_message()
    return {"actual_success": ground_truth, "estimated_performance": quantified_results["content"]}
