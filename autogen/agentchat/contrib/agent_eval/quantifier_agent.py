# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from typing import Any, Optional

from ...conversable_agent import ConversableAgent


class QuantifierAgent(ConversableAgent):
    """An agent for quantifying the performance of a system using the provided criteria."""

    DEFAULT_SYSTEM_MESSAGE = """"You are a helpful assistant. You quantify the output of different tasks based on the given criteria.
    The criterion is given in a json list format where each element is a distinct criteria.
    The each element is a dictionary as follows {"name": name of the criterion, "description": criteria description , "accepted_values": possible accepted inputs for this key}
    You are going to quantify each of the crieria for a given task based on the task description.
    Return a dictionary where the keys are the criteria and the values are the assessed performance based on accepted values for each criteria.
    Return only the dictionary, no code."""

    DEFAULT_DESCRIPTION = "An AI agent for quantifing the performance of a system using the provided criteria."

    def __init__(
        self,
        name="quantifier",
        system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
        description: Optional[str] = DEFAULT_DESCRIPTION,
        **kwargs: Any,
    ):
        """Args:
        name (str): agent name.
        system_message (str): system message for the ChatCompletion inference.
            Please override this attribute if you want to reprogram the agent.
        description (str): The description of the agent.
        **kwargs (dict): Please refer to other kwargs in
            [ConversableAgent](/docs/api-reference/autogen/ConversableAgent#conversableagent).
        """
        super().__init__(name=name, system_message=system_message, description=description, **kwargs)
