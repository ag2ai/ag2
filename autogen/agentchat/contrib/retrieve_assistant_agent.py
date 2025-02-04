# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import warnings
from typing import Any, Optional, Union

from ..agent import Agent
from ..assistant_agent import AssistantAgent


class RetrieveAssistantAgent(AssistantAgent):
    """(Experimental) Retrieve Assistant agent, designed to solve a task with LLM.

    RetrieveAssistantAgent is a subclass of AssistantAgent configured with a default system message.
    The default system message is designed to solve a task with LLM,
    including suggesting python code blocks and debugging.
    `human_input_mode` is default to "NEVER"
    and `code_execution_config` is default to False.
    This agent doesn't execute code by default, and expects the user to execute the code.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The RetrieveAssistantAgent is deprecated. Please use the AssistantAgent instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
        self.register_reply(Agent, RetrieveAssistantAgent._generate_retrieve_assistant_reply)

    def _generate_retrieve_assistant_reply(
        self,
        messages: Optional[list[dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> tuple[bool, Union[str, dict, None]]:
        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        if "exitcode: 0 (execution succeeded)" in message.get("content", ""):
            # Terminate the conversation when the code execution succeeds. Although sometimes even when the
            # code execution succeeds, the task is not solved, but it's hard to tell. If the human_input_mode
            # of RetrieveUserProxyAgent is "TERMINATE" or "ALWAYS", user can still continue the conversation.
            return True, "TERMINATE"
        elif (
            "UPDATE CONTEXT" in message.get("content", "")[-20:].upper()
            or "UPDATE CONTEXT" in message.get("content", "")[:20].upper()
        ):
            return True, "UPDATE CONTEXT"
        else:
            return False, None
