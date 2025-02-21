# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from typing import Optional, Union

from .... import Agent, ConversableAgent, OpenAIWrapper
from ....doc_utils import export_module
from ....io.base import IOStream
from ....messages.agent_messages import (
    TerminationAndHumanReplyMessage,
)

__all__ = ["OccamAgent"]


@export_module("autogen.agents")
class OccamAgent(ConversableAgent):
    """Occam.ai Agent"""

    def my_reply_func(
        self,
        messages: Optional[list[dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> tuple[bool, Union[str, dict, None]]:
        # Do all your processing in here.

        # Loop and wait for status to be completed or failed, etc.

        # Send a message to the user to notify them it's waiting. Create a message as needed
        iostream = IOStream.get_default()
        iostream.send(
            TerminationAndHumanReplyMessage(no_human_input_msg="Dummy message", sender=sender, recipient=self)
        )

        # True indicates final reply, string goes back into the chat's messages.
        return True, "Message from Occam Agent to go back into workflow."

    def __init__(
        self,
        agent_params,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the Occam Agent.

        Args:
            llm_config (dict[str, Any]): The LLM configuration.
        """

        super().__init__(*args, **kwargs)

        self.register_reply(
            trigger=[Agent, None],
            reply_func=self.my_reply_func,
            remove_other_reply_funcs=True,
        )
