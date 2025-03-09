# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

from occam_core.agents.model import AgentIOModel, OccamLLMMessage
from occamai.api_client import AgentInstanceParamsModel, AgentRunDetail, OccamClient

from autogen.agentchat import Agent, ConversableAgent
from autogen.doc_utils import export_module

__all__ = ["OccamAgent"]


@export_module("autogen.agents")
class OccamAgent(ConversableAgent):
    client: OccamClient

    def _convert_ag_messages_to_agent_io_model(self, messages: list[dict[str, Any]]) -> AgentIOModel:  # type: ignore[no-any-unimported]
        return AgentIOModel(
            chat_messages=[OccamLLMMessage(role=message["role"], content=message["content"]) for message in messages]
        )

    def occam_reply_func(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> tuple[bool, Optional[Union[str, dict[str, Any]]]]:
        """
        TODO:
        - Support path of resume, can do by ping before running the agent.
        - Some errors should be fashioned as communication, not raise errors, e.g. out of quota.
        Auto-resume considerations:
        * Auto-resume when topped up after an auto-pause, we don't have control over the workflow that the agent is running within.
        * Don't auto-resume and rely on user to always resume.
        """
        if messages is None:
            messages = []

        # Convert messages to AgentIOModel
        agent_io_model = self._convert_ag_messages_to_agent_io_model(messages)

        agent_run_detail: AgentRunDetail = self.occam_client.agents.run_agent(
            agent_instance_id=self.occam_agent_instance_id,
            sync=True,
            agent_input_model=agent_io_model,
        )  # type: ignore[no-any-unimported]
        print(f"Agent run status: {agent_run_detail.status}")
        print(f"Agent run result: {agent_run_detail.result}")

        agent_output = agent_run_detail.result

        # Send a message to the user to notify them it's waiting. Create a message as needed
        # iostream = IOStream.get_default()
        # iostream.send(
        #     TerminationAndHumanReplyMessage(no_human_input_msg="Dummy message", sender=sender, recipient=self)
        # )

        # True indicates final reply, string goes back into the chat's messages.
        return True, "".join([m.content for m in agent_output.chat_messages])

    def __init__(
        self,
        client: OccamClient,
        agent_name: str,
        agent_params: Optional[AgentInstanceParamsModel] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the Occam Agent.

        Args:
            llm_config (dict[str, Any]): The LLM configuration.
        """

        super().__init__(*args, **kwargs)

        self.occam_client = client
        if not agent_params:
            agent_params = AgentInstanceParamsModel()

        # Initialise agent.
        occam_agent_instance = self.occam_client.agents.instantiate_agent(
            agent_name=agent_name, agent_params=agent_params
        )
        self.occam_agent_instance_id = occam_agent_instance.agent_instance_id
        print(f"Created Occam Agent instance: {self.occam_agent_instance_id}")

        self.register_reply(
            trigger=[Agent, None],
            reply_func=OccamAgent.occam_reply_func,
            remove_other_reply_funcs=True,
        )
