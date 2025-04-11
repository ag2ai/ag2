# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Optional, Union

from ....agentchat import Agent, ConversableAgent
from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import

with optional_import_block():
    from occam_core.agents.model import AgentIOModel, OccamLLMMessage
    from occam_core.api.util import AgentResponseModel
    from occam_core.chat.model import MultiAgentWorkspaceCoreMessageModel
    from occam_agent_params.instance_params_models import MultiAgentWorkspaceParamsModel
    from occamai.api_client import AgentInstanceParamsModel, OccamClient, AgentsApi, WorkspacesApi

__all__ = ["OccamAgent"]


@require_optional_import(["occam_core.agents.model", "occamai.api_client"], "occam")
@export_module("autogen.agents")
class OccamAgent(ConversableAgent):
    _api_client: Union["WorkspacesApi", "AgentsApi"]  # type: ignore[no-any-unimported]

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

        agent_run_response: AgentResponseModel = self._api_client.run_agent(  # type: ignore[no-any-unimported]
            agent_instance_id=self.occam_agent_instance_id,
            sync=True,
            agent_input_model=agent_io_model,
        )  # type: ignore[no-any-unimported]
        print(f"Agent run status: {agent_run_response.status}")

        # Send a message to the user to notify them it's waiting. Create a message as needed
        # iostream = IOStream.get_default()
        # iostream.send(
        #     TerminationAndHumanReplyMessage(no_human_input_msg="Dummy message", sender=sender, recipient=self)
        # )

        # True indicates final reply, string goes back into the chat's messages.
        merged_output_message = ""
        if isinstance(self._api_client, WorkspacesApi):
            for msg in agent_run_response.chat_messages:
                msg: MultiAgentWorkspaceCoreMessageModel = msg
                merged_output_message += f"\n{msg.full_content}"
        else:
            merged_output_message = "\n".join([m.content for m in agent_run_response.chat_messages])
        return True, merged_output_message

    def __init__(  # type: ignore[no-any-unimported]
        self,
        occam_client: "OccamClient",
        agent_name: str,
        agent_params: Optional["AgentInstanceParamsModel"] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the Occam Agent.

        Args:
            client (OccamClient): The Occam client.
            agent_name (str): The agent name.
            agent_params (Optional[AgentInstanceParamsModel]): The agent instance parameters.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """

        super().__init__(*args, **kwargs)

        if not agent_params:
            agent_params = AgentInstanceParamsModel()

        if isinstance(agent_params, MultiAgentWorkspaceParamsModel):
            self._api_client: WorkspacesApi = occam_client.workspaces
        else:
            self._api_client: AgentsApi = occam_client.agents

        # Initialise agent.
        occam_agent_instance = self._api_client.instantiate_agent(
            agent_name=agent_name, agent_params=agent_params
        )
        self.occam_agent_instance_id = occam_agent_instance.instance_id
        print(f"Created Occam Agent instance: {self.occam_agent_instance_id}")
        if occam_agent_instance.session_id:
            self.occam_session_id = occam_agent_instance.session_id
            print(f"Created Occam Session with ID: {self.occam_session_id}")

        # POPULATE `system_message` and, optionally, `description` for use in group chat auto-speaker selection
        # This should contain a description of what the agent does.
        self.update_system_message("POPULATE THIS - DON'T LEAVE IT LIKE THIS!")

        self.register_reply(
            trigger=[Agent, None],
            reply_func=OccamAgent.occam_reply_func,
            remove_other_reply_funcs=True,
        )
