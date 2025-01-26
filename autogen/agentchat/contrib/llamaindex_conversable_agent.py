# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from typing import Optional, Union

from ... import OpenAIWrapper
from ...import_utils import optional_import_block, require_optional_import
from .. import Agent, ConversableAgent
from .vectordb.utils import get_logger

logger = get_logger(__name__)

with optional_import_block() as result:
    from llama_index.core.agent.runner.base import AgentRunner
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.chat_engine.types import AgentChatResponse
    from pydantic import BaseModel
    from pydantic import __version__ as pydantic_version

if result.is_successful:
    # let's Avoid: AttributeError: type object 'Config' has no attribute 'copy'
    # check for v1 like in autogen/_pydantic.py
    is_pydantic_v1 = pydantic_version.startswith("1.")
    if not is_pydantic_v1:
        from pydantic import ConfigDict

        Config = ConfigDict(arbitrary_types_allowed=True)
    else:

        class Config:
            arbitrary_types_allowed = True

    # Add Pydantic configuration to allow arbitrary types
    # Added to mitigate PydanticSchemaGenerationError
    BaseModel.model_config = Config


@require_optional_import("llama_index", "neo4j")
class LLamaIndexConversableAgent(ConversableAgent):
    def __init__(
        self,
        name: str,
        llama_index_agent: "AgentRunner",
        description: Optional[str] = None,
        **kwargs,
    ):
        """Args:
        name (str): agent name.
        llama_index_agent (AgentRunner): llama index agent.
            Please override this attribute if you want to reprogram the agent.
        description (str): a short description of the agent. This description is used by other agents
            (e.g. the GroupChatManager) to decide when to call upon this agent.
        **kwargs (dict): Please refer to other kwargs in
            [ConversableAgent](../conversable_agent#init).
        """
        if llama_index_agent is None:
            raise ValueError("llama_index_agent must be provided")

        if description is None or description.isspace():
            raise ValueError("description must be provided")

        super().__init__(
            name,
            description=description,
            **kwargs,
        )

        self._llama_index_agent = llama_index_agent

        # Override the `generate_oai_reply`
        self.replace_reply_func(ConversableAgent.generate_oai_reply, LLamaIndexConversableAgent._generate_oai_reply)

        self.replace_reply_func(ConversableAgent.a_generate_oai_reply, LLamaIndexConversableAgent._a_generate_oai_reply)

    def _generate_oai_reply(
        self,
        messages: Optional[list[dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> tuple[bool, Union[str, dict, None]]:
        """Generate a reply using autogen.oai."""
        user_message, history = self._extract_message_and_history(messages=messages, sender=sender)

        chat_response: AgentChatResponse = self._llama_index_agent.chat(message=user_message, chat_history=history)

        extracted_response = chat_response.response

        return (True, extracted_response)

    async def _a_generate_oai_reply(
        self,
        messages: Optional[list[dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> tuple[bool, Union[str, dict, None]]:
        """Generate a reply using autogen.oai."""
        user_message, history = self._extract_message_and_history(messages=messages, sender=sender)

        chat_response: AgentChatResponse = await self._llama_index_agent.achat(
            message=user_message, chat_history=history
        )

        extracted_response = chat_response.response

        return (True, extracted_response)

    def _extract_message_and_history(
        self, messages: Optional[list[dict]] = None, sender: Optional[Agent] = None
    ) -> tuple[str, list["ChatMessage"]]:
        """Extract the message and history from the messages."""
        if not messages:
            messages = self._oai_messages[sender]

        if not messages:
            return "", []

        message = messages[-1].get("content", "")

        history = messages[:-1]
        history_messages: list[ChatMessage] = []
        for history_message in history:
            content = history_message.get("content", "")
            role = history_message.get("role", "user")
            if role and (role == "user" or role == "assistant"):
                history_messages.append(ChatMessage(content=content, role=role, additional_kwargs={}))
        return message, history_messages
