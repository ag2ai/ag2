# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import asyncio
from typing import Any

from ... import OpenAIWrapper
from ...import_utils import optional_import_block, require_optional_import
from .. import Agent, ConversableAgent
from .vectordb.utils import get_logger

logger = get_logger(__name__)

with optional_import_block():
    from llama_index.core.base.llms.types import ChatMessage
    from pydantic import BaseModel, ConfigDict

    Config = ConfigDict(arbitrary_types_allowed=True)

    # Add Pydantic configuration to allow arbitrary types
    # Added to mitigate PydanticSchemaGenerationError
    BaseModel.model_config = Config

# llama-index 0.13+ removed the `AgentRunner`/`chat()` surface in favour of
# the workflow-based agents (`FunctionAgent`, `ReActAgent`, `CodeActAgent`)
# which expose `run(user_msg=..., chat_history=...)` returning an awaitable
# `WorkflowHandler` that resolves to an `AgentOutput`. Earlier versions of
# this wrapper imported `AgentRunner` and `AgentChatResponse` directly; both
# of those modules disappeared on 0.14, so we keep the imports optional and
# fall back to the new surface at runtime via `hasattr`.
with optional_import_block():
    from llama_index.core.agent.runner.base import (
        AgentRunner,  # noqa: F401 - legacy alias for the type hint  # type: ignore[no-redef,import-not-found]
    )
    from llama_index.core.chat_engine.types import (
        AgentChatResponse,  # noqa: F401 - legacy response type  # type: ignore[no-redef,import-not-found]
    )


@require_optional_import("llama_index", "neo4j")
class LLamaIndexConversableAgent(ConversableAgent):
    def __init__(
        self,
        name: str,
        llama_index_agent: Any,
        description: str | None = None,
        **kwargs: Any,
    ):
        """Args:
        name (str): agent name.
        llama_index_agent (Any): A llama-index agent instance. Both the
            legacy `llama_index.core.agent.AgentRunner` (chat/achat) and the
            new workflow-based agents introduced in 0.13 (FunctionAgent,
            ReActAgent, CodeActAgent — all exposing `run`) are accepted.
            Please override this attribute if you want to reprogram the agent.
        description (str): a short description of the agent. This description is used by other agents
            (e.g. the GroupChatManager) to decide when to call upon this agent.
        **kwargs (dict): Please refer to other kwargs in
            `ConversableAgent`.
        """
        if llama_index_agent is None:
            raise ValueError("llama_index_agent must be provided")

        if not description or description.strip() == "":
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
        messages: list[dict[str, Any]] | None = None,
        sender: Agent | None = None,
        config: OpenAIWrapper | None = None,
    ) -> tuple[bool, str | dict[str, Any] | None]:
        """Generate a reply using autogen.oai."""
        user_message, history = self._extract_message_and_history(messages=messages, sender=sender)

        # Legacy llama-index <= 0.12 AgentRunner exposes a synchronous chat
        # entry point.
        if hasattr(self._llama_index_agent, "chat"):
            chat_response = self._llama_index_agent.chat(message=user_message, chat_history=history)
            return True, chat_response.response

        # llama-index >= 0.13 workflow-based agents only expose async `run`.
        # We can't call it directly from a synchronous reply hook, so drive
        # the coroutine to completion on a fresh event loop. Callers that
        # need a non-blocking path should use `a_initiate_chat`, which
        # routes through `_a_generate_oai_reply` below.
        return True, asyncio.run(self._run_workflow_agent(user_message, history))

    async def _a_generate_oai_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: Agent | None = None,
        config: OpenAIWrapper | None = None,
    ) -> tuple[bool, str | dict[str, Any] | None]:
        """Generate a reply using autogen.oai."""
        user_message, history = self._extract_message_and_history(messages=messages, sender=sender)

        if hasattr(self._llama_index_agent, "achat"):
            chat_response = await self._llama_index_agent.achat(message=user_message, chat_history=history)
            return True, chat_response.response

        return True, await self._run_workflow_agent(user_message, history)

    async def _run_workflow_agent(self, user_message: str, history: list["ChatMessage"]) -> str:
        """Drive a llama-index 0.13+ workflow agent (FunctionAgent /
        ReActAgent / CodeActAgent) through one user turn and return the
        textual response. The agent's `run()` returns a `WorkflowHandler`
        that resolves to an `AgentOutput` with the assistant `ChatMessage`
        on `.response`.
        """
        result = await self._llama_index_agent.run(user_msg=user_message, chat_history=history)
        response = getattr(result, "response", None)
        if response is None:
            return str(result)
        # AgentOutput.response is itself a ChatMessage in 0.13+; older
        # responses may carry a plain string. Prefer the structured form
        # when available so we always emit a real assistant utterance
        # rather than the ChatMessage's repr.
        if hasattr(response, "content") and response.content is not None:
            return str(response.content)
        return str(response)

    def _extract_message_and_history(
        self, messages: list[dict[str, Any]] | None = None, sender: Agent | None = None
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
