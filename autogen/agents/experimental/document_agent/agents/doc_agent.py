# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Optional

from autogen import ConversableAgent

from .....doc_utils import export_module
from .....llm_config import LLMConfig
from ..core.base_interfaces import RAGQueryEngine
from ..core.config import DocAgentConfig

__all__ = ["DocAgent"]

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE = """
You are a document query agent.
You answer questions based on documents that have been previously ingested into the vector database.
You can only answer questions about documents that are in the database.
"""


@export_module("autogen.agents.experimental.document_agent")
class DocAgent:
    """Simplified DocAgent that only handles queries against pre-ingested documents.

    This agent follows the pattern: user query -> query vector db -> answer
    Document ingestion is handled separately by the ingestion service.
    """

    def __init__(
        self,
        name: str | None = None,
        llm_config: LLMConfig | dict[str, Any] | None = None,
        system_message: str | None = None,
        query_engine: RAGQueryEngine | None = None,
        config: DocAgentConfig | None = None,
    ) -> None:
        """Initialize the simplified DocAgent.

        Args:
            name: The name of the DocAgent
            llm_config: The configuration for the LLM
            system_message: The system message for the DocAgent
            query_engine: The RAG query engine to use
            config: Configuration for the DocAgent
        """
        name = name or "DocAgent"
        llm_config = llm_config or LLMConfig.get_current_llm_config()
        system_message = system_message or DEFAULT_SYSTEM_MESSAGE
        config = config or DocAgentConfig()

        # Import here to avoid circular imports
        from ....agentchat import ConversableAgent

        self._agent = ConversableAgent(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

        self.config = config
        self.query_engine = query_engine

        # Register the query function
        self._agent.register_reply([ConversableAgent, None], self._query_reply, position=0)

    def _query_reply(
        self,
        messages: list[dict[str, Any]] | str | None = None,
        sender: Optional["ConversableAgent"] = None,
        config: Any | None = None,
    ) -> tuple[bool, str | dict[str, Any] | None]:
        """Handle query requests."""
        if not self.query_engine:
            return True, "No query engine configured. Please set up a RAG backend first."

        # Extract the query from messages
        if isinstance(messages, str):
            query = messages
        elif isinstance(messages, list) and len(messages) > 0:
            query = messages[-1].get("content", "")
        else:
            return True, "Invalid message format."

        try:
            # Execute the query
            answer = self.query_engine.query(query)
            return True, answer
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return True, f"Error processing query: {str(e)}"

    def set_query_engine(self, query_engine: RAGQueryEngine) -> None:
        """Set the query engine for this agent."""
        self.query_engine = query_engine

    def run(
        self,
        message: str,
        max_turns: int | None = None,
    ) -> Any:
        """Run the DocAgent with a query message."""
        return self._agent.run(
            message=message,
            max_turns=max_turns or 1,
        )

    @property
    def name(self) -> str:
        """Get the agent name."""
        return str(self._agent.name)

    @property
    def system_message(self) -> str:
        """Get the system message."""
        return str(self._agent.system_message)
