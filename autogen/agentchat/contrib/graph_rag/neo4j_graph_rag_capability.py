# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

from .... import Agent, ConversableAgent, UserProxyAgent
from .graph_query_engine import GraphStoreQueryResult
from .graph_rag_capability import GraphRagCapability
from .neo4j_graph_query_engine import Neo4jGraphQueryEngine


class Neo4jGraphCapability(GraphRagCapability):
    """The Neo4j graph capability integrates Neo4j Property graph into a graph rag agent.
    Ref: https://neo4j.com/labs/genai-ecosystem/llamaindex/#_property_graph_constructing_modules


    For usage, please refer to example notebook/agentchat_graph_rag_neo4j.ipynb
    """

    def __init__(self, query_engine: Neo4jGraphQueryEngine):
        """Initialize GraphRAG capability with a graph query engine"""
        self.query_engine = query_engine

    def add_to_agent(self, agent: ConversableAgent) -> None:
        """Add Neo4j GraphRAG capability to a UserProxyAgent.
        The restriction to a UserProxyAgent to make sure the returned message only contains information retrieved from the graph DB instead of any LLMs.
        """
        if not isinstance(agent, UserProxyAgent):
            raise Exception("Neo4j GraphRAG capability can only be added to a UserProxyAgent.")

        self.graph_rag_agent = agent

        # Validate the agent config
        if agent.llm_config not in (None, False):
            raise Exception(
                "Agents with GraphRAG capabilities do not use an LLM configuration. Please set your llm_config to None or False."
            )

        # Register method to generate the reply using a Neo4j query
        # All other reply methods will be removed
        agent.register_reply(
            [ConversableAgent, None], self._reply_using_neo4j_query, position=0, remove_other_reply_funcs=True
        )

    def _reply_using_neo4j_query(
        self,
        recipient: ConversableAgent,
        messages: Optional[list[dict[str, Any]]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> tuple[bool, Optional[Union[str, dict[str, Any]]]]:
        """Query neo4j and return the message. Internally, it queries the Property graph
        and returns the answer from the graph query engine.
        TODO: reply with a dictionary including both the answer and semantic source triplets.

        Args:
            recipient: The agent instance that will receive the message.
            messages: A list of messages in the conversation history with the sender.
            sender: The agent instance that sent the message.
            config: Optional configuration for message processing.

        Returns:
            A tuple containing a boolean indicating success and the assistant's reply.
        """
        if not messages:
            return False, None

        question = self._get_last_question(messages[-1])
        if not question:
            return False, None

        result: GraphStoreQueryResult = self.query_engine.query(question)  # type: ignore[arg-type]

        return True, result.answer

    def _get_last_question(self, message: Union[dict[str, Any], str]) -> Optional[Union[str, dict[str, Any]]]:
        """Retrieves the last message from the conversation history."""
        if isinstance(message, str):
            return message
        if isinstance(message, dict) and "content" in message:
            return message["content"]  # type: ignore[no-any-return]
        return None
