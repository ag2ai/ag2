# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

from .... import Agent, ConversableAgent
from .graph_query_engine import GraphStoreQueryResult
from .graph_rag_capability import GraphRagCapability
from .neo4j_native_graph_query_engine import Neo4jNativeGraphQueryEngine


class Neo4jNativeGraphCapability(GraphRagCapability):
    """The Neo4j native graph capability integrates Neo4j native query engine into a graph rag agent.

    For usage, please refer to example notebook/agentchat_graph_rag_neo4j_native.ipynb
    """

    def __init__(self, query_engine: Neo4jNativeGraphQueryEngine):
        """Initialize GraphRAG capability with a neo4j native graph query engine"""
        self.query_engine = query_engine

    def add_to_agent(self, agent: ConversableAgent):
        """Add native Neo4j GraphRAG capability to a ConversableAgent.
        llm_config of the agent must be None/False (default) to make sure the returned message only contains information retrieved from the graph DB instead of any LLMs.
        """
        self.graph_rag_agent = agent

        # Validate the agent config
        if agent.llm_config not in (None, False):
            raise Exception(
                "Agents with GraphRAG capabilities do not use an LLM configuration. Please set your llm_config to None or False."
            )

        # Register method to generate the reply using a Neo4j query
        # All other reply methods will be removed
        agent.register_reply(
            [ConversableAgent, None], self._reply_using_native_neo4j_query, position=0, remove_other_reply_funcs=True
        )

    def _reply_using_native_neo4j_query(
        self,
        recipient: ConversableAgent,
        messages: Optional[list[dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> tuple[bool, Union[str, dict, None]]:
        """Query Neo4j and return the message. Internally, it uses the Neo4jNativeGraphQueryEngine to query the graph.

        The agent's system message will be incorporated into the query, if it's not blank.

        If no results are found, a default message is returned: "I'm sorry, I don't have an answer for that."

        Args:
            recipient: The agent instance that will receive the message.
            messages: A list of messages in the conversation history with the sender.
            sender: The agent instance that sent the message.
            config: Optional configuration for message processing.

        Returns:
            A tuple containing a boolean indicating success and the assistant's reply.
        """
        question = self._messages_summary(messages, recipient.system_message)
        result: GraphStoreQueryResult = self.query_engine.query(question)

        return True, result.answer if result.answer else "I'm sorry, I don't have an answer for that."

    def _messages_summary(self, messages: Union[dict, str], system_message: str) -> str:
        """Summarize the messages in the conversation history. Excluding any message with 'tool_calls' and 'tool_responses'
        Includes the 'name' (if it exists) and the 'content', with a new line between each one, like:
        customer:
        <content>

        agent:
        <content>
        """
        if isinstance(messages, str):
            if system_message:
                summary = f"IMPORTANT: {system_message}\nContext:\n\n{messages}"
            else:
                return messages

        elif isinstance(messages, list):
            summary = ""
            for message in messages:
                if "content" in message and "tool_calls" not in message and "tool_responses" not in message:
                    summary += f"{message.get('name', '')}: {message.get('content', '')}\n\n"

            if system_message:
                summary = f"IMPORTANT: {system_message}\nContext:\n\n{summary}"

            return summary

        else:
            raise ValueError("Invalid messages format. Must be a list of messages or a string.")
