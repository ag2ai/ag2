# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple, Union

from autogen import Agent, ConversableAgent, UserProxyAgent

from .falkor_graph_query_engine import FalkorGraphQueryEngine, FalkorGraphQueryResult
from .graph_rag_capability import GraphRagCapability


class FalkorGraphRagCapability(GraphRagCapability):
    """
    The Falkor graph rag capability integrate FalkorDB graphrag_sdk version: 0.1.3b0.
    Ref: https://github.com/FalkorDB/GraphRAG-SDK/tree/2-move-away-from-sql-to-json-ontology-detection

    For usage, please refer to example notebook/agentchat_graph_rag_falkordb.ipynb
    """

    def __init__(self, query_engine: FalkorGraphQueryEngine):
        """
        initialize graph rag capability with a graph query engine
        """
        self.query_engine = query_engine

        # Graph DB query history.
        self._history = []

    def add_to_agent(self, agent: UserProxyAgent):
        """
        Add FalkorDB graph RAG capability to a UserProxyAgent.
        The restriction to a UserProxyAgent to make sure the returned message does not contain information retrieved from the graph DB instead of any LLMs.
        """
        self.graph_rag_agent = agent

        # Validate the agent config
        if agent.llm_config not in (None, False):
            raise Exception(
                "Graph rag capability limits the query to graph DB, llm_config must be a dict or False or None."
            )

        # Register method to generate reply
        agent.register_reply(
            [ConversableAgent, None], self._reply_using_falkordb_query, position=0, remove_other_reply_funcs=True
        )

    def _reply_using_falkordb_query(
        self,
        recipient: ConversableAgent,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """
        Query FalkorDB before return the message. Internally, it invokes the OpenAI assistant to generate a reply based on the given messages.
        The history with FalkorDB is also logged and updated.

        Args:
            recipient: The agent instance that will receive the message.
            messages: A list of messages in the conversation history with the sender.
            sender: The agent instance that sent the message.
            config: Optional configuration for message processing.

        Returns:
            A tuple containing a boolean indicating success and the assistant's reply.
        """
        question = self._get_last_question(messages[-1])
        result: FalkorGraphQueryResult = self.query_engine.query(question, messages=self._history)
        self._history = result.messages
        return True, result.answer if result.answer else "I'm sorry, I don't have an answer for that."

    def _get_last_question(self, message: Union[Dict, str]):
        if isinstance(message, str):
            return message
        if isinstance(message, Dict):
            if "content" in message:
                return message["content"]
        return None
