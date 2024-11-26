# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Union

from autogen import UserProxyAgent

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

        # Register a hook for processing the last message.
        agent.register_hook(hookable_method="process_last_received_message", hook=self.process_last_received_message)

        # Append extra info to the system message.
        agent.update_system_message(
            agent.system_message + "\nYou've been given the special ability to use graph rag to retrieve information."
        )

    def process_last_received_message(self, message: Union[Dict, str]):
        """
        Query FalkorDB before return the message.
        The history with FalkorDB is also logged and updated.
        """
        question = self._get_last_question(message)
        result: FalkorGraphQueryResult = self.query_engine.query(question, self._history)
        self._history = result.messages
        return result.answer

    def _get_last_question(self, message: Union[Dict, str]):
        if isinstance(message, str):
            return message
        if isinstance(message, Dict):
            if "content" in message:
                return message["content"]
        return None
