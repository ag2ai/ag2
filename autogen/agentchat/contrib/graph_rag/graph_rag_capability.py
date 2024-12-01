# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from autogen.agentchat.contrib.capabilities.agent_capability import AgentCapability
from autogen.agentchat.conversable_agent import ConversableAgent

from .graph_query_engine import GraphQueryEngine


class GraphRagCapability(AgentCapability):
    """
    A graph-based RAG capability uses a graph query engine to give a conversable agent the graph-based RAG ability.

    An agent class with graph-based RAG capability could
    1. create a graph in the underlying database with input documents.
    2. retrieved relevant information based on messages received by the agent.
    3. generate answers from retrieved information and send messages back.

    For example,
    graph_query_engine = GraphQueryEngine(...)
    graph_query_engine.init_db([Document(doc1), Document(doc2), ...])

    graph_rag_agent = ConversableAgent(
        name="graph_rag_agent",
        max_consecutive_auto_reply=3,
        ...
    )
    graph_rag_capability = GraphRagCapbility(graph_query_engine)
    graph_rag_capability.add_to_agent(graph_rag_agent)

    user_proxy = UserProxyAgent(
        name="user_proxy",
        code_execution_config=False,
        is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
        human_input_mode="ALWAYS",
    )
    user_proxy.initiate_chat(graph_rag_agent, message="Name a few actors who've played in 'The Matrix'")

    # ChatResult(
        # chat_id=None,
        # chat_history=[
            # {'content': 'Name a few actors who've played in \'The Matrix\'', 'role': 'graph_rag_agent'},
            # {'content': 'A few actors who have played in The Matrix are:
            #   - Keanu Reeves
            #   - Laurence Fishburne
            #   - Carrie-Anne Moss
            #   - Hugo Weaving',
            #   'role': 'user_proxy'},
        # ...)

    """

    def __init__(self, query_engine: GraphQueryEngine):
        """
        Initialize graph-based RAG capability with a graph query engine
        """
        ...

    def add_to_agent(self, agent: ConversableAgent):
        """Add the capability to an agent"""
        ...
