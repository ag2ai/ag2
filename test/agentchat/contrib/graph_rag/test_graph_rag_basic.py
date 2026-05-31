# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from unittest.mock import Mock

from autogen.agentchat.contrib.graph_rag.graph_query_engine import GraphQueryEngine
from autogen.agentchat.contrib.graph_rag.graph_rag_capability import GraphRagCapability
from autogen.agentchat.conversable_agent import ConversableAgent


def test_dry_run() -> None:
    """Dry run for basic graph rag objects."""
    mock_graph_query_engine = Mock(spec=GraphQueryEngine)

    graph_rag_agent = ConversableAgent(
        name="graph_rag_agent",
        max_consecutive_auto_reply=3,
    )
    graph_rag_capability = GraphRagCapability(mock_graph_query_engine)
    graph_rag_capability.add_to_agent(graph_rag_agent)
