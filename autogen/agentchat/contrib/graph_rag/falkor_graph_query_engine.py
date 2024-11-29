# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass, field
from typing import List

from graphrag_sdk import KnowledgeGraph, Source
from graphrag_sdk.model_config import KnowledgeGraphModelConfig
from graphrag_sdk.models.openai import OpenAiGenerativeModel
from graphrag_sdk.ontology import Ontology

from .document import Document
from .graph_query_engine import GraphStoreQueryResult


class FalkorGraphQueryEngine:
    """
    This is a wrapper for FalkorDB KnowledgeGraph.
    """

    def __init__(
        self,
        name: str,
        host: str = "127.0.0.1",
        port: int = 6379,
        username: str | None = None,
        password: str | None = None,
        model: str = "gpt-4o",
        ontology: Ontology | None = None,
    ):
        """
        Initialize a FalkorDB knowledge graph.
        Please also refer to https://github.com/FalkorDB/GraphRAG-SDK/blob/2-move-away-from-sql-to-json-ontology-detection/graphrag_sdk/kg.py

        Args:
            name (str): Knowledge graph name.
            host (str): FalkorDB hostname.
            port (int): FalkorDB port number.
            username (str|None): FalkorDB username.
            password (str|None): FalkorDB password.
            model (str): OpenAI model to use for FalkorDB to build and retrieve from the graph.
            ontology: FalkorDB knowledge graph schema/ontology, https://github.com/FalkorDB/GraphRAG-SDK/blob/2-move-away-from-sql-to-json-ontology-detection/graphrag_sdk/schema/schema.py
                If None, FalkorDB will auto generate an ontology from the input docs.
        """
        openai_model = OpenAiGenerativeModel(model)
        self.knowledge_graph = KnowledgeGraph(
            name=name,
            host=host,
            port=port,
            username=username,
            password=password,
            model_config=KnowledgeGraphModelConfig.with_model(openai_model),
            ontology=ontology,
        )

        # Establish a chat session, this will maintain the history
        self._chat_session = self.knowledge_graph.chat_session()

    def init_db(self, input_doc: List[Document] | None):
        """
        Build the knowledge graph with input documents.
        """
        sources = []
        for doc in input_doc:
            if os.path.exists(doc.path_or_url):
                sources.append(Source(doc.path_or_url))

        if sources:
            self.knowledge_graph.process_sources(sources)

    def add_records(self, new_records: List) -> bool:
        raise NotImplementedError("This method is not supported by FalkorDB SDK yet.")

    def query(self, question: str, n_results: int = 1, **kwargs) -> GraphStoreQueryResult:
        """
        Query the knowledge graph with a question and optional message history.

        Args:
        question: a human input question.
        n_results: number of returned results.
        kwargs:
            messages: a list of message history.

        Returns: FalkorGraphQueryResult
        """
        response = self._chat_session.send_message(question)

        # History will be considered when querying by setting the last_answer
        self._chat_session.last_answer = response["response"]

        return GraphStoreQueryResult(answer=response["response"], results=[])
