# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import List

from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai import OpenAI

from .document import Document
from .graph_query_engine import GraphQueryEngine, GraphStoreQueryResult


class Neo4jGraphQueryEngine(GraphQueryEngine):
    """
    This is a wrapper for Neo4j KnowledgeGraph.
    """

    def __init__(
        self,
        host: str = "bolt://localhost",
        port: int = 7687,
        database: str = "neo4j",
        username: str = "neo4j",
        password: str = "neo4j",
        model: str = "gpt-3.5-turbo",
        embed_model: str = "text-embedding-3-small",
    ):
        """
        Initialize a Neo4j knowledge graph.
        Please also refer to https://neo4j.com/docs/

        Args:
            name (str): Knowledge graph name.
            host (str): Neo4j hostname.
            port (int): Neo4j port number.
            database (str): Neo4j database name.
            username (str): Neo4j username.
            password (str): Neo4j password.
            model (str): LLM model to use for Neo4j to build and retrieve from the graph, default to use OAI gpt-3.5-turbo.
            include_embeddings (bool): Whether to include embeddings in the graph.
        """
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.model = model
        self.embed_model = embed_model

    def init_db(self, input_doc: List[Document] | None = None):
        """
        Build the knowledge graph with input documents.
        """
        self.input_files = []
        for doc in input_doc:
            if os.path.exists(doc.path_or_url):
                self.input_files.append(doc.path_or_url)
            else:
                raise ValueError(f"Document file not found: {doc.path_or_url}")

        self.graph_store = Neo4jPropertyGraphStore(
            username=self.username,
            password=self.password,
            url=self.host + ":" + str(self.port),
            database=self.database,
        )
        self.documents = SimpleDirectoryReader(input_files=self.input_files).load_data()

        self.index = PropertyGraphIndex.from_documents(
            self.documents,
            embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
            kg_extractors=[SchemaLLMPathExtractor(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.0))],
            property_graph_store=self.graph_store,
            show_progress=True,
        )

    def add_records(self, new_records: List) -> bool:
        """
        Add a record to the knowledge graph.
        """
        pass

    def query(self, question: str, n_results: int = 1, **kwargs) -> GraphStoreQueryResult:
        """
        Query the knowledge graph with a question and optional message history.

        Args:
        question: a human input question.
        n_results: number of results to return.

        Returns:
        Neo4j GrapStorehQueryResult
        """
        if self.graph_store is None:
            raise ValueError("Knowledge graph is not created.")

        # query the graph to get the answer
        query_engine = self.index.as_query_engine(include_text=True)
        response = str(query_engine.query(question))

        # retrieve source tripelets from the graph
        retriever = self.index.as_retriever(include_text=False)
        nodes = retriever.retrieve(question)

        return GraphStoreQueryResult(answer=response, results=nodes)
