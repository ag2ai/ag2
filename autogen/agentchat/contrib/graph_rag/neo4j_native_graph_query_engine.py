# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import List, Optional

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import Embedder, OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.llm.openai_llm import LLMInterface, OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever

from .document import Document, DocumentType
from .graph_query_engine import GraphQueryEngine, GraphStoreQueryResult


class Neo4jNativeGraphQueryEngine(GraphQueryEngine):
    """
    A graph query engine implemented by Neo4j GraphRag sdk.
    """

    def __init__(
        self,
        host: str = "neo4j://localhost",
        port: int = 7687,
        username: str = "neo4j",
        password: str = "password",
        embeddings: Embedder | None = None,
        embedding_dimension: int | None = None,
        llm: LLMInterface | None = None,
        query_llm: LLMInterface | None = None,
        entities: Optional[List[str]] = None,
        relations: Optional[List[str]] = None,
        potential_schema: Optional[List[tuple[str, str, str]]] = None,
    ):
        """
        Initialize a Neo4j graph query engine.

        Args:
        host: Neo4j host.
        port: Neo4j port.
        username: Neo4j username.
        password: Neo4j password.
        embeddings: embedding model used to embed chunk data and retrieve answer.
        embedding_dimension: embedding dimension that matches the embedding model.
        llm: language model used to create knowledge graph, which needs to return json format response.
        query_llm: language model used to query knowledge graph
        entities: custom entities to guide the graph construction.
        relations: custom relations to guide the graph construction.
        potential_schema: potential schema, as a list of triplets (entity -> relationship -> entity, to guide the graph construction.
        """
        self.uri = host + ":" + str(port)
        self.driver = GraphDatabase.driver(self.uri, auth=(username, password))
        self.embeddings = embeddings if embeddings else OpenAIEmbeddings(model="text-embedding-3-large")
        self.embedding_dimension = embedding_dimension if embedding_dimension else 3072
        if llm:
            self.llm = llm
        else:
            self.llm = OpenAILLM(model_name="gpt-4o", model_params={"response_format": "json_object", "temperature": 0})
        self.query_llm = query_llm if query_llm else OpenAILLM(model_name="gpt-4o")
        self.entities = entities
        self.relations = relations
        self.potential_schema = potential_schema

    def init_db(self, input_doc: Document | None = None):
        """
        Initialize the Neo4j graph database with the input document. The query engine only supports text and pdf input.
        It takes the following steps:
        1. Connect to the Neo4j graph database.
        2. Extract graph nodes and relationships based on input data and build a knowledge graph.
        3. Build a vector index for the knowledge graph for retrieval.

        Args:
        input_doc: a list of input documents that are used to build the graph in database.

        """
        if input_doc is None:
            raise ValueError("Input document is required to initialize the database.")
        elif input_doc.doctype == DocumentType.TEXT:
            from_pdf = False
        elif input_doc.doctype == DocumentType.PDF:
            from_pdf = True
        else:
            raise ValueError("Only support pdf or text input.")

        self._clear_db()

        self.kg_builder = SimpleKGPipeline(
            driver=self.driver,
            embedder=self.embeddings,
            llm=self.llm,
            entities=self.entities,
            relations=self.relations,
            potential_schema=self.potential_schema,
            on_error="IGNORE",
            from_pdf=from_pdf,
        )

        if from_pdf:
            asyncio.run(self.kg_builder.run_async(file_path=input_doc.path_or_url))
        else:
            asyncio.run(self.kg_builder.run_async(text=str(input_doc.data)))

        self.index_name = "vector-index-name"
        self._create_index(self.index_name)

    def add_records(self, new_records: list) -> bool:
        """
        Add new records to the underlying Neo4j database and add to the graph if required.
        """
        pass

    def query(self, question: str, n_results: int = 1, **kwargs) -> GraphStoreQueryResult:
        """
        Transform a string format question into a Neo4j database query and return the result.
        """

        self.retriever = VectorRetriever(
            driver=self.driver,
            index_name=self.index_name,
            embedder=self.embeddings,
        )
        rag = GraphRAG(retriever=self.retriever, llm=self.query_llm)
        result = rag.search(query_text=question, retriever_config={"top_k": 5})

        return GraphStoreQueryResult(answer=result.answer)

    def _create_index(self, name: str):
        """
        Create a vector index for the knowledge graph.
        """

        create_vector_index(
            self.driver,
            name=name,
            label="Chunk",
            embedding_property="embedding",
            dimensions=self.embedding_dimension,
            similarity_fn="euclidean",
        )

    def _clear_db(self):
        """
        Clear the Neo4j database.
        """
        self.driver.execute_query("MATCH (n) DETACH DELETE n;")
