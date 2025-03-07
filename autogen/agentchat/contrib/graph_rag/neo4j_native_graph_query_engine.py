# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import List, Optional, Union

from ....import_utils import optional_import_block, require_optional_import
from .document import Document, DocumentType
from .graph_query_engine import GraphStoreQueryResult

with optional_import_block():
    from neo4j import GraphDatabase
    from neo4j_graphrag.embeddings import Embedder, OpenAIEmbeddings
    from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
    from neo4j_graphrag.generation import GraphRAG
    from neo4j_graphrag.indexes import create_vector_index
    from neo4j_graphrag.llm.openai_llm import LLMInterface, OpenAILLM
    from neo4j_graphrag.retrievers import VectorRetriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@require_optional_import(["neo4j", "neo4j_graphrag"], "neo4j")
class Neo4jNativeGraphQueryEngine:
    """A graph query engine implemented using the Neo4j GraphRAG SDK.
    Provides functionality to initialize a knowledge graph,
    create a vector index, and query the graph using Neo4j and LLM.
    """

    def __init__(  # type: ignore[no-any-unimported]
        self,
        host: str = "neo4j://localhost",
        port: int = 7687,
        username: str = "neo4j",
        password: str = "password",
        embeddings: Optional["Embedder"] = None,
        embedding_dimension: Optional[int] = 3072,
        llm: Optional["LLMInterface"] = None,
        query_llm: Optional["LLMInterface"] = None,
        entities: Optional[List[str]] = None,
        relations: Optional[List[str]] = None,
        potential_schema: Optional[List[tuple[str, str, str]]] = None,
    ):
        """Initialize a Neo4j graph query engine.

        Args:
            host (str): Neo4j host URL.
            port (int): Neo4j port.
            username (str): Neo4j username.
            password (str): Neo4j password.
            embeddings (Embedder): Embedding model to embed chunk data and retrieve answers.
            embedding_dimension (int): Dimension of the embeddings for the model.
            llm (LLMInterface): Language model for creating the knowledge graph (returns JSON responses).
            query_llm (LLMInterface): Language model for querying the knowledge graph.
            entities (List[str], optional): Custom entities for guiding graph construction.
            relations (List[str], optional): Custom relations for guiding graph construction.
            potential_schema (List[tuple[str, str, str]], optional): Schema (triplets, i.e., [entity] -> [relationship] -> [entity]) to guide graph construction.
        """
        self.uri = f"{host}:{port}"
        self.driver = GraphDatabase.driver(self.uri, auth=(username, password))
        self.embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-large")
        self.embedding_dimension = embedding_dimension
        self.llm = llm or OpenAILLM(
            model_name="gpt-4o",
            model_params={"response_format": {"type": "json_object"}, "temperature": 0},
        )
        self.query_llm = query_llm or OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})
        self.entities = entities
        self.relations = relations
        self.potential_schema = potential_schema

    def init_db(self, input_doc: Optional[Union[list[Document]]] = None) -> None:
        """Initialize the Neo4j graph database using the provided input doc.
        Currently this method only supports single document input (only reads the first doc).

        This method supports both text and PDF documents. It performs the following steps:
        1. Clears the existing database.
        2. Extracts graph nodes and relationships from the input data to build a knowledge graph.
        3. Creates a vector index for efficient retrieval.

        Args:
            input_doc (list[Document]): Input documents for building the graph.

        Raises:
            ValueError: If the input document is not provided or its type is unsupported.
        """
        if input_doc is None or len(input_doc) == 0:
            raise ValueError("Input document is required to initialize the database.")
        elif len(input_doc) > 1:
            raise ValueError("Only the first document will be used to initialize the database.")

        logger.info("Clearing the database...")
        self._clear_db()

        self._initialize_kg_builders()

        self._build_graph(input_doc)

        self.index_name = "vector-index-name"
        logger.info(f"Creating vector index '{self.index_name}'...")
        self._create_index(self.index_name)

    def add_records(self, new_records: list[Document]) -> bool:
        """Add new records to the Neo4j database.

        Args:
            new_records (list[Document]): List of new Documents to be added

        Returns:
            bool: True if records were added successfully, False otherwise.
        """
        for record in new_records:
            if not isinstance(record, Document):
                raise ValueError("Invalid record type. Expected Document.")

        self._build_graph(new_records)

        return True

    def query(self, question: str) -> GraphStoreQueryResult:
        """Query the Neo4j database using a natural language question.

        Args:
            question (str): The question to be answered by querying the graph.

        Returns:
            GraphStoreQueryResult: The result of the query.
        """
        self.retriever = VectorRetriever(
            driver=self.driver,
            index_name=self.index_name,
            embedder=self.embeddings,
        )
        rag = GraphRAG(retriever=self.retriever, llm=self.query_llm)
        result = rag.search(query_text=question, retriever_config={"top_k": 5})

        return GraphStoreQueryResult(answer=result.answer)

    def _create_index(self, name: str) -> None:
        """Create a vector index for the Neo4j knowledge graph.

        Args:
            name (str): Name of the vector index to create.
        """
        logger.info(f"Creating vector index '{name}'...")
        create_vector_index(
            self.driver,
            name=name,
            label="Chunk",
            embedding_property="embedding",
            dimensions=self.embedding_dimension,
            similarity_fn="euclidean",
        )
        logger.info(f"Vector index '{name}' created successfully.")

    def _clear_db(self) -> None:
        """Clear all nodes and relationships from the Neo4j database."""
        logger.info("Clearing all nodes and relationships in the database...")
        self.driver.execute_query("MATCH (n) DETACH DELETE n;")
        logger.info("Database cleared successfully.")

    def _initialize_kg_builders(self) -> None:
        """Initialize the knowledge graph builders"""
        logger.info("Initializing the knowledge graph builders...")
        self.text_kg_builder = SimpleKGPipeline(
            driver=self.driver,
            embedder=self.embeddings,
            llm=self.llm,
            entities=self.entities,
            relations=self.relations,
            potential_schema=self.potential_schema,
            on_error="IGNORE",
            from_pdf=False,
        )

        self.pdf_kg_builder = SimpleKGPipeline(
            driver=self.driver,
            embedder=self.embeddings,
            llm=self.llm,
            entities=self.entities,
            relations=self.relations,
            potential_schema=self.potential_schema,
            on_error="IGNORE",
            from_pdf=True,
        )

    def _build_graph(self, input_doc: List[Document]) -> None:
        """Build the knowledge graph using the provided input documents.

        Args:
            input_doc (List[Document]): List of input documents for building the graph.
        """
        logger.info("Building the knowledge graph...")
        for doc in input_doc:
            if doc.doctype == DocumentType.TEXT:
                # todo: we assume this is a path, and not URL
                with open(doc.path_or_url) as file:  # type: ignore[arg-type]
                    text = file.read()
                asyncio.run(self.text_kg_builder.run_async(text=text))
            elif doc.doctype == DocumentType.PDF:
                asyncio.run(self.pdf_kg_builder.run_async(file_path=doc.path_or_url))
            else:
                raise ValueError(f"Unsupported document type: {doc.doctype}")

        logger.info("Knowledge graph built successfully.")
