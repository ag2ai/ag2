# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Dict, List, Optional, TypeAlias, Union

from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.indices.property_graph.transformations.schema_llm import Triple
from llama_index.core.llms import LLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai import OpenAI

from .document import Document
from .graph_query_engine import GraphQueryEngine, GraphStoreQueryResult


class Neo4jGraphQueryEngine(GraphQueryEngine):
    """
    This class serves as a wrapper for a Neo4j database-backed PropertyGraphIndex query engine,
    facilitating the creation, updating, and querying of graphs.

    It builds a PropertyGraph Index from input documents,
    storing and retrieving data from a property graph in the Neo4j database.

    Using SchemaLLMPathExtractor, it defines schemas with entities, relationships, and other properties based on the input,
    which are added into the preprty graph.

    For usage, please refer to example notebook/agentchat_graph_rag_neo4j.ipynb
    """

    def __init__(
        self,
        host: str = "bolt://localhost",
        port: int = 7687,
        database: str = "neo4j",
        username: str = "neo4j",
        password: str = "neo4j",
        llm: LLM = OpenAI(model="gpt-4o", temperature=0.0),
        embedding: BaseEmbedding = OpenAIEmbedding(model_name="text-embedding-3-small"),
        entities: Optional[TypeAlias] = None,
        relations: Optional[TypeAlias] = None,
        validation_schema: Optional[Union[Dict[str, str], List[Triple]]] = None,
        strict: Optional[bool] = True,
    ):
        """
        Initialize a Neo4j Property graph.
        Please also refer to https://docs.llamaindex.ai/en/stable/examples/property_graph/graph_store/

        Args:
            name (str): Property graph name.
            host (str): Neo4j hostname.
            port (int): Neo4j port number.
            database (str): Neo4j database name.
            username (str): Neo4j username.
            password (str): Neo4j password.
            llm (LLM): Language model to use for extracting tripletss.
            embedding (BaseEmbedding): Embedding model to use constructing index and query
            entities (Optional[TypeAlias]): Custom possible entities to include in the graph.
            relations (Optional[TypeAlias]): Custom poissble relations to include in the graph.
            validation_schema (Optional[Union[Dict[str, str], List[Triple]]): Custom schema to validate the extracted triplets
            strict (Optional[bool]): If false, allows for values outside of the schema, useful for using the schema as a suggestion.
        """
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.llm = llm
        self.embedding = embedding
        self.entities = entities
        self.relations = relations
        self.validation_schema = validation_schema
        self.strict = strict

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

        # delete all entities and relationships in case a graph pre-exists
        self._clear()

        self.documents = SimpleDirectoryReader(input_files=self.input_files).load_data()

        # Extract paths following a strict schema of allowed entities, relationships, and which entities can be connected to which relationships.
        # To add more extractors, please refer to https://docs.llamaindex.ai/en/latest/module_guides/indexing/lpg_index_guide/#construction
        self.kg_extractors = [
            SchemaLLMPathExtractor(
                llm=self.llm,
                possible_entities=self.entities,
                possible_relations=self.relations,
                kg_validation_schema=self.validation_schema,
                strict=self.strict,
            )
        ]

        self.index = PropertyGraphIndex.from_documents(
            self.documents,
            embed_model=self.embedding,
            kg_extractors=self.kg_extractors,
            property_graph_store=self.graph_store,
            show_progress=True,
        )

    def add_records(self, new_records: List) -> bool:
        """
        Add new records to the knowledge graph. Must be local files.

        Args:
            new_records (List[Document]): List of new documents to add.

        Returns:
            bool: True if successful, False otherwise.
        """
        if self.graph_store is None:
            raise ValueError("Knowledge graph is not initialized. Please call init_db first.")

        try:
            """
            SimpleDirectoryReader will select the best file reader based on the file extensions, including:
            [DocxReader, EpubReader, HWPReader, ImageReader, IPYNBReader, MarkdownReader, MboxReader,
            PandasCSVReader, PandasExcelReader,PDFReader,PptxReader, VideoAudioReader]
            """
            new_documents = SimpleDirectoryReader(input_files=[doc.path_or_url for doc in new_records]).load_data()

            for doc in new_documents:
                self.index.insert(doc)

            return True
        except Exception as e:
            print(f"Error adding records: {e}")
            return False

    def query(self, question: str, n_results: int = 1, **kwargs) -> GraphStoreQueryResult:
        """
        Query the knowledge graph with a question.

        Args:
        question: a human input question.
        n_results: number of results to return.

        Returns:
        A GrapStoreQueryResult object containing the answer and related triplets.
        """
        if self.graph_store is None:
            raise ValueError("Knowledge graph is not created.")

        # query the graph to get the answer
        query_engine = self.index.as_query_engine(include_text=True)
        response = str(query_engine.query(question))

        # retrieve source triplets that are semantically related to the question
        retriever = self.index.as_retriever(include_text=False)
        nodes = retriever.retrieve(question)
        triplets = []
        for node in nodes:
            entities = [sub.split("(")[0].strip() for sub in node.text.split("->")]
            triplet = " -> ".join(entities)
            triplets.append(triplet)

        return GraphStoreQueryResult(answer=response, results=triplets)

    def _clear(self) -> None:
        """
        Delete all entities and relationships in the graph.
        TODO: Delete all the data in the database including indexes and constraints.
        """
        with self.graph_store._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n;")
