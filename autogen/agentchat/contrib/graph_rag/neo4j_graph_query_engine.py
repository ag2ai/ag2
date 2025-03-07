# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import os
import sys
from typing import Any, Optional, Union

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from ....import_utils import optional_import_block, require_optional_import
from .document import Document, DocumentType
from .graph_query_engine import GraphStoreQueryResult

with optional_import_block():
    from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.chat_engine.types import ChatMode
    from llama_index.core.indices.property_graph import (
        DynamicLLMPathExtractor,
        SchemaLLMPathExtractor,
    )
    from llama_index.core.indices.property_graph.transformations.schema_llm import Triple
    from llama_index.core.llms import LLM
    from llama_index.core.readers.json import JSONReader
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.core.schema import TransformComponent
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
    from llama_index.llms.openai import OpenAI


@require_optional_import("llama_index", "neo4j")
class Neo4jGraphQueryEngine:
    """This class serves as a wrapper for a property graph query engine backed by LlamaIndex and Neo4j,
    facilitating the creating, connecting, updating, and querying of LlamaIndex property graphs.

    It builds a property graph Index from input documents,
    storing and retrieving data from the property graph in the Neo4j database.

    It extracts triplets, i.e., [entity] -> [relationship] -> [entity] sets,
    from the input documents using llamIndex extractors.

    Users can provide custom entities, relationships, and schema to guide the extraction process.

    If strict is True, the engine will extract triplets following the schema
    of allowed relationships for each entity specified in the schema.

    It also leverages LlamaIndex's chat engine which has a conversation history internally to provide context-aware responses.

    For usage, please refer to example notebook/agentchat_graph_rag_neo4j.ipynb
    """

    def __init__(  # type: ignore[no-any-unimported]
        self,
        host: str = "bolt://localhost",
        port: int = 7687,
        database: str = "neo4j",
        username: str = "neo4j",
        password: str = "neo4j",
        llm: Optional["LLM"] = None,
        embedding: Optional["BaseEmbedding"] = None,
        entities: Optional["TypeAlias"] = None,
        relations: Optional["TypeAlias"] = None,
        schema: Optional[Union[dict[str, str], list["Triple"]]] = None,
        strict: Optional[bool] = False,
    ):
        """Initialize a Neo4j Property graph.
        Please also refer to https://docs.llamaindex.ai/en/stable/examples/property_graph/graph_store/

        Args:
            name (str): Property graph name.
            host (str): Neo4j hostname.
            port (int): Neo4j port number.
            database (str): Neo4j database name.
            username (str): Neo4j username.
            password (str): Neo4j password.
            llm (LLM): Language model to use for extracting triplets.
            embedding (BaseEmbedding): Embedding model to use constructing index and query
            entities (Optional[TypeAlias]): Custom suggested entities to include in the graph.
            relations (Optional[TypeAlias]): Custom suggested relations to include in the graph.
            schema (Optional[Union[Dict[str, str], List[Triple]]): Custom schema to specify allowed relationships for each entity.
            strict (Optional[bool]): If false, allows for values outside of the input schema.
        """
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.llm = llm or OpenAI(model="gpt-4o", temperature=0.0)
        self.embedding = embedding or OpenAIEmbedding(model_name="text-embedding-3-small")
        self.entities = entities
        self.relations = relations
        self.schema = schema
        self.strict = strict

    def init_db(self, input_doc: Optional[list[Document]] = None) -> None:
        """Build the knowledge graph with input documents."""
        self.documents = self._load_doc(input_doc if input_doc is not None else [])

        self.graph_store = Neo4jPropertyGraphStore(
            username=self.username,
            password=self.password,
            url=self.host + ":" + str(self.port),
            database=self.database,
        )

        # delete all entities and relationships in case a graph pre-exists
        self._clear()

        # Create knowledge graph extractors.
        self.kg_extractors = self._create_kg_extractors()

        self.index = PropertyGraphIndex.from_documents(
            self.documents,
            llm=self.llm,
            embed_model=self.embedding,
            kg_extractors=self.kg_extractors,
            property_graph_store=self.graph_store,
            show_progress=True,
        )

    def connect_db(self) -> None:
        """Connect to an existing knowledge graph database."""
        self.graph_store = Neo4jPropertyGraphStore(
            username=self.username,
            password=self.password,
            url=self.host + ":" + str(self.port),
            database=self.database,
        )

        self.kg_extractors = self._create_kg_extractors()

        self.index = PropertyGraphIndex.from_existing(
            property_graph_store=self.graph_store,
            kg_extractors=self.kg_extractors,
            llm=self.llm,
            embed_model=self.embedding,
            show_progress=True,
        )

    def add_records(self, new_records: list[Document]) -> bool:
        """Add new records to the knowledge graph. Must be local files.

        Args:
            new_records (List[Document]): List of new documents to add.

        Returns:
            bool: True if successful, False otherwise.
        """
        if self.graph_store is None:
            raise ValueError("Knowledge graph is not initialized. Please call init_db or connect_db first.")

        try:
            """
            SimpleDirectoryReader will select the best file reader based on the file extensions,
            see  _load_doc for supported file types.
            """
            new_documents = SimpleDirectoryReader(input_files=[doc.path_or_url for doc in new_records]).load_data()

            for doc in new_documents:
                self.index.insert(doc)

            return True
        except Exception as e:
            print(f"Error adding records: {e}")
            return False

    def query(self, question: str, n_results: int = 1, **kwargs: Any) -> GraphStoreQueryResult:
        """Query the property graph with a question using LlamaIndex chat engine.
        We use the condense_plus_context chat mode
        which condenses the conversation history and the user query into a standalone question,
        and then build a context for the standadlone question
        from the property graph to generate a response.

        Args:
            question: a human input question.
            n_results: number of results to return.
            **kwargs: additional keyword arguments.

        Returns:
            A GrapStoreQueryResult object containing the answer and related triplets.
        """
        if not hasattr(self, "index"):
            raise ValueError("Property graph index is not created.")

        # Initialize chat engine if not already initialized
        if not hasattr(self, "chat_engine"):
            self.chat_engine = self.index.as_chat_engine(chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT, llm=self.llm)

        response = self.chat_engine.chat(question)
        return GraphStoreQueryResult(answer=str(response))

    def _clear(self) -> None:
        """Delete all entities and relationships in the graph.
        TODO: Delete all the data in the database including indexes and constraints.
        """
        with self.graph_store._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n;")

    def _load_doc(self, input_doc: list[Document]) -> list["LlamaDocument"]:  # type: ignore[no-any-unimported]
        """Load documents from the input files. Currently support the following file types:
        .csv - comma-separated values
        .docx - Microsoft Word
        .epub - EPUB ebook format
        .hwp - Hangul Word Processor
        .ipynb - Jupyter Notebook
        .jpeg, .jpg - JPEG image
        .mbox - MBOX email archive
        .md - Markdown
        .mp3, .mp4 - audio and video
        .pdf - Portable Document Format
        .png - Portable Network Graphics
        .ppt, .pptm, .pptx - Microsoft PowerPoint
        .json JSON files
        """
        for doc in input_doc:
            if not os.path.exists(doc.path_or_url):  # type: ignore[arg-type]
                raise ValueError(f"Document file not found: {doc.path_or_url}")

        common_type_input_files = []
        json_type_input_files = []
        for doc in input_doc:
            if doc.doctype is DocumentType.JSON:
                json_type_input_files.append(doc.path_or_url)
            else:
                common_type_input_files.append(doc.path_or_url)
        loaded_documents = []
        if common_type_input_files:
            loaded_documents.extend(SimpleDirectoryReader(input_files=common_type_input_files).load_data())
        for json_file in json_type_input_files:
            loaded_documents.extend(JSONReader().load_data(input_file=json_file))  # type: ignore[arg-type]

        return loaded_documents

    def _create_kg_extractors(self) -> list["TransformComponent"]:  # type: ignore[no-any-unimported]
        """If strict is True,
        extract paths following a strict schema of allowed relationships for each entity.

        If strict is False,
        auto-create relationships and schema that fit the graph

        # To add more extractors, please refer to https://docs.llamaindex.ai/en/latest/module_guides/indexing/lpg_index_guide/#construction
        """
        #
        kg_extractors: list["TransformComponent"] = [  # type: ignore[no-any-unimported]
            SchemaLLMPathExtractor(
                llm=self.llm,
                possible_entities=self.entities,
                possible_relations=self.relations,
                kg_validation_schema=self.schema,
                strict=self.strict if self.strict else False,
            ),
        ]

        # DynamicLLMPathExtractor will auto-create relationships and schema that fit the graph
        if not self.strict:
            kg_extractors.append(
                DynamicLLMPathExtractor(
                    llm=self.llm,
                    allowed_entity_types=self.entities,
                    allowed_relation_types=self.relations,
                )
            )

        return kg_extractors
