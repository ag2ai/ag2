# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

from autogen.agentchat.contrib.rag.query_engine import VectorDbQueryEngine
from autogen.agentchat.contrib.vectordb.base import VectorDBFactory
from autogen.import_utils import optional_import_block, require_optional_import

with optional_import_block():
    from llama_index.core.llms import LLM
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
    from pymongo import MongoClient


DEFAULT_COLLECTION_NAME = "docling-parsed-docs"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@require_optional_import(["pymongo", "llama_index"], "rag")
class MongoDBQueryEngine(VectorDbQueryEngine):
    """
    MongoDBQueryEngine is a production-ready implementation of the VectorDbQueryEngine for MongoDB.

    This engine leverages the VectorDBFactory to instantiate a MongoDB vector database
    (MongoDBAtlasVectorDB) and wraps its collection with a LlamaIndex vector store
    (MongoDBAtlasVectorSearch) to enable document indexing and retrieval.

    Conceptually, it mirrors the approach used in DoclingMdQueryEngine for ChromaDB.
    It provides methods to:
      - Connect to the database.
      - Initialize (or reinitialize) the collection with documents.
      - Add new documents.
      - Execute natural language queries against the vector index.
    """

    def __init__(  # type: ignore[misc, no-any-unimported, return]
        self,
        connection_string: str,
        db_name: str = "vector_db",
        collection_name: str = "default_collection",
        vector_index_name: str = "vector_index",
        embedding_function: Optional[Any] = None,
        llm: Optional[LLM] = None,
        **kwargs: Any,
    ) -> Any | None:
        """
        Initializes the MongoDBQueryEngine.

        Args:
            connection_string: MongoDB connection string.
            db_name: Name of the MongoDB database.
            collection_name: Name of the collection to use (default is DEFAULT_COLLECTION_NAME).
            vector_index_name: Name of the vector search index.
            embedding_function: Function to compute embeddings (if needed by the underlying vector DB).
            llm: LLM for query processing (default uses OpenAI's GPT-4 variant).
            **kwargs: Additional keyword arguments.
        """
        self.connection_string = connection_string
        self.db_name = db_name
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
        self.vector_index_name = vector_index_name

        # Set up the LLM; if not provided, use a default OpenAI model.
        self.llm = llm or OpenAI(model="gpt-4o", temperature=0.0)  # type: ignore

        # Create a MongoDB client for use by the vector search wrapper.
        self.mongodb_client = MongoClient(connection_string)

        # Initialize the LlamaIndex-style vector store wrapper for advanced query pipelines.
        self.vector_search = MongoDBAtlasVectorSearch(
            mongodb_client=self.mongodb_client,
            db_name=db_name,
            collection_name=collection_name,
            vector_index_name=vector_index_name,
            **kwargs,
        )

        # Create the full vector database instance via the VectorDBFactory.
        self.vector_db = VectorDBFactory.create_vector_db(
            "mongodb",
            connection_string=connection_string,
            database_name=db_name,
            collection_name=collection_name,
            index_name=vector_index_name,
            embedding_function=embedding_function,
            **kwargs,
        )

    def connect_db(self, *args: Any, **kwargs: Any) -> bool:
        """
        Connect to the MongoDB database by issuing a ping.

        Returns:
            True if the connection is successful; False otherwise.
        """
        try:
            self.mongodb_client.admin.command("ping")
            return True
        except Exception as error:
            logger.error("Failed to connect to MongoDB: %s", error)
            return False

    def init_db(
        self,
        new_doc_dir: Optional[Union[Path, str]] = None,
        new_doc_paths: Optional[List[Union[Path, str]]] = None,
        overwrite: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """
        Initialize the database with documents.

        This method:
          1. Connects to MongoDB.
          2. Creates (or overwrites) the target collection via the vector DB interface.
          3. Loads documents from a directory and/or file paths.
          4. Inserts the documents into the collection.
          5. Creates the vector search index.

        Args:
            new_doc_dir: Directory containing document files.
            new_doc_paths: List of file paths to individual documents.
            overwrite: If True, the existing collection is overwritten.
            *args, **kwargs: Additional arguments.

        Returns:
            True if initialization is successful; False otherwise.
        """
        if not self.connect_db():
            return False

        try:
            self.vector_db.create_collection(
                collection_name=self.collection_name, overwrite=overwrite, get_or_create=True
            )
        except Exception as e:
            logger.error("Error creating collection: %s", e)
            return False

        # Load documents from file paths and/or a directory.
        docs = []
        if new_doc_paths:
            for i, doc_path in enumerate(new_doc_paths):
                path_obj = Path(doc_path)
                if path_obj.is_file():
                    content = path_obj.read_text(encoding="utf-8")
                    docs.append({
                        "id": f"doc_{i}",
                        "content": content,
                        "metadata": None,
                        "embedding": None,  # Embeddings will be computed internally.
                    })
        if new_doc_dir:
            dir_path = Path(new_doc_dir)
            for i, file in enumerate(dir_path.glob("*")):
                if file.is_file():
                    content = file.read_text(encoding="utf-8")
                    docs.append({"id": f"doc_dir_{i}", "content": content, "metadata": None, "embedding": None})

        if docs:
            self.vector_db.insert_docs(docs, collection_name=self.collection_name)  # type: ignore[arg-type]

        # Create the vector search index.
        try:
            dimensions = getattr(self.vector_db, "dimensions", 1536)
            self.vector_search.create_vector_search_index(dimensions=dimensions, path="embedding", similarity="cosine")
        except Exception as e:
            logger.error("Error creating vector search index: %s", e)

        return True

    def add_records(
        self,
        new_doc_dir: Optional[Union[Path, str]] = None,
        new_doc_paths_or_urls: Optional[List[Union[Path, str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """
        Add new documents to the existing collection.

        Loads documents from the specified directory and/or file paths and inserts them into the vector DB.

        Returns:
            True if records were added successfully; False otherwise.
        """
        docs = []
        if new_doc_paths_or_urls:
            for i, doc_path in enumerate(new_doc_paths_or_urls):
                path_obj = Path(doc_path)
                if path_obj.is_file():
                    content = path_obj.read_text(encoding="utf-8")
                    docs.append({"id": f"new_doc_{i}", "content": content, "metadata": None, "embedding": None})
        if new_doc_dir:
            dir_path = Path(new_doc_dir)
            for i, file in enumerate(dir_path.glob("*")):
                if file.is_file():
                    content = file.read_text(encoding="utf-8")
                    docs.append({"id": f"new_doc_dir_{i}", "content": content, "metadata": None, "embedding": None})
        if docs:
            self.vector_db.insert_docs(docs, collection_name=self.collection_name)  # type: ignore[arg-type]
            return True
        return False

    def query(self, question: str, *args: Any, **kwargs: Any) -> str:
        """
        Execute a natural language query against the vector database.

        Converts the query string into a vector search, retrieves the most relevant document,
        and returns its content.

        Args:
            question: The natural language query.
            *args, **kwargs: Additional query parameters (e.g., n_results, distance_threshold).

        Returns:
            The content of the top matching document, or an empty string if no match is found.
        """
        results = self.vector_db.retrieve_docs(
            queries=[question], collection_name=self.collection_name, n_results=1, **kwargs
        )
        if results and results[0]:
            best_doc, _ = results[0][0]
            return best_doc.get("content", "")
        return ""
