# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from autogen.agentchat.contrib.vectordb.base import VectorDBFactory
from autogen.agentchat.contrib.vectordb.mongodb import MongoDBAtlasVectorDB
from autogen.import_utils import optional_import_block, require_optional_import

with optional_import_block():
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.llms.langchain.base import LLM
    from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
    from pymongo import MongoClient

DEFAULT_COLLECTION_NAME = "docling-parsed-docs"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@require_optional_import(["pymongo", "llama_index"], "rag")
class MongoDBQueryEngine:
    """
    A query engine backed by MongoDB Atlas that supports document insertion and querying.
    This is a autogen.agentchat.contrib.rag.RAGQueryEngine.

    This engine initializes a vector database, builds an index from input documents,
    and allows querying using the chat engine interface.

    Attributes:
        vector_db (MongoDBAtlasVectorDB): The MongoDB vector database instance.
        vector_search_engine (MongoDBAtlasVectorSearch): The vector search engine.
        storage_context (StorageContext): The storage context for the vector store.
        indexer (Optional[VectorStoreIndex]): The index built from the documents.
    """

    def __init__(  # type: ignore[no-any-unimported]
        self,
        connection_string: str = "",
        database_name: str = "vector_db",
        embedding_function: Optional[Callable[..., Any]] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        index_name: str = "vector_index",
        llm: Union[str, LLM] = "gpt-4o",
    ):
        """
        Initialize the MongoDBQueryEngine.

        Note: The actual connection and creation of the vector database is deferred to
        connect_db (to use an existing collection) or init_db (to create a new collection).
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.index_name = index_name

        # These will be initialized later.
        self.vector_db: Optional[MongoDBAtlasVectorDB] = None
        self.vector_search_engine = None
        self.storage_context = None
        self.index: Optional[VectorStoreIndex] = None  # type: ignore[no-any-unimported]

        self.llm = llm

    def _setup_vector_db(self, overwrite: bool) -> None:
        """
        Helper method to create the vector database, vector search engine, and storage context.

        Args:
            overwrite (bool): If True, create a new collection (overwriting if exists).
                              If False, use an existing collection.
        """
        # Pass the overwrite flag to the factory if supported.
        self.vector_db: MongoDBAtlasVectorDB = VectorDBFactory.create_vector_db(  # type: ignore[assignment, no-redef]
            db_type="mongodb",
            connection_string=self.connection_string,
            database_name=self.database_name,
            index_name=self.index_name,
            embedding_function=self.embedding_function,
            collection_name=self.collection_name,
            overwrite=overwrite,  # new parameter to control creation behavior
        )
        self.vector_search_engine = MongoDBAtlasVectorSearch(
            mongodb_client=self.vector_db.client,  # type: ignore[union-attr]
            db_name=self.database_name,
            collection_name=self.collection_name,
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_search_engine)
        self.index = VectorStoreIndex.from_vector_store(self.vector_search_engine, storage_context=self.storage_context)

    def connect_db(self, overwrite: bool = False, *args: Any, **kwargs: Any) -> bool:
        """
        Connect to the MongoDB database by issuing a ping using an existing collection.
        This method first checks if the target database and collection exist.
        - If not, it raises an error instructing the user to run init_db.
        - If the collection exists and overwrite is True, it reinitializes the database.
        - Otherwise, it uses the existing collection.

        Returns:
            bool: True if the connection is successful; False otherwise.
        """
        try:
            # Check if the target collection exists.
            client = MongoClient(self.connection_string)
            db = client[self.database_name]
            if self.collection_name not in db.list_collection_names():
                raise ValueError(
                    f"Collection '{self.collection_name}' not found in database '{self.database_name}'. "
                    "Please run init_db to create a new collection."
                )
            # Reinitialize if the caller requested overwrite.
            if overwrite:
                logger.info("Overwriting existing collection as requested.")
                self._setup_vector_db(overwrite=True)
            else:
                self._setup_vector_db(overwrite=False)
            self.vector_db.client.admin.command("ping")  # type: ignore[union-attr]
            logger.info("Connected to MongoDB successfully.")
            return True
        except Exception as error:
            logger.error("Failed to connect to MongoDB: %s", error)
            return False

    def init_db(
        self,
        new_doc_dir: Optional[Union[str, Path]] = None,
        new_doc_paths: Optional[List[Union[str, Path]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """
        Initialize the database by loading documents from the given directory or file paths,
        then building an index. This method is intended for first-time creation of the database,
        so it expects that the collection does not already exist (i.e. overwrite is False).

        Args:
            new_doc_dir (Optional[Union[str, Path]]): Directory containing input documents.
            new_doc_paths (Optional[List[Union[str, Path]]]): List of document paths or URLs.

        Returns:
            bool: True if initialization is successful; False otherwise.
        """
        try:
            # Check if the collection already exists.
            client = MongoClient(self.connection_string)
            db = client[self.database_name]
            if self.collection_name in db.list_collection_names():
                raise ValueError(
                    f"Collection '{self.collection_name}' already exists in database '{self.database_name}'. "
                    "Use connect_db with overwrite=True to reinitialize it."
                )
            # Set up the database without overwriting.
            self._setup_vector_db(overwrite=False)
            self.vector_db.client.admin.command("ping")  # type: ignore[union-attr]
            # Gather document paths.
            document_list: List[Union[str, Path]] = []
            if new_doc_dir:
                document_list.extend(Path(new_doc_dir).glob("**/*"))
            if new_doc_paths:
                document_list.extend(new_doc_paths)

            if not document_list:
                logger.warning("No input documents provided to initialize the database.")
                return False

            documents = SimpleDirectoryReader(input_files=document_list).load_data()
            self.index = VectorStoreIndex.from_documents(documents, storage_context=self.storage_context)
            logger.info("Database initialized with %d documents.", len(documents))
            return True
        except Exception as e:
            logger.error("Failed to initialize the database: %s", e)
            return False

    def add_records(
        self,
        new_doc_dir: Optional[Union[str, Path]] = None,
        new_doc_paths_or_urls: Optional[Union[List[Union[str, Path]], Union[str, Path]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Load, parse, and insert documents into the index.

        This method uses a SentenceSplitter to break documents into chunks before insertion.

        Args:
            new_doc_dir (Optional[Union[str, Path]]): Directory containing input documents.
            new_doc_paths_or_urls (Optional[Union[List[Union[str, Path]], Union[str, Path]]]):
                List of document paths or a single document path/URL.
        """
        # Collect document paths.
        document_list: List[Union[str, Path]] = []
        if new_doc_dir:
            document_list.extend(Path(new_doc_dir).glob("**/*"))
        if new_doc_paths_or_urls:
            if isinstance(new_doc_paths_or_urls, (list, tuple)):
                document_list.extend(new_doc_paths_or_urls)
            else:
                document_list.append(new_doc_paths_or_urls)

        if not document_list:
            logger.warning("No documents found for adding records.")
            return

        try:
            raw_documents = SimpleDirectoryReader(input_files=document_list).load_data()
        except Exception as e:
            logger.error("Error loading documents: %s", e)
            return

        if not raw_documents:
            logger.warning("No document chunks created for insertion.")
            return

        try:
            for doc in raw_documents:
                self.index.insert(doc)  # type: ignore[union-attr]
            logger.info("Inserted %d document chunks successfully.", len(raw_documents))
        except Exception as e:
            logger.error("Error inserting documents into the index: %s", e)

    def query(self, question: str, *args: Any, **kwargs: Any) -> Any:  # type: ignore[no-any-unimported, type-arg]
        """
        Query the index using the given question.

        Args:
            question (str): The query string.
            llm (Union[str, LLM, BaseLanguageModel]): The language model to use.

        Returns:
            Any: The response from the chat engine, or None if an error occurs.
        """
        try:
            response = self.index.as_query_engine(llm=self.llm).query(question)  # type: ignore[union-attr]
            return response
        except Exception as e:
            logger.error("Query failed: %s", e)
            return None
