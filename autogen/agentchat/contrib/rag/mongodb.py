# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from autogen.agentchat.contrib.rag.query_engine import VectorDbQueryEngine
from autogen.agentchat.contrib.vectordb.base import VectorDBFactory
from autogen.agentchat.contrib.vectordb.mongodb import MongoDBAtlasVectorDB
from autogen.import_utils import optional_import_block, require_optional_import

with optional_import_block():
    from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

DEFAULT_COLLECTION_NAME = "docling-parsed-docs"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@require_optional_import(["pymongo", "llama_index"], "rag")
class MongoDBQueryEngine(VectorDbQueryEngine):
    """
    A query engine backed by MongoDB Atlas that supports document insertion and querying.

    This engine initializes a vector database, builds an index from input documents,
    and allows querying using the chat engine interface.

    Attributes:
        vector_db (MongoDBAtlasVectorDB): The MongoDB vector database instance.
        vector_search_engine (MongoDBAtlasVectorSearch): The vector search engine.
        storage_context (StorageContext): The storage context for the vector store.
        indexer (Optional[VectorStoreIndex]): The index built from the documents.
    """

    def __init__(
        self,
        connection_string: str = "",
        database_name: str = "vector_db",
        embedding_function: Optional[Callable[..., Any]] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        index_name: str = "vector_index",
    ):
        """
        Initialize the MongoDBQueryEngine.

        Args:
            connection_string (str): The MongoDB connection string.
            database_name (str): The name of the database to use.
            embedding_function (Optional[Callable[..., Any]]): The function to compute embeddings.
            collection_name (str): The name of the collection to use.
            index_name (str): The name of the vector index.
        """
        super().__init__()
        self.vector_db: MongoDBAtlasVectorDB = VectorDBFactory.create_vector_db(  # type: ignore[assignment]
            db_type="mongodb",
            connection_string=connection_string,
            database_name=database_name,
            index_name=index_name,
            embedding_function=embedding_function,
            collection_name=collection_name,
        )
        self.vector_search_engine = MongoDBAtlasVectorSearch(
            mongodb_client=self.vector_db.client,
            db_name=database_name,
            collection_name=collection_name,
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_search_engine)
        self.indexer: Optional[VectorStoreIndex] = None  # type: ignore[no-any-unimported]

    def connect_db(self, *args, **kwargs) -> bool:  # type: ignore[no-untyped-def]
        """
        Connect to the MongoDB database by issuing a ping.

        Returns:
            bool: True if the connection is successful; False otherwise.
        """
        try:
            self.vector_db.client.admin.command("ping")
            logger.info("Connected to MongoDB successfully.")
            return True
        except Exception as error:
            logger.error("Failed to connect to MongoDB: %s", error)
            return False

    def init_db(  # type: ignore[no-untyped-def]
        self,
        new_doc_dir: Optional[Union[str, Path]] = None,
        new_doc_paths: Optional[List[Union[str, Path]]] = None,
        *args,
        **kwargs,
    ) -> bool:
        """
        Initialize the database by loading documents from the given directory or file paths,
        then building an index.

        Args:
            new_doc_dir (Optional[Union[str, Path]]): Directory containing input documents.
            new_doc_paths (Optional[List[Union[str, Path]]]): List of document paths or URLs.

        Returns:
            bool: True if initialization is successful; False otherwise.
        """
        if not self.connect_db():
            return False

        # Gather document paths.
        document_list: List[Union[str, Path]] = []
        if new_doc_dir:
            document_list.extend(Path(new_doc_dir).glob("**/*"))
        if new_doc_paths:
            document_list.extend(new_doc_paths)

        if not document_list:
            logger.warning("No input documents provided to initialize the database.")
            return False

        try:
            documents = SimpleDirectoryReader(input_files=document_list).load_data()
            self.indexer = VectorStoreIndex.from_documents(documents, storage_context=self.storage_context)
            logger.info("Database initialized with %d documents.", len(documents))
            return True
        except Exception as e:
            logger.error("Failed to initialize the database: %s", e)
            return False

    def add_records(  # type: ignore[no-untyped-def, override]
        self,
        new_doc_dir: Optional[Union[str, Path]] = None,
        new_doc_paths_or_urls: Optional[Union[List[Union[str, Path]], Union[str, Path]]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 300,
        *args,
        **kwargs,
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

        node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        doc_chunks: List[Document] = []  # type: ignore[no-any-unimported]

        for document in raw_documents:
            try:
                # Split the document text into chunks.
                chunks = node_parser.split_text(document.text)
                metadata = document.metadata
                doc_id = document.id_  # Ensure this is not a tuple.
                for chunk in chunks:
                    new_doc = Document(text=chunk, id_=doc_id, metadata=metadata)
                    doc_chunks.append(new_doc)
            except Exception as e:
                logger.error("Error parsing document %s: %s", document.id_, e)

        if not doc_chunks:
            logger.warning("No document chunks created for insertion.")
            return

        # Insert document chunks using the indexer.
        try:
            for doc in doc_chunks:
                self.indexer.insert(doc)  # type: ignore[union-attr]
            logger.info("Inserted %d document chunks successfully.", len(doc_chunks))
        except Exception as e:
            logger.error("Error inserting documents into the index: %s", e)

    def query(self, question: str, *args, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        """
        Query the index using the given question.

        Args:
            question (str): The query string.

        Returns:
            Any: The response from the chat engine, or None if an error occurs.
        """
        try:
            response = self.indexer.as_chat_engine().query(question)  # type: ignore[union-attr]
            return response
        except Exception as e:
            logger.error("Query failed: %s", e)
            return None
