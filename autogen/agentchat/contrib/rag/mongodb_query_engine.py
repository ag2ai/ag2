# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union

from autogen.agentchat.contrib.vectordb.base import VectorDBFactory
from autogen.agentchat.contrib.vectordb.mongodb import MongoDBAtlasVectorDB
from autogen.doc_utils import export_module
from autogen.import_utils import optional_import_block, require_optional_import

with optional_import_block():
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.llms.langchain.base import LLM
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
    from pymongo import MongoClient
    from sentence_transformers import SentenceTransformer

__all__ = ["MongoDBQueryEngine"]

DEFAULT_COLLECTION_NAME = "docling-parsed-docs"
EMPTY_RESPONSE_TEXT = "Empty Response"  # Indicates that the query did not return any results
EMPTY_RESPONSE_REPLY = "Sorry, I couldn't find any information on that. If you haven't ingested any documents, please try that."  # Default response for queries without results

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@require_optional_import(["pymongo", "llama_index", "sentence_transformers"], "rag")
@export_module("autogen.agentchat.contrib.rag")
class MongoDBQueryEngine:
    """
    A query engine backed by MongoDB Atlas that supports document insertion and querying.

    This engine initializes a vector database, builds an index from input documents,
    and allows querying using the chat engine interface.

    Attributes:
        vector_db (MongoDBAtlasVectorDB): The MongoDB vector database instance.
        vector_search_engine (MongoDBAtlasVectorSearch): The vector search engine.
        storage_context (StorageContext): The storage context for the vector store.
        index (Optional[VectorStoreIndex]): The index built from the documents.
    """

    def __init__(  # type: ignore[no-any-unimported]
        self,
        connection_string: str,
        llm: Optional["LLM"] = None,
        database_name: Optional[str] = None,
        embedding_function: Optional[Union["BaseEmbedding", Callable[..., Any]]] = None,  # type: ignore[type-arg]
        embedding_model: Optional[Union["BaseEmbedding", str]] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initializes a MongoDBQueryEngine instance.

        Args:
            connection_string (str): Connection string used to connect to MongoDB.
            llm (Optional[LLM]): Language model for querying. Defaults to an OpenAI model if not provided.
            database_name (Optional[str]): Name of the MongoDB database.
            embedding_function (Optional[Union["BaseEmbedding", Callable[..., Any]]]): Custom embedding function. If None (default),
                defaults to SentenceTransformer encoding.
            embedding_model (Optional[Union["BaseEmbedding", str]]): Embedding model identifier or instance. If None (default),
                "local:all-MiniLM-L6-v2" will be used.
            collection_name (Optional[str]): Name of the MongoDB collection. If None (default), `DEFAULT_COLLECTION_NAME` will be used.

        Raises:
            ValueError: If no connection string is provided.
        """
        if not connection_string:
            raise ValueError("Connection string is required to connect to MongoDB.")

        self.connection_string = connection_string
        # ToDo: Is it okay if database_name is None?
        self.database_name = database_name
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
        self.llm: LLM = llm or OpenAI(model="gpt-4o", temperature=0.0)  # type: ignore[no-any-unimported]
        self.embedding_model = embedding_model or "local:all-MiniLM-L6-v2"  # type: ignore[no-any-unimported]
        self.embedding_function = embedding_function or SentenceTransformer("all-MiniLM-L6-v2").encode

        # These will be initialized later.
        self.vector_db: Optional[MongoDBAtlasVectorDB] = None
        self.vector_search_engine: Optional["MongoDBAtlasVectorSearch"] = None  # type: ignore[no-any-unimported]
        self.storage_context: Optional["StorageContext"] = None  # type: ignore[no-any-unimported]
        self.index: Optional[VectorStoreIndex] = None  # type: ignore[no-any-unimported]

    def _set_up(self, overwrite: bool) -> None:
        """
        Sets up the MongoDB vector database, search engine, and storage context.

        This method initializes the vector database using the provided connection details,
        creates a vector search engine instance, and sets the storage context for indexing.

        Args:
            overwrite (bool): Flag indicating whether to overwrite the existing collection.
        """
        logger.info("Setting up the database.")
        self.vector_db: MongoDBAtlasVectorDB = VectorDBFactory.create_vector_db(  # type: ignore[assignment, no-redef]
            db_type="mongodb",
            connection_string=self.connection_string,
            database_name=self.database_name,
            embedding_function=self.embedding_function,
            collection_name=self.collection_name,
            overwrite=overwrite,  # new parameter to control creation behavior
        )
        logger.info("Vector database created.")
        self.vector_search_engine = MongoDBAtlasVectorSearch(
            mongodb_client=self.vector_db.client,  # type: ignore[union-attr]
            db_name=self.database_name,
            collection_name=self.collection_name,
        )
        logger.info("Vector search engine created.")
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_search_engine)

    def _check_existing_collection(self) -> bool:
        """
        Checks if the specified collection exists in the MongoDB database.

        Returns:
            bool: True if the collection exists; False otherwise.
        """
        client: "MongoClient[Any]" = MongoClient(self.connection_string)  # type: ignore[no-any-unimported]
        db = client[self.database_name]  # type: ignore[index]
        return self.collection_name in db.list_collection_names()

    def connect_db(self, *args: Any, **kwargs: Any) -> bool:
        """
        Connects to the MongoDB database and initializes the query index from the existing collection.

        This method verifies the existence of the collection, sets up the database connection,
        builds the vector store index, and pings the MongoDB server.

        Returns:
            bool: True if connection is successful; False otherwise.
        """
        try:
            # Check if the target collection exists.
            if not self._check_existing_collection():
                raise ValueError(
                    f"Collection '{self.collection_name}' not found in database '{self.database_name}'. "
                    "Please run init_db to create a new collection."
                )
            # Reinitialize without overwriting the existing collection.
            self._set_up(overwrite=False)

            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_search_engine,  # type: ignore[arg-type]
                storage_context=self.storage_context,
                embed_model=self.embedding_model,
            )

            self.vector_db.client.admin.command("ping")  # type: ignore[union-attr]
            logger.info("Connected to MongoDB successfully.")
            return True
        except Exception as error:
            logger.error("Failed to connect to MongoDB: %s", error)
            return False

    def init_db(
        self,
        new_doc_dir: Optional[Union[Path, str]] = None,
        new_doc_paths_or_urls: Optional[Sequence[Union[Path, str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """
        Initializes the MongoDB database by creating or overwriting the collection and indexing documents.

        This method loads documents from a directory or provided file paths, sets up the database (optionally
        overwriting any existing collection), builds the vector store index, and inserts the documents.

        Args:
            new_doc_dir (Optional[Union[Path, str]]): Directory containing documents to be indexed.
            new_doc_paths_or_urls (Optional[Sequence[Union[Path, str]]]): List of file paths or URLs for documents.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            bool: True if the database is successfully initialized; False otherwise.
        """
        try:
            # Check if the collection already exists.
            if self._check_existing_collection():
                logger.warning(
                    f"Collection '{self.collection_name}' already exists in database '{self.database_name}'. "
                    "Please use connect_db to connect to the existing collection or use init_db to overwrite it."
                )
            # Set up the database with overwriting.
            self._set_up(overwrite=True)
            self.vector_db.client.admin.command("ping")  # type: ignore[union-attr]
            # Gather document paths.
            logger.info("Setting up the database with existing collection.")
            documents = self._load_doc(input_dir=new_doc_dir, input_docs=new_doc_paths_or_urls)
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_search_engine,  # type: ignore[arg-type]
                storage_context=self.storage_context,
                embed_model=self.embedding_model,
            )
            for doc in documents:
                self.index.insert(doc)
            logger.info("Database initialized with %d documents.", len(documents))
            return True
        except Exception as e:
            logger.error("Failed to initialize the database: %s", e)
            return False

    def _validate_query_index(self) -> None:
        """
        Validates that the query index is initialized.

        Raises:
            Exception: If the query index is not initialized.
        """
        if not hasattr(self, "index"):
            raise Exception("Query index is not initialized. Please call init_db or connect_db first.")

    def _load_doc(  # type: ignore[no-any-unimported]
        self, input_dir: Optional[Union[Path, str]], input_docs: Optional[Sequence[Union[Path, str]]]
    ) -> Sequence["LlamaDocument"]:
        """
        Loads documents from a directory or a list of file paths.

        Args:
            input_dir (Optional[Union[Path, str]]): Directory from which to load documents.
            input_docs (Optional[Sequence[Union[Path, str]]]): List of document file paths or URLs.

        Returns:
            Sequence[LlamaDocument]: A sequence of loaded LlamaDocument objects.

        Raises:
            ValueError: If the input directory or any specified document file does not exist.
        """
        loaded_documents = []
        if input_dir:
            logger.info("Loading docs from directory: %s", input_dir)
            if not os.path.exists(input_dir):
                raise ValueError(f"Input directory not found: {input_dir}")
            loaded_documents.extend(SimpleDirectoryReader(input_dir=input_dir).load_data())

        if input_docs:
            for doc in input_docs:
                logger.info("Loading input doc: %s", doc)
                if not os.path.exists(doc):
                    raise ValueError(f"Document file not found: {doc}")
            loaded_documents.extend(SimpleDirectoryReader(input_files=input_docs).load_data())  # type: ignore[arg-type]
        if not input_dir and not input_docs:
            raise ValueError("No input directory or docs provided!")

        return loaded_documents

    def add_docs(
        self,
        new_doc_dir: Optional[Union[Path, str]] = None,
        new_doc_paths_or_urls: Optional[Sequence[Union[Path, str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Adds new documents to the existing vector store index.

        This method validates that the index exists, loads documents from the specified directory or file paths,
        and inserts them into the vector store index.

        Args:
            new_doc_dir (Optional[Union[Path, str]]): Directory containing new documents.
            new_doc_paths_or_urls (Optional[Sequence[Union[Path, str]]]): List of file paths or URLs for new documents.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.
        """
        self._validate_query_index()
        documents = self._load_doc(input_dir=new_doc_dir, input_docs=new_doc_paths_or_urls)
        for doc in documents:
            self.index.insert(doc)  # type: ignore[union-attr]

    def query(self, question: str, *args: Any, **kwargs: Any) -> Any:  # type: ignore[no-any-unimported, type-arg]
        """
        Queries the indexed documents using the provided question.

        This method validates that the query index is initialized, creates a query engine from the vector store index,
        and executes the query. If the response is empty, a default reply is returned.

        Args:
            question (str): The query question.
            args (Any): Additional positional arguments.
            kwargs (Any): Additional keyword arguments.

        Returns:
            Any: The query response as a string, or a default reply if no results are found.
        """
        self._validate_query_index()
        self.query_engine = self.index.as_query_engine(llm=self.llm)  # type: ignore[union-attr]
        response = self.query_engine.query(question)

        if str(response) == EMPTY_RESPONSE_TEXT:
            return EMPTY_RESPONSE_REPLY

        return str(response)

    def get_collection_name(self) -> str:
        """
        Retrieves the name of the MongoDB collection.

        Returns:
            str: The collection name.

        Raises:
            ValueError: If the collection name is not set.
        """
        if self.collection_name:
            return self.collection_name
        else:
            raise ValueError("Collection name not set.")


if TYPE_CHECKING:
    from .query_engine import RAGQueryEngine

    def _check_implement_protocol(o: MongoDBQueryEngine) -> RAGQueryEngine:
        return o
