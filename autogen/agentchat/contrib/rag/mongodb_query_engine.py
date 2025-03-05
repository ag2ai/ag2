# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from autogen.agentchat.contrib.vectordb.base import VectorDBFactory
from autogen.agentchat.contrib.vectordb.mongodb import MongoDBAtlasVectorDB
from autogen.import_utils import optional_import_block, require_optional_import

with optional_import_block():
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.llms.langchain.base import LLM
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
    from pymongo import MongoClient

DEFAULT_COLLECTION_NAME = "docling-parsed-docs"
EMPTY_RESPONSE_TEXT = "Empty Response"  # Indicates that the query did not return any results
EMPTY_RESPONSE_REPLY = "Sorry, I couldn't find any information on that. If you haven't ingested any documents, please try that."  # Default response for queries without results

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@require_optional_import(["pymongo", "llama_index"], "rag")
class MongoDBQueryEngine:
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

    def __init__(  # type: ignore[no-any-unimported]
        self,
        connection_string: str,
        llm: Optional[LLM] = None,
        database_name: Optional[str] = None,
        embedding_function: "Optional[BaseEmbedding]" = None,
        collection_name: Optional[str] = DEFAULT_COLLECTION_NAME,
    ):
        """
        Initialize the MongoDBQueryEngine.

        Note: The actual connection and creation of the vector database is deferred to
        connect_db (to use an existing collection) or init_db (to create a new collection).
        """
        if not connection_string:
            raise ValueError("Connection string is required to connect to MongoDB.")

        self.connection_string = connection_string
        self.database_name = database_name
        self.embedding_function = embedding_function
        self.collection_name = collection_name

        # These will be initialized later.
        self.vector_db: Optional[MongoDBAtlasVectorDB] = None
        self.vector_search_engine = None
        self.storage_context = None
        self.index: Optional[VectorStoreIndex] = None  # type: ignore[no-any-unimported]

        self.llm: LLM = llm or OpenAI(model="gpt-4o", temperature=0.0)  # type: ignore[no-any-unimported]

    def _set_up(self, overwrite: bool) -> None:
        """
        Helper method to create the vector database, vector search engine, and storage context.

        Args:
            overwrite (bool): If True, create a new collection (overwriting if exists). If False, use an existing collection.
        """
        # Pass the overwrite flag to the factory if supported.
        self.vector_db: MongoDBAtlasVectorDB = VectorDBFactory.create_vector_db(  # type: ignore[assignment, no-redef]
            db_type="mongodb",
            connection_string=self.connection_string,
            database_name=self.database_name,
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

    def _check_existing_collection(self) -> bool:
        """
        Check if the collection already exists in the database.

        Returns:
            bool: True if the collection exists; False otherwise.
        """
        client = MongoClient(self.connection_string)
        db = client[self.database_name]
        return self.collection_name in db.list_collection_names()

    def connect_db(self, *args: Any, **kwargs: Any) -> bool:
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
            if not self._check_existing_collection():
                raise ValueError(
                    f"Collection '{self.collection_name}' not found in database '{self.database_name}'. "
                    "Please run init_db to create a new collection."
                )
            # Reinitialize if the caller requested overwrite.
            self._set_up(overwrite=False)
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_search_engine, storage_context=self.storage_context
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
            if self._check_existing_collection():
                logger.warning(
                    f"Collection '{self.collection_name}' already exists in database '{self.database_name}."
                    "Please use connect_db to connect to the existing collection."
                    "Or else this function will overwrite the collection."
                )
            # Set up the database with overwriting.
            self._set_up(overwrite=True)
            self.vector_db.client.admin.command("ping")  # type: ignore[union-attr]
            # Gather document paths.
            documents = self._load_doc(input_dir=new_doc_dir, input_docs=new_doc_paths_or_urls)
            self.index = VectorStoreIndex.from_documents(documents, storage_context=self.storage_context)
            logger.info("Database initialized with %d documents.", len(documents))
            return True
        except Exception as e:
            logger.error("Failed to initialize the database: %s", e)
            return False

    def _validate_query_index(self) -> None:
        """Ensures an index exists"""
        if not hasattr(self, "index"):
            raise Exception("Query index is not initialized. Please call init_db or connect_db first.")

    def _load_doc(  # type: ignore[no-any-unimported]
        self, input_dir: Optional[Union[Path, str]], input_docs: Optional[Sequence[Union[Path, str]]]
    ) -> Sequence["LlamaDocument"]:
        """
        Load documents from a directory and/or a sequence of file paths.

        It uses LlamaIndex's SimpleDirectoryReader that supports multiple file[formats]((https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/#supported-file-types)).

        Args:
            input_dir (Optional[Union[Path, str]]): The directory containing documents to be loaded.
                If provided, all files in the directory will be considered.
            input_docs (Optional[Sequence[Union[Path, str]]]): A sequence of individual file paths to load.
                Each path must point to an existing file.

        Returns:
            A sequence of documents loaded as LlamaDocument objects.

        Raises:
            ValueError: If the specified directory does not exist.
            ValueError: If any provided file path does not exist.
            ValueError: If neither input_dir nor input_docs is provided.
        """
        loaded_documents = []
        if input_dir:
            logger.info(f"Loading docs from directory: {input_dir}")
            if not os.path.exists(input_dir):
                raise ValueError(f"Input directory not found: {input_dir}")
            loaded_documents.extend(SimpleDirectoryReader(input_dir=input_dir).load_data())

        if input_docs:
            for doc in input_docs:
                logger.info(f"Loading input doc: {doc}")
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
        Load, parse, and insert documents into the index.

        This method uses a SentenceSplitter to break documents into chunks before insertion.

        Args:
            new_doc_dir (Optional[Union[str, Path]]): Directory containing input documents.
            new_doc_paths_or_urls (Optional[Union[List[Union[str, Path]], Union[str, Path]]]):
                List of document paths or a single document path/URL.
        """
        self._validate_query_index()
        documents = self._load_doc(input_dir=new_doc_dir, input_docs=new_doc_paths_or_urls)
        for doc in documents:
            self.index.insert(doc)  # type: ignore[union-attr]

    def query(self, question: str, *args: Any, **kwargs: Any) -> Any:  # type: ignore[no-any-unimported, type-arg]
        """
        Query the index using the given question.

        Args:
            question (str): The query string.
            llm (Union[str, LLM, BaseLanguageModel]): The language model to use.

        Returns:
            Any: The response from the chat engine, or None if an error occurs.
        """
        self._validate_query_index()
        self.query_engine = self.index.as_query_engine(llm=self.llm)  # type: ignore[union-attr]
        response = self.query_engine.query(question)

        if str(response) == EMPTY_RESPONSE_TEXT:
            return EMPTY_RESPONSE_REPLY

        return str(response)

    def get_collection_name(self) -> str:
        """
        Get the name of the collection used by the query engine.

        Returns:
            The name of the collection.
        """
        if self.collection_name:
            return self.collection_name
        else:
            raise ValueError("Collection name not set.")


if TYPE_CHECKING:
    from .query_engine import RAGQueryEngine

    def _check_implement_protocol(o: MongoDBQueryEngine) -> RAGQueryEngine:
        return o
