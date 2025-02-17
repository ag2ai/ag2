# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from autogen.import_utils import optional_import_block, require_optional_import

with optional_import_block():
    import chromadb
    import pymongo
    from chromadb.api.models.Collection import Collection
    from chromadb.api.types import EmbeddingFunction
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.llms import LLM
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
    from pymongo import MongoClient

DEFAULT_COLLECTION_NAME = "docling-parsed-docs"

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def load_documents(input_dir: Optional[str], input_docs: Optional[List[str]]) -> List[LlamaDocument]:  # type: ignore[no-any-unimported]
    """
    Load documents from a directory and/or a list of file paths.

    Description:
        This helper function loads documents using the SimpleDirectoryReader from llama_index.
        It checks the existence of the provided directory or files, logs the process, and raises
        a ValueError if the inputs are invalid.

    Args:
        input_dir (Optional[str]): Path to the directory containing documents.
        input_docs (Optional[List[str]]): List of document file paths.

    Returns:
        List[LlamaDocument]: A list of loaded LlamaDocument objects.

    Raises:
        ValueError: If the input directory does not exist, if any document file does not exist,
                    or if neither input_dir nor input_docs is provided.
    """
    logger.info("Starting load_documents.")
    documents: List[LlamaDocument] = []  # type: ignore[no-any-unimported]
    if input_dir:
        logger.info(f"Loading docs from directory: {input_dir}")
        if not os.path.exists(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            raise ValueError(f"Input directory not found: {input_dir}")
        documents.extend(SimpleDirectoryReader(input_dir=input_dir).load_data())
    if input_docs:
        for doc in input_docs:
            logger.info(f"Loading input doc: {doc}")
            if not os.path.exists(doc):
                logger.error(f"Document file not found: {doc}")
                raise ValueError(f"Document file not found: {doc}")
        documents.extend(SimpleDirectoryReader(input_files=input_docs).load_data())
    if not input_dir and not input_docs:
        logger.error("No input directory or docs provided!")
        raise ValueError("No input directory or docs provided!")
    logger.info("Completed loading documents.")
    return documents


class BaseDoclingQueryEngine(ABC):
    """
    Abstract base class for Docling query engines.

    Description:
        Provides the standard interface for initializing databases, querying, adding documents,
        and retrieving collection names.

    Methods:
        init_db: Abstract method to load documents and create the vector index.
        query: Abstract method to process a natural language query.
        add_docs: Abstract method to add additional documents to the index.
        get_collection_name: Abstract method to obtain the collection/index name.
    """

    def __init__(self, llm: Optional[LLM] = None) -> None:  # type: ignore[no-any-unimported]
        """
        Initialize the query engine with a specific language model.

        Args:
            llm (Optional[LLM]): Optional language model to be used for querying. Defaults to OpenAI GPT-4.
        """
        self.llm: LLM = llm or OpenAI(model="gpt-4o", temperature=0.0)  # type: ignore[no-any-unimported]
        self.index = None

    @abstractmethod
    def init_db(
        self,
        input_dir: Optional[str] = None,
        input_doc_paths: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the database.

        Description:
            Loads documents from a directory or from file paths and creates the vector index.
            This method must be implemented by subclasses.

        Args:
            input_dir (Optional[str]): Directory of input documents.
            input_doc_paths (Optional[List[str]]): List of input document file paths.
            collection_name (Optional[str]): Optional name of the collection/index.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def query(self, question: str) -> str:
        """
        Process a natural language query.

        Args:
            question (str): The query string.

        Returns:
            str: The answer resulting from the query.

        Raises:
            ValueError: If the index is not initialized.
        """
        pass

    @abstractmethod
    def add_docs(
        self,
        new_doc_dir: Optional[str] = None,
        new_doc_paths: Optional[List[str]] = None,
    ) -> None:
        """
        Add additional documents to the existing index.

        Args:
            new_doc_dir (Optional[str]): Directory containing new documents.
            new_doc_paths (Optional[List[str]]): List of new document file paths.
        """
        pass

    @abstractmethod
    def get_collection_name(self) -> Optional[str]:
        """
        Retrieve the name of the collection/index if available.

        Returns:
            Optional[str]: The collection or index name, or None if not initialized.
        """
        pass


@require_optional_import(["chromadb", "llama_index"], "rag")
class DoclingChromaMdQueryEngine(BaseDoclingQueryEngine):
    """
    Query engine using Chromadb to store document embeddings with LlamaIndex for querying.

    Description:
        This engine utilizes Chromadb for persistent storage and manages document embeddings.
        It creates or retrieves a Chromadb collection and constructs a vector index with loaded documents.

    Args:
        db_path (Optional[str]): Path to the Chromadb database directory.
        embedding_function (Optional[EmbeddingFunction[Any]]): Custom embedding function.
        metadata (Optional[dict[str, Any]]): Metadata configuration for the embedding engine.
        llm (Optional[LLM]): Optional language model to use for queries.

    Attributes:
        collection_name (Optional[str]): Name of the Chromadb collection.
        collection (Optional[Collection]): The Chromadb collection instance.
        vector_store: Underlying vector store.
        storage_context: Storage context for the vector index.
    """

    def __init__(  # type: ignore[no-any-unimported]
        self,
        db_path: Optional[str] = None,
        embedding_function: Optional[EmbeddingFunction[Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        llm: Optional[LLM] = None,
    ) -> None:  # type: ignore[no-any-unimported]
        """
        Initialize the DoclingChromaMdQueryEngine.

        Args:
            db_path (Optional[str]): Optional path for Chromadb persistent storage.
            embedding_function (Optional[EmbeddingFunction[Any]]): Optional embedding function.
            metadata (Optional[dict[str, Any]]): Optional metadata settings for the vector store.
            llm (Optional[LLM]): Optional language model.
        """
        logger.info("Initializing DoclingChromaMdQueryEngine.")
        super().__init__(llm=llm)
        self.embedding_function: EmbeddingFunction[Any] = embedding_function or DefaultEmbeddingFunction()  # type: ignore[assignment]
        self.metadata: dict[str, Any] = metadata or {
            "hnsw:space": "ip",
            "hnsw:construction_ef": 30,
            "hnsw:M": 32,
        }
        self.client = chromadb.PersistentClient(path=db_path or "./chroma")
        self.collection_name: Optional[str] = None
        self.collection: Optional[Collection] = None
        self.vector_store = None
        self.storage_context = None

    def init_db(
        self,
        input_dir: Optional[str] = None,
        input_doc_paths: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the Chromadb database and index.

        Args:
            input_dir (Optional[str]): Directory containing documents.
            input_doc_paths (Optional[List[str]]): List of document file paths.
            collection_name (Optional[str]): Optional name for the Chromadb collection.

        Raises:
            ValueError: If the documents cannot be loaded.
        """
        logger.info("Initializing database for DoclingChromaMdQueryEngine.")
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata=self.metadata,
            get_or_create=True,
        )
        logger.info(f"Chroma collection '{self.collection_name}' created or retrieved.")
        documents = load_documents(input_dir, input_doc_paths)
        logger.info("Documents loaded successfully.")
        self.index = self._create_index(documents)
        logger.info("Vector index created with input documents.")

    def _create_index(self, docs: List[LlamaDocument]) -> VectorStoreIndex:  # type: ignore[no-any-unimported]
        """
        Create a vector index from the provided documents.

        Args:
            docs (List[LlamaDocument]): List of documents to index.

        Returns:
            VectorStoreIndex: The initialized vector index.
        """
        logger.info("Creating vector index for Chromadb.")
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        return VectorStoreIndex.from_documents(docs, storage_context=self.storage_context)

    def query(self, question: str) -> str:
        """
        Process a natural language query using the vector index.

        Args:
            question (str): The query string.

        Returns:
            str: The response from the query engine.

        Raises:
            ValueError: If the vector index is not initialized.
        """
        logger.info(f"Received query: {question}")
        if self.index is None:
            logger.error("Query requested before index initialization.")
            raise ValueError("Index not initialized. Call init_db() first.")
        query_engine = self.index.as_query_engine(llm=self.llm)
        response = query_engine.query(question)
        logger.info("Query processed successfully.")
        return str(response)

    def add_docs(
        self,
        new_doc_dir: Optional[str] = None,
        new_doc_paths: Optional[List[str]] = None,
    ) -> None:
        """
        Add additional documents to the existing vector index.

        Args:
            new_doc_dir (Optional[str]): Directory containing new documents.
            new_doc_paths (Optional[List[str]]): List of new document file paths.

        Raises:
            ValueError: If the vector index is not initialized.
        """
        logger.info("Adding documents to the existing Chromadb index.")
        if self.index is None:
            logger.error("Attempted to add docs before index initialization.")
            raise ValueError("Index not initialized. Call init_db() first.")
        new_docs = load_documents(new_doc_dir, new_doc_paths)
        for doc in new_docs:
            self.index.insert(doc)
            logger.info("Inserted a new document into the index.")

    def get_collection_name(self) -> Optional[str]:
        """
        Get the Chromadb collection name.

        Returns:
            Optional[str]: The collection name if the index is initialized; otherwise, None.
        """
        return self.collection_name if self.index else None


@require_optional_import(["pymongo", "llama_index"], "rag")
class DoclingMongoAtlasMdQueryEngine(BaseDoclingQueryEngine):
    """
    Query engine using MongoDB Atlas to store document embeddings with LlamaIndex for querying.

    Description:
        Uses MongoDB Atlas as the backend to store embeddings and construct a vector index.
        It connects to MongoDB Atlas, creates a vector search index if necessary, and loads
        documents via llama_index.

    Args:
        connection_string (str): MongoDB Atlas connection string.
        database_name (str): Name of the database.
        collection_name (str): Name of the collection.
        vector_index_name (str): Name of the vector index.
        llm (Optional[LLM]): Optional language model for querying.
        **kwargs: Additional configuration parameters.
    """

    def __init__(  # type: ignore[no-any-unimported]
        self,
        connection_string: str = "mongodb+srv://<username>:<password>@<host>?retryWrites=true&w=majority",
        database_name: str = "vector_db",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_index_name: str = "vector_index",
        llm: Optional[LLM] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the DoclingMongoAtlasMdQueryEngine.

        Args:
            connection_string (str): MongoDB Atlas connection string.
            database_name (str): Name of the MongoDB database.
            collection_name (str): Collection name for storing embeddings.
            vector_index_name (str): Name of the vector index.
            llm (Optional[LLM]): Optional language model.
            **kwargs: Other optional parameters for configuration.
        """
        logger.info("Initializing DoclingMongoAtlasMdQueryEngine.")
        super().__init__(llm=llm)
        self.client = self._get_mongo_client(connection_string)
        self.collection_name = collection_name
        self.collection = MongoDBAtlasVectorSearch(
            self.client,
            db_name=database_name,
            collection_name=self.collection_name,
            vector_index_name=vector_index_name,
            **kwargs,
        )
        self.vector_index_name = vector_index_name
        self.vector_store = None
        self.storage_context = None
        self.index = None
        try:
            self.collection.create_vector_search_index(dimensions=1536, path="embedding", similarity="cosine")
            logger.info("MongoDB Atlas vector search index created.")
        except Exception as e:
            logger.warning(f"Vector search index may already exist or could not be created: {e}")

    def _get_mongo_client(self, connection_string: str) -> MongoClient:  # type: ignore[no-any-unimported]
        """
        Establish a connection to MongoDB Atlas.

        Args:
            connection_string (str): The connection string for MongoDB Atlas.

        Returns:
            MongoClient: An instance of MongoClient connected to MongoDB Atlas.

        Raises:
            pymongo.errors.ConnectionFailure: If the connection fails.
        """
        logger.info("Attempting to connect to MongoDB Atlas.")
        try:
            client = MongoClient(connection_string)
            logger.info("Connected to MongoDB Atlas.")
            return client
        except pymongo.errors.ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB Atlas: {e}")
            raise e

    def init_db(
        self,
        input_dir: Optional[str] = None,
        input_doc_paths: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the MongoDB Atlas database and vector index.

        Args:
            input_dir (Optional[str]): Directory containing documents.
            input_doc_paths (Optional[List[str]]): List of document file paths.
            collection_name (Optional[str]): Optional new collection name.

        Raises:
            ValueError: If the documents cannot be loaded.
        """
        logger.info("Initializing database for DoclingMongoAtlasMdQueryEngine.")
        if collection_name:
            self.collection_name = collection_name
        documents = load_documents(input_dir, input_doc_paths)
        logger.info(f"Documents loaded successfully. Total docs loaded: {len(documents)}")
        self.index = self._create_index(documents)
        logger.info("Vector index created with input documents.")

    def _create_index(self, docs: List[LlamaDocument]) -> VectorStoreIndex:  # type: ignore[no-any-unimported]
        """
        Create a vector index from documents using MongoDB Atlas.

        Args:
            docs (List[LlamaDocument]): List of documents to index.

        Returns:
            VectorStoreIndex: The initialized vector index.
        """
        logger.info("Creating vector index for MongoDB Atlas.")
        self.storage_context = StorageContext.from_defaults(vector_store=self.collection)
        index = VectorStoreIndex.from_documents(docs, storage_context=self.storage_context)
        logger.info(f"Index created with {len(docs)} documents.")
        return index

    def query(self, question: str) -> str:
        """
        Process a query using the MongoDB Atlas vector index.

        Args:
            question (str): The query string.

        Returns:
            str: The response from the query engine.

        Raises:
            ValueError: If the vector index is not initialized.
        """
        logger.info(f"Received query: {question}")
        if self.index is None:
            logger.error("Query requested before index initialization.")
            raise ValueError("Index not initialized. Call init_db() first.")
        query_engine = self.index.as_query_engine(llm=self.llm)
        response = query_engine.query(question)
        logger.info("Query processed successfully.")
        return str(response)

    def add_docs(
        self,
        new_doc_dir: Optional[str] = None,
        new_doc_paths: Optional[List[str]] = None,
    ) -> None:
        """
        Add additional documents to the MongoDB Atlas index.

        Args:
            new_doc_dir (Optional[str]): Directory containing new documents.
            new_doc_paths (Optional[List[str]]): List of new document file paths.

        Raises:
            ValueError: If the vector index is not initialized.
        """
        logger.info("Adding documents to the MongoDB Atlas index.")
        if self.index is None:
            logger.error("Attempted to add docs before index initialization.")
            raise ValueError("Index not initialized. Call init_db() first.")
        new_docs = load_documents(new_doc_dir, new_doc_paths)
        for doc in new_docs:
            self.index.insert(doc)
            logger.info("Inserted a new document into the index.")

    def get_collection_name(self) -> Optional[str]:
        """
        Get the name of the MongoDB collection.

        Returns:
            Optional[str]: The collection name if the index is initialized; otherwise, None.
        """
        return self.collection_name if self.index else None


class DoclingQueryEngine:
    """
    Unified interface for querying Docling databases.

    Description:
        This class abstracts the underlying query engine implementation.
        It supports both Chromadb and MongoDB Atlas backends, instantiating the appropriate engine
        based on the provided 'db_type' parameter.

    Example:
        engine = DoclingQueryEngine(
            db_type="chroma",
            db_path="./my_chroma_db",
            embedding_function=my_embedding_function,
            llm=my_llm
        )
        engine.init_db(input_dir="./docs")
        answer = engine.query("What is the summary?")

    Args:
        db_type (str): The type of database ('chroma' or 'mongodb').
        **config (Any): Additional configuration parameters for the chosen engine.
    """

    def __init__(self, db_type: str, **config: Any) -> None:
        """
        Initialize the unified DoclingQueryEngine.

        Args:
            db_type (str): The type of database backend ('chroma', 'chromadb', 'mongodb', 'mongo', or 'atlas').
            **config (Any): Additional configuration parameters.

        Raises:
            ValueError: If an unsupported db_type is provided.
        """
        logger.info(f"Initializing DoclingQueryEngine with db_type: {db_type}")
        db_type_lower = db_type.lower()
        if db_type_lower in ["chroma", "chromadb"]:
            self.engine: BaseDoclingQueryEngine = DoclingChromaMdQueryEngine(**config)
        elif db_type_lower in ["mongodb", "mongo", "atlas"]:
            self.engine = DoclingMongoAtlasMdQueryEngine(**config)
        else:
            logger.error(f"Unsupported db type: {db_type}")
            raise ValueError(f"Unsupported db type '{db_type}'. Supported types are: 'chroma' and 'mongodb'.")

    def init_db(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the underlying database.

        Args:
            *args (Any): Positional arguments for database initialization.
            **kwargs (Any): Keyword arguments for database initialization.
        """
        logger.info("Initializing the underlying database.")
        self.engine.init_db(*args, **kwargs)

    def query(self, question: str) -> str:
        """
        Process a query using the underlying engine.

        Args:
            question (str): The natural language query.

        Returns:
            str: The answer generated by the engine.
        """
        logger.info(f"Processing query through unified interface: {question}")
        return self.engine.query(question)

    def add_docs(self, *args: Any, **kwargs: Any) -> None:
        """
        Add documents to the underlying index.

        Args:
            *args (Any): Positional arguments for adding documents.
            **kwargs (Any): Keyword arguments for adding documents.
        """
        logger.info("Adding documents through unified interface.")
        self.engine.add_docs(*args, **kwargs)

    def get_collection_name(self) -> Optional[str]:
        """
        Retrieve the collection name from the underlying engine.

        Returns:
            Optional[str]: The collection name if present, else None.
        """
        return self.engine.get_collection_name()
