# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any, Optional

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


@require_optional_import(["chromadb", "llama_index"], "rag")
class DoclingChromaMdQueryEngine:
    """DoclingChromaMdQueryEngine leverages ChromaDB and LlamaIndex to index and query documents.

    Description:
        Persists document embeddings in a named Chromadb collection and creates a vector index using LlamaIndex.
        This index is then used to efficiently retrieve documents and generate answers to natural language queries.

    Args:
        db_path (Optional[str]): Path where Chromadb stores its persistent data. Defaults to "./chroma" if not provided.
        embedding_function (Optional[EmbeddingFunction[Any]]): Function to convert text into embeddings.
        metadata (Optional[dict[str, Any]]): Configuration parameters for the Chromadb collection.
        llm (Optional[LLM]): Language model for query processing. Defaults to OpenAI with specific settings.
    """

    def __init__(  # type: ignore
        self,
        db_path: Optional[str] = None,
        embedding_function: "Optional[EmbeddingFunction[Any]]" = None,
        metadata: Optional[dict[str, Any]] = None,
        llm: Optional["LLM"] = None,
    ) -> None:
        """Initializes the DoclingChromaMdQueryEngine.

        Description:
            Sets up default values, initializes the persistent Chromadb client,
            and configures the language model and embedding function.

        Args:
            db_path (Optional[str]): Filesystem path for persistent data.
            embedding_function (Optional[EmbeddingFunction[Any]]): Embedding converter function.
            metadata (Optional[dict[str, Any]]): Collection configuration parameters.
            llm (Optional[LLM]): Language model for querying.

        Returns:
            None
        """
        logger.info("Initializing DoclingChromaMdQueryEngine")
        self.llm: LLM = llm or OpenAI(model="gpt-4o", temperature=0.0)  # type: ignore[no-any-unimported]
        self.embedding_function: EmbeddingFunction[Any] = embedding_function or DefaultEmbeddingFunction()  # type: ignore[no-any-unimported,assignment]
        self.metadata: dict[str, Any] = metadata or {
            "hnsw:space": "ip",
            "hnsw:construction_ef": 30,
            "hnsw:M": 32,
        }

        self.client = chromadb.PersistentClient(path=db_path or "./chroma")

    def init_db(
        self,
        input_dir: Optional[str] = None,
        input_doc_paths: Optional[list[str]] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """Initializes the Chromadb collection and vector index by loading documents.

        Description:
            Creates or retrieves a Chromadb collection, loads documents from the specified directory or file paths,
            and builds the vector index for efficient retrieval.

        Args:
            input_dir (Optional[str]): Directory path containing Docling-parsed Markdown files.
            input_doc_paths (Optional[list[str]]): List of file paths for individual documents.
            collection_name (Optional[str]): Custom name for the Chromadb collection.

        Returns:
            None

        Raises:
            ValueError: If no documents are provided.
        """
        logger.info("Starting database initialization in DoclingChromaMdQueryEngine")
        input_dir = input_dir or ""
        input_doc_paths = input_doc_paths or []
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME

        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata=self.metadata,
            get_or_create=True,  # If collection already exists, get the collection
        )
        logger.info(f"Collection {collection_name} was created in the database.")

        documents = self._load_doc(input_dir, input_doc_paths)
        logger.info("Documents are loaded successfully.")

        self.index = self._create_index(self.collection, documents)
        logger.info("VectorDB index was created with input documents")

    def query(self, question: str) -> str:
        """Queries the vector index using a natural language question.

        Description:
            Constructs a query engine from the vector index and retrieves a response by querying the indexed documents.

        Args:
            question (str): A natural language query string.

        Returns:
            str: Response generated by the language model.
        """
        logger.info(f"Processing query: {question}")
        self.query_engine = self.index.as_query_engine(llm=self.llm)
        response = self.query_engine.query(question)

        return str(response)

    def add_docs(self, new_doc_dir: Optional[str] = None, new_doc_paths: Optional[list[str]] = None) -> None:
        """Adds additional documents to the existing vector index.

        Description:
            Loads new documents from a directory or list of file paths and inserts each document into the index.

        Args:
            new_doc_dir (Optional[str]): Directory containing extra documents.
            new_doc_paths (Optional[list[str]]): List of additional document file paths.

        Returns:
            None
        """
        logger.info("Adding new documents to the index")
        new_doc_dir = new_doc_dir or ""
        new_doc_paths = new_doc_paths or []
        new_docs = self._load_doc(input_dir=new_doc_dir, input_docs=new_doc_paths)
        for doc in new_docs:
            self.index.insert(doc)

    def _load_doc(  # type: ignore
        self, input_dir: Optional[str], input_docs: Optional[list[str]]
    ) -> list["LlamaDocument"]:
        """Loads documents from a directory and/or list of file paths.

        Description:
            Reads Docling-parsed Markdown files using SimpleDirectoryReader. Validates that the directory
            and files exist before loading.

        Args:
            input_dir (Optional[str]): Directory containing documents.
            input_docs (Optional[list[str]]): List of document file paths.

        Returns:
            list[LlamaDocument]: A list of loaded documents.

        Raises:
            ValueError: If the provided directory or file paths do not exist, or if both are missing.
        """
        logger.info("Loading documents via _load_doc")
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
            loaded_documents.extend(SimpleDirectoryReader(input_files=input_docs).load_data())

        if not input_dir and not input_docs:
            raise ValueError("No input directory or docs provided!")

        return loaded_documents

    def _create_index(  # type: ignore
        self, collection: "Collection", docs: list["LlamaDocument"]
    ) -> "VectorStoreIndex":
        """Creates a vector index using a Chromadb collection.

        Description:
            Wraps the Chromadb collection and uses the StorageContext and VectorStoreIndex from LlamaIndex
            to build a queryable document index.

        Args:
            collection (Collection): Chromadb collection with document embeddings.
            docs (list[LlamaDocument]): List of documents to index.

        Returns:
            VectorStoreIndex: The constructed vector index.
        """
        logger.info("Creating vector index with loaded documents")
        self.vector_store = ChromaVectorStore(chroma_collection=collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        index = VectorStoreIndex.from_documents(docs, storage_context=self.storage_context)

        return index

    def get_collection_name(self) -> Optional[str]:
        """Retrieves the name of the Chromadb collection.

        Description:
            Returns the name of the collection if the index is initialized.

        Returns:
            Optional[str]: Collection name or None if index does not exist.
        """
        logger.debug("Retrieving collection name")
        if self.index:
            return self.collection_name
        return None


@require_optional_import(["pymongo", "llama_index"], "rag")
class DoclingMongoAtlasMdQueryEngine:
    """DoclingMongoAtlasMdQueryEngine leverages MongoDB Atlas and LlamaIndex to index and query documents.

    Description:
        Stores document embeddings in a MongoDB Atlas collection and creates a vector index using LlamaIndex.
        Facilitates efficient document retrieval and natural language query processing.

    Args:
        connection_string (str): MongoDB Atlas connection string.
        database_name (str): Database name.
        collection_name (str): Name of the collection for embeddings.
        vector_index_name (str): Name of the vector index.
        llm (Optional[LLM]): Language model for query operations.
        **kwargs: Additional parameters for MongoDBAtlasVectorSearch.
    """

    def __init__(  # type: ignore
        self,
        connection_string: str = "mongodb+srv://<username>:<password>@<host>?retryWrites=true&w=majority",
        database_name: str = "vector_db",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_index_name: str = "vector_index",
        llm: Optional["LLM"] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the DoclingMongoAtlasMdQueryEngine and sets up the MongoDB Atlas vector search index.

        Args:
            connection_string (str): MongoDB Atlas connection URI.
            database_name (str): Database name.
            collection_name (str): Name of the collection for embeddings.
            vector_index_name (str): Name of the vector index.
            llm (Optional[LLM]): Language model for query processing.
            **kwargs: Additional parameters for configuring the MongoDBAtlasVectorSearch.

        Returns:
            None

        Raises:
            Exception: If the vector search index cannot be created.
        """
        logger.info("Initializing DoclingMongoAtlasMdQueryEngine")
        self.llm: LLM = llm or OpenAI(model="gpt-4o", temperature=0.0)  # type: ignore[no-any-unimported]
        self.client = self.get_mongo_client(connection_string)
        self.collection_name = collection_name

        self.collection = MongoDBAtlasVectorSearch(
            self.client,
            db_name=database_name,
            collection_name=self.collection_name,
            vector_index_name=vector_index_name,
            **kwargs,
        )
        self.index = None
        self.storage_context = None

        try:
            self.collection.create_vector_search_index(dimensions=1536, path="embedding", similarity="cosine")
            logger.info("Vector search index created for MongoDB Atlas collection.")
        except Exception as e:
            logger.warning(f"Vector search index may already exist or could not be created: {e}")

    def get_mongo_client(self, connection_string: str) -> MongoClient:  # type: ignore[no-any-unimported]
        """Creates and returns a MongoClient instance with the given connection string.

        Args:
            connection_string (str): MongoDB Atlas connection URI.

        Returns:
            MongoClient: The client instance for MongoDB Atlas.

        Raises:
            pymongo.errors.ConnectionFailure: If connection fails.
        """
        logger.info("Attempting to connect to MongoDB Atlas")
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
        input_doc_paths: Optional[list[str]] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """Initializes the MongoDB Atlas vector index with documents.

        Description:
            Loads documents from a directory or list of file paths and creates a vector index in MongoDB Atlas.

        Args:
            input_dir (Optional[str]): Directory containing documents.
            input_doc_paths (Optional[list[str]]): List of document file paths.
            collection_name (Optional[str]): Optionally specify a new collection name.

        Returns:
            None

        Raises:
            ValueError: If document loading fails.
        """
        logger.info("Initializing database in DoclingMongoAtlasMdQueryEngine")
        if collection_name:
            self.collection_name = collection_name

        docs = self._load_doc(input_dir, input_doc_paths)
        logger.info(f"Documents are loaded successfully. Total docs loaded: {len(docs)}")

        self.index = self._create_index(self.collection, docs)
        logger.info("Vector index created with input documents.")

    def query(self, question: str) -> str:
        """Processes a natural language query using the vector index.

        Args:
            question (str): The query string.

        Returns:
            str: Answer generated by the language model.

        Raises:
            ValueError: If the index is not initialized.
        """
        logger.info(f"Running query on MongoAtlas engine: {question}")
        if self.index is None:
            raise ValueError("The index has not been initialized. Call init_db() first.")

        self.query_engine = self.index.as_query_engine(llm=self.llm)
        response = self.query_engine.query(question)
        return str(response)

    def add_docs(self, new_doc_dir: Optional[str] = None, new_doc_paths: Optional[list[str]] = None) -> None:
        """Adds additional documents to the MongoDB Atlas vector index.

        Args:
            new_doc_dir (Optional[str]): Directory for extra documents.
            new_doc_paths (Optional[list[str]]): List of additional document file paths.

        Returns:
            None

        Raises:
            ValueError: If document loading fails.
        """
        logger.info("Adding documents to MongoDB Atlas index")
        new_doc_dir = new_doc_dir or ""
        new_doc_paths = new_doc_paths or []
        new_docs = self._load_doc(input_dir=new_doc_dir, input_docs=new_doc_paths)
        for doc in new_docs:
            self.index.insert(doc)  # type: ignore[attr-defined]
            logger.info("Inserted a new document into the index.")

    def _load_doc(self, input_dir: Optional[str], input_docs: Optional[list[str]]) -> list["LlamaDocument"]:  # type: ignore[no-any-unimported]
        """Loads documents from a directory and/or list of file paths.

        Args:
            input_dir (Optional[str]): Directory containing documents.
            input_docs (Optional[list[str]]): List of document file paths.

        Returns:
            list[LlamaDocument]: A list of loaded documents.

        Raises:
            ValueError: If the directory or any file path does not exist.
        """
        logger.info("Loading documents in MongoAtlas engine")
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
            loaded_documents.extend(SimpleDirectoryReader(input_files=input_docs).load_data())

        if not input_dir and not input_docs:
            raise ValueError("No input directory or docs provided!")

        return loaded_documents

    def _create_index(self, collection: "MongoDBAtlasVectorSearch", docs: list["LlamaDocument"]) -> "VectorStoreIndex":  # type: ignore[no-any-unimported]
        """Creates a vector index using MongoDB Atlas collection and loaded documents.

        Args:
            collection (MongoDBAtlasVectorSearch): The vector search collection.
            docs (list[LlamaDocument]): List of documents to index.

        Returns:
            VectorStoreIndex: The constructed vector index.

        Raises:
            Exception: Propagates any errors during index creation.
        """
        logger.info("Creating vector index in MongoAtlas engine")
        self.storage_context = StorageContext.from_defaults(vector_store=collection)
        index = VectorStoreIndex.from_documents(docs, storage_context=self.storage_context)
        logger.info(f"Index created with {len(docs)} documents.")
        return index

    def get_collection_name(self) -> Optional[str]:
        """Retrieves the MongoDB Atlas collection name used for indexing.

        Returns:
            Optional[str]: Collection name if the index exists; otherwise, None.
        """
        logger.debug("Getting collection name from MongoAtlas engine")
        if self.index:
            return self.collection_name
        return None
