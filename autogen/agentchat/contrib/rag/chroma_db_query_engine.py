# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import
from ..vectordb.base import VectorDBFactory

with optional_import_block():
    from chromadb import HttpClient
    from chromadb.api.types import EmbeddingFunction
    from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, Settings
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.llms import LLM
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.chroma import ChromaVectorStore

__all__ = ["ChromaDBQueryEngine"]

DEFAULT_COLLECTION_NAME = "docling-parsed-docs"
EMPTY_RESPONSE_TEXT = "Empty Response"  # Indicates that the query did not return any results
EMPTY_RESPONSE_REPLY = "Sorry, I couldn't find any information on that. If you haven't ingested any documents, please try that."  # Default response for queries without results


# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@require_optional_import(["chromadb", "llama_index"], "rag")
@export_module("autogen.agentchat.contrib.rag")
class ChromaDBQueryEngine:
    """
    This engine leverages Chromadb to persist document embeddings in a named collection
    and LlamaIndex's VectorStoreIndex to efficiently index and retrieve documents, and generate an answer in response
    to natural language queries. Collection can be regarded as an abstraction of group of documents in the database.

    It expects a Chromadb server to be running and accessible at the specified host and port.
    Refer to this [link](https://docs.trychroma.com/production/containers/docker) for running Chromadb in a Docker container.
    If the host and port are not provided, the engine will create an in-memory ChromaDB client.


    """

    def __init__(  # type: ignore[no-any-unimported]
        self,
        host: Optional[str] = "localhost",
        port: Optional[int] = 8000,
        settings: Optional["Settings"] = None,
        tenant: Optional[str] = None,
        database: Optional[str] = None,
        embedding_function: "Optional[EmbeddingFunction[Any]]" = None,
        metadata: Optional[dict[str, Any]] = None,
        llm: Optional["LLM"] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Initializes the ChromaDBQueryEngine with db_path, metadata, and embedding function and llm.
        Args:
            host: The host address of the ChromaDB server. Default is localhost.
            port: The port number of the ChromaDB server. Default is 8000.
            settings: A dictionary of settings to communicate with the chroma server. Default is None.
            tenant: The tenant to use for this client. Defaults to the default tenant.
            database: The database to use for this client. Defaults to the default database.
            embedding_function: A callable that converts text into vector embeddings. Default embedding uses Sentence Transformers model all-MiniLM-L6-v2.
                For more embeddings that ChromaDB support, please refer to [embeddings](https://docs.trychroma.com/docs/embeddings/embedding-functions)
            metadata: A dictionary containing configuration parameters for the Chromadb collection.
                This metadata is typically used to configure the HNSW indexing algorithm. Defaults to `{"hnsw:space": "ip", "hnsw:construction_ef": 30, "hnsw:M": 32}`
                For more details about the default metadata, please refer to [HNSW configuration](https://cookbook.chromadb.dev/core/configuration/#hnsw-configuration)
            llm: LLM model used by LlamaIndex for query processing.
                 You can find more supported LLMs at [LLM](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/)
            collection_name (str): The unique name for the Chromadb collection. If omitted, a constant name will be used. Populate this to reuse previous ingested data.
        """
        self.llm: LLM = llm or OpenAI(model="gpt-4o", temperature=0.0)  # type: ignore[no-any-unimported]
        if not host or not port:
            logger.warning(
                "Can't connect to remote Chroma client without host or port not. Using an ephemeral, in-memory client."
            )
            self.client = None
        else:
            try:
                self.client = HttpClient(
                    host=host,
                    port=port,
                    settings=settings,
                    tenant=tenant if tenant else DEFAULT_TENANT,  # type: ignore[arg-type, no-any-unimported]
                    database=database if database else DEFAULT_DATABASE,  # type: ignore[arg-type, no-any-unimported]
                )
            except Exception as e:
                raise ValueError(f"Failed to connect to the ChromaDB client: {e}")

        self.db_config = {"client": self.client, "embedding_function": embedding_function, "metadata": metadata}
        self.collection_name = collection_name if collection_name else DEFAULT_COLLECTION_NAME

    def init_db(
        self,
        new_doc_dir: Optional[Union[Path, str]] = None,
        new_doc_paths_or_urls: Optional[Sequence[Union[Path, str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Initialize the database with the input documents or records.
        It overwrites the existing collection in the database.

        It takes the following steps,
        1. Set up ChromaDB and LlamaIndex storage.
        2. insert documents and build indexes upon them.

        Args:
            new_doc_dir: a dir of input documents that are used to create the records in database.
            new_doc_paths_or_urls:
                a sequence of input documents that are used to create the records in database.
                a document can be a path to a file or a url.
            *args: Any additional arguments
            **kwargs: Any additional keyword arguments

        Returns:
            bool: True if initialization is successful

        """

        self._set_up(overwrite=True)
        documents = self._load_doc(input_dir=new_doc_dir, input_docs=new_doc_paths_or_urls)
        self.index = VectorStoreIndex.from_documents(documents=documents, storage_context=self.storage_context)
        return True

    def connect_db(self, *args: Any, **kwargs: Any) -> bool:
        """Connect to the database.
        It does not overwrite the existing collection in the database.

         It takes the following steps,
        1. Set up ChromaDB and LlamaIndex storage.
        2. Create the llamaIndex vector store index for querying or inserting docs later

        Args:
            *args: Any additional arguments
            **kwargs: Any additional keyword arguments

        Returns:
            bool: True if connection is successful
        """
        self._set_up(overwrite=False)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store, storage_context=self.storage_context
        )

        return True

    def add_docs(
        self,
        new_doc_dir: Optional[Union[Path, str]] = None,
        new_doc_paths_or_urls: Optional[Sequence[Union[Path, str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add new documents to the underlying database and add to the index.

        Args:
            new_doc_dir: A dir of input documents that are used to create the records in database.
            new_doc_paths_or_urls: A sequence of input documents that are used to create the records in database. A document can be a path to a file or a url.
            *args: Any additional arguments
            **kwargs: Any additional keyword arguments
        """
        self._validate_query_index()
        documents = self._load_doc(input_dir=new_doc_dir, input_docs=new_doc_paths_or_urls)
        for doc in documents:
            self.index.insert(doc)

    def query(self, question: str) -> str:
        """
        Retrieve information from indexed documents by processing a query using the engine's LLM.

        Args:
            question: A natural language query string used to search the indexed documents.

        Returns:
            A string containing the response generated by LLM.
        """
        self._validate_query_index()
        self.query_engine = self.index.as_query_engine(llm=self.llm)
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

    def _validate_query_index(self) -> None:
        """Ensures an index exists"""
        if not hasattr(self, "index"):
            raise Exception("Query index is not initialized. Please call init_db or connect_db first.")

    def _set_up(self, overwrite: bool) -> None:
        """
        Set up ChromaDB and LlamaIndex storage by:
        1. Initialize the ChromaDB using VectorDBFactory and create a collection with the given name.
        2. Create the LlamaIndex vector store and storage context for the collection.
        Args:
            overwrite: If True, overwrite the existing collection with the same name.
        """
        self.vector_db = VectorDBFactory().create_vector_db(db_type="chroma", **self.db_config)
        self.collection = self.vector_db.create_collection(collection_name=self.collection_name, overwrite=overwrite)
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

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


# mypy will fail if ChromaDBQueryEngine does not implement RAGQueryEngine protocol
if TYPE_CHECKING:
    from .query_engine import RAGQueryEngine

    def _check_implement_protocol(o: ChromaDBQueryEngine) -> RAGQueryEngine:
        return o
