# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any, Optional

from autogen.import_utils import optional_import_block, require_optional_import

with optional_import_block():
    import chromadb
    from chromadb.api.models.Collection import Collection
    from chromadb.api.types import EmbeddingFunction
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.llms import LLM
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.chroma import ChromaVectorStore

DEFAULT_COLLECTION_NAME = "docling-parsed-docs"

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@require_optional_import(["chromadb", "llama_index"], "rag")
class DoclingMdQueryEngine:
    """
    This engine leverages Chromadb to persist document embeddings in a named collection
    and LlamaIndex’s VectorStoreIndex to efficiently index and retrieve documents, and generate an answer in response
    to natural language queries. The Chromadb collection serves as the storage layer, while
    the collection name uniquely identifies the set of documents within the persistent database.
    """

    def __init__(  # type: ignore
        self,
        db_path: Optional[str] = None,
        embedding_function: "Optional[EmbeddingFunction[Any]]" = None,
        metadata: Optional[dict[str, Any]] = None,
        llm: Optional["LLM"] = None,
    ) -> None:
        """
        Initializes the DoclingMdQueryEngine with db_path, metadata, and embedding function and llm.
        Args:
            db_path: The file system path where Chromadb will store its persistent data.
                If not specified, the default directory "./chroma" is used.
            embedding_function: A callable that converts text into vector embeddings. Default embedding uses Sentence Transformers model all-MiniLM-L6-v2.
                For more embeddings that ChromaDB support, please refer to [embeddings](https://docs.trychroma.com/docs/embeddings/embedding-functions)
            metadata: A dictionary containing configuration parameters for the Chromadb collection.
                This metadata is typically used to configure the HNSW indexing algorithm.
                For more details about the default metadata, please refer to [HNSW configuration](https://cookbook.chromadb.dev/core/configuration/#hnsw-configuration)
            llm: LLM model used by LlamaIndex for query processing.
                 You can find more supported LLMs at [LLM](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/)
        """
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
        """
        Initialize the database for document indexing.

        Creates (or retrieves) a Chromadb collection using the provided collection name,
        loads documents from a directory or a list of file paths, and builds the vector index
        to enable efficient querying.

        Args:
            input_dir: The directory path from which to load documents.
                The directory should contain Docling-parsed Markdown files.
                If not provided, no directory-based documents are loaded.
            input_doc_paths: A list of file paths to individual documents.
                Each file should be a Docling-parsed Markdown file.
                If not provided, no file-based documents are loaded.
            collection_name: The unique name for the Chromadb collection.
                This name identifies the collection that saves the embeddings of input documents.
                If the collection name pre-exists, this method will
                    retrieve the collection instead of creating a new one.
                If omitted, the default name ("docling-parsed-docs") is used.
        """
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
        """
        Retrieve information from indexed documents by processing a natural language query.

        Args:
            question: A natural language query string used to search the indexed documents.

        Returns:
            A string containing the response generated by LLM.
        """
        self.query_engine = self.index.as_query_engine(llm=self.llm)
        response = self.query_engine.query(question)

        return str(response)

    def add_docs(self, new_doc_dir: Optional[str] = "", new_doc_paths: Optional[list[str]] = []) -> None:
        """
        Add additional documents to the existing vector index.

        Loads new Docling-parsed Markdown files from a specified directory or a list of file paths
        and inserts them into the current index for future queries.

        Args:
            new_doc_dir: The directory path from which to load additional documents.
                If provided, all eligible files in this directory are loaded.
            new_doc_paths: A list of file paths specifying additional documents to load.
                Each file should be a Docling-parsed Markdown file.
        """
        new_docs = self._load_doc(input_dir=new_doc_dir, input_docs=new_doc_paths)
        for doc in new_docs:
            self.index.insert(doc)

    def _load_doc(  # type: ignore
        self, input_dir: Optional[str], input_docs: Optional[list[str]]
    ) -> list["LlamaDocument"]:
        """
        Load documents from a directory and/or a list of file paths.

        This helper method reads Docling-parsed Markdown files using LlamaIndex's
        SimpleDirectoryReader. It supports multiple file [formats]((https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/#supported-file-types)),
          but the intended use is for documents processed by Docling.

        Args:
            input_dir: The directory containing documents to be loaded.
                If provided, all files in the directory will be considered.
            input_docs: A list of individual file paths to load.
                Each path must point to an existing file.

        Returns:
            A list of documents loaded as LlamaDocument objects.

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
            loaded_documents.extend(SimpleDirectoryReader(input_files=input_docs).load_data())

        if not input_dir and not input_docs:
            raise ValueError("No input directory or docs provided!")

        return loaded_documents

    def _create_index(  # type: ignore
        self, collection: "Collection", docs: list["LlamaDocument"]
    ) -> "VectorStoreIndex":
        """
        Build a vector index for document retrieval using a Chromadb collection.

        Wraps the provided Chromadb collection into a vector store and uses LlamaIndex's
        StorageContext to create a VectorStoreIndex from the supplied documents.

        Args:
            collection: A Chromadb Collection object that stores the embeddings of the documents.
            docs: A list of LlamaDocument objects representing the documents to be indexed.

        Returns:
            A VectorStoreIndex object built from the provided documents and backed by the given collection.
        """

        self.vector_store = ChromaVectorStore(chroma_collection=collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        index = VectorStoreIndex.from_documents(docs, storage_context=self.storage_context)

        return index

    def get_collection_name(self) -> Optional[str]:
        """
        Retrieve the name of the Chromadb collection used for indexing.

        Returns:
            The name of the Chromadb collection as a string, or None if no index exists.
        """
        if self.index:
            return self.collection_name
        return None
