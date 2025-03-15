# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from pydantic import BaseModel

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import

with optional_import_block():
    import chromadb
    from chromadb.api.models.Collection import Collection
    from chromadb.api.types import EmbeddingFunction
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.llms import LLM
    from llama_index.core.query_engine import CitationQueryEngine
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.core.schema import NodeWithScore
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.chroma import ChromaVectorStore

__all__ = ["VectorChromaCitationQueryEngine", "VectorChromaQueryEngine"]

DEFAULT_COLLECTION_NAME = "docling-parsed-docs"
EMPTY_RESPONSE_TEXT = "Empty Response"  # Indicates that the query did not return any results
EMPTY_RESPONSE_REPLY = "Sorry, I couldn't find any information on that. If you haven't ingested any documents, please try that."  # Default response for queries without results

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@require_optional_import(["chromadb", "llama_index"], "rag")
@export_module("autogen.agents.experimental")
class VectorChromaQueryEngine:
    """
    This engine leverages Chromadb to persist document embeddings in a named collection
    and LlamaIndex's VectorStoreIndex to efficiently index and retrieve documents, and generate an answer in response
    to natural language queries. The Chromadb collection serves as the storage layer, while
    the collection name uniquely identifies the set of documents within the persistent database.

    This implements the autogen.agentchat.contrib.rag.RAGQueryEngine protocol.
    """

    def __init__(  # type: ignore[no-any-unimported]
        self,
        db_path: Optional[str] = None,
        embedding_function: "Optional[EmbeddingFunction[Any]]" = None,
        metadata: Optional[dict[str, Any]] = None,
        llm: Optional["LLM"] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Initializes the VectorChromaQueryEngine with db_path, metadata, and embedding function and llm.
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
            collection_name (str): The unique name for the Chromadb collection. If omitted, a constant name will be used. Populate this to reuse previous ingested data.
        """
        self.llm: LLM = llm or OpenAI(model="gpt-4o", temperature=0.0)  # type: ignore[no-any-unimported]
        self.embedding_function: EmbeddingFunction[Any] = embedding_function or DefaultEmbeddingFunction()  # type: ignore[no-any-unimported,assignment]
        self.metadata: dict[str, Any] = metadata or {
            "hnsw:space": "ip",
            "hnsw:construction_ef": 30,
            "hnsw:M": 32,
        }
        self.client = chromadb.PersistentClient(path=db_path or "./chroma")
        self.collection_name: Optional[str] = collection_name

        self.connect_db()

    def connect_db(self, *args: Any, **kwargs: Any) -> bool:
        """
        Establish a connection to the Chromadb database and initialize the collection.
        """

        self.collection_name = self.collection_name or DEFAULT_COLLECTION_NAME

        if self._collection_exists(self.collection_name):
            logger.info(f"Using existing collection {self.collection_name} from the database.")
        else:
            logger.info(f"Creating new collection {self.collection_name} in the database.")

        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata=self.metadata,
            get_or_create=True,  # If collection already exists, get the collection
        )
        self.index = self._create_index(self.collection)

        return True

    def query(self, question: str) -> str:
        """
        Retrieve information from indexed documents by processing a natural language query.

        Args:
            question: A natural language query string used to search the indexed documents.

        Returns:
            A string containing the response generated by LLM.
        """
        self.validate_query_index()
        self.query_engine = self.index.as_query_engine(llm=self.llm)
        response = self.query_engine.query(question)

        if str(response) == EMPTY_RESPONSE_TEXT:
            return EMPTY_RESPONSE_REPLY

        return str(response)

    def add_docs(
        self,
        new_doc_dir: Optional[Union[Path, str]] = None,
        new_doc_paths_or_urls: Optional[Sequence[Union[Path, str]]] = None,
    ) -> None:
        """
        Add additional documents to the existing vector index.

        Loads new Docling-parsed Markdown files from a specified directory or a list of file paths
        and inserts them into the current index for future queries.

        Args:
            new_doc_dir: The directory path from which to load additional documents.
                If provided, all eligible files in this directory are loaded.
            new_doc_paths_or_urls: A list of file paths specifying additional documents to load.
                Each file should be a Docling-parsed Markdown file.
        """
        self.validate_query_index()
        new_doc_dir = new_doc_dir or ""
        new_doc_paths = new_doc_paths_or_urls or []
        new_docs = self._load_doc(input_dir=new_doc_dir, input_docs=new_doc_paths)
        for doc in new_docs:
            self.index.insert(doc)

    def _load_doc(  # type: ignore[no-any-unimported]
        self, input_dir: Optional[Union[Path, str]], input_docs: Optional[Sequence[Union[Path, str]]]
    ) -> list["LlamaDocument"]:
        """
        Load documents from a directory and/or a list of file paths.

        This helper method reads Docling-parsed Markdown files using LlamaIndex's
        SimpleDirectoryReader. It supports multiple file [formats]((https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/#supported-file-types)),
          but the intended use is for documents processed by Docling.

        Args:
            input_dir (Optional[Union[Path, str]]): The directory containing documents to be loaded.
                If provided, all files in the directory will be considered.
            input_docs (Optional[list[Union[Path, str]]]): A list of individual file paths to load.
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
            loaded_documents.extend(SimpleDirectoryReader(input_files=list(input_docs)).load_data())

        if not input_dir and not input_docs:
            raise ValueError("No input directory or docs provided!")

        return loaded_documents

    def _create_index(  # type: ignore[no-any-unimported]
        self, collection: "Collection"
    ) -> "VectorStoreIndex":
        """
        Build a vector index for document retrieval using a Chromadb collection.

        Wraps the provided Chromadb collection into a vector store and uses LlamaIndex's
        StorageContext to create a VectorStoreIndex from the collection.

        Args:
            collection (Collection): A Chromadb Collection object that stores the embeddings of the documents.

        Returns:
            A VectorStoreIndex object built from the provided collection.
        """
        self.vector_store = ChromaVectorStore(chroma_collection=collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store, storage_context=self.storage_context)

        return index

    def _collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection with the given name exists in the database.

        Args:
            collection_name (str): The name of the collection to check.

        Returns:
            True if a collection with the given name exists in the database, False otherwise.
        """
        existing_collections = self.client.list_collections()
        return any(col == collection_name for col in existing_collections)

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

    def validate_query_index(self) -> None:
        """Ensures an index exists"""
        if not hasattr(self, "index"):
            raise Exception("Query index is not initialized. Please ingest some documents before querying.")

    def init_db(
        self,
        new_doc_dir: Optional[Union[Path, str]] = None,
        new_doc_paths_or_urls: Optional[Sequence[Union[Path, str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Not required nor implemented for VectorChromaQueryEngine"""
        raise NotImplementedError("Method, init_db, not required nor implemented for VectorChromaQueryEngine")


class AnswerWithCitations(BaseModel):  # type: ignore[no-any-unimported]
    answer: str
    citations: list["NodeWithScore"]  # type: ignore[no-any-unimported]


@require_optional_import(["chromadb", "llama_index"], "rag")
@export_module("autogen.agents.experimental")
class VectorChromaCitationQueryEngine(VectorChromaQueryEngine):
    """
    This engine leverages VectorChromaQueryEngine and CitationQueryEngine to answer queries with citations.
    """

    def __init__(  # type: ignore[no-any-unimported]
        self,
        db_path: Optional[str] = None,
        embedding_function: "Optional[EmbeddingFunction[Any]]" = None,
        metadata: Optional[dict[str, Any]] = None,
        llm: Optional["LLM"] = None,
        collection_name: Optional[str] = None,
        enable_query_citations: bool = False,
        citation_chunk_size: int = 512,
    ) -> None:
        """
        see parent class VectorChromaQueryEngine.
        """
        super().__init__(db_path, embedding_function, metadata, llm, collection_name)
        self.enable_query_citations = enable_query_citations
        self.citation_chunk_size = citation_chunk_size

    def query_with_citations(self, query: str) -> AnswerWithCitations:
        """
        Query the index with the given query and return the answer along with citations.

        Args:
            query (str): The query to be answered.
            citation_chunk_size (int): The size of chunks to use for each citation. Default is 512.

        Returns:
            AnswerWithCitations: An object containing the answer and citations.
        """

        query_engine = CitationQueryEngine.from_args(
            index=self.index,
            citation_chunk_size=self.citation_chunk_size,
        )

        response = query_engine.query(query)

        if hasattr(response, "response"):
            return AnswerWithCitations(answer=response.response, citations=response.source_nodes)
        else:
            raise ValueError(f"Query response of type '{type(response)}' does not contain a response attribute.")


# mypy will fail if ChromaDBQueryEngine does not implement RAGQueryEngine protocol
if TYPE_CHECKING:
    from ....agentchat.contrib.rag.query_engine import RAGQueryEngine

    def _check_implement_protocol(o: VectorChromaQueryEngine) -> RAGQueryEngine:
        return o
