# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import

with optional_import_block():
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.llms import LLM
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.core.vector_stores.types import BasePydanticVectorStore
    from llama_index.llms.openai import OpenAI

__all__ = ["LlamaIndexQueryEngine"]


EMPTY_RESPONSE_TEXT = "Empty Response"  # Indicates that the query did not return any results
EMPTY_RESPONSE_REPLY = "Sorry, I couldn't find any information on that. If you haven't ingested any documents, please try that."  # Default response for queries without results


# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@require_optional_import("llama_index", "rag")
@export_module("autogen.agentchat.contrib.rag")
class LlamaIndexQueryEngine:
    """
    This engine leverages LlamaIndex's VectorStoreIndex to efficiently index and retrieve documents, and generate an answer in response
    to natural language queries. It use any LlamaIndex [vector store](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/).

    By default the engine will use OpenAI's GPT-4o model (use the `llm` parameter to change that).
    """

    def __init__(  # type: ignore[no-any-unimported]
        self,
        vector_store: "BasePydanticVectorStore",
        llm: Optional["LLM"] = None,
        file_reader_class: Optional[type["SimpleDirectoryReader"]] = None,
    ) -> None:
        """
        Initializes the LlamaIndexQueryEngine with the given vector store.
        Args:
            vector_store: The vector store to use for indexing and querying documents.
            llm: LLM model used by LlamaIndex for query processing. You can find more supported LLMs at [LLM](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/).
            file_reader_class: The file reader class to use for loading documents. Only SimpleDirectoryReader is currently supported.
        """
        self.llm: LLM = llm or OpenAI(model="gpt-4o", temperature=0.0)  # type: ignore[no-any-unimported]
        self.vector_store = vector_store
        self.file_reader_class = file_reader_class if file_reader_class else SimpleDirectoryReader

    def init_db(
        self,
        new_doc_dir: Optional[Union[Path, str]] = None,
        new_doc_paths_or_urls: Optional[Sequence[Union[Path, str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Initialize the database with the input documents or records.

        It takes the following steps:
        1. Set up LlamaIndex storage context.
        2. insert documents and build an index upon them.

        Args:
            new_doc_dir: a dir of input documents that are used to create the records in database.
            new_doc_paths_or_urls: A sequence of input documents that are used to create the records in database. A document can be a Path to a file or a url.
            *args: Any additional arguments
            **kwargs: Any additional keyword arguments

        Returns:
            bool: True if initialization is successful

        """

        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        documents = self._load_doc(input_dir=new_doc_dir, input_docs=new_doc_paths_or_urls)
        self.index = VectorStoreIndex.from_documents(documents=documents, storage_context=self.storage_context)
        return True

    def connect_db(self, *args: Any, **kwargs: Any) -> bool:
        """Connect to the database.
        It sets up the LlamaIndex storage and create an index from the existing vector store.

        Args:
            *args: Any additional arguments
            **kwargs: Any additional keyword arguments

        Returns:
            bool: True if connection is successful
        """
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
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
            new_doc_paths_or_urls: A sequence of input documents that are used to create the records in database. A document can be a Path to a file or a url.
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

    def _validate_query_index(self) -> None:
        """Ensures an index exists"""
        if not hasattr(self, "index"):
            raise Exception("Query index is not initialized. Please call init_db or connect_db first.")

    def _load_doc(  # type: ignore[no-any-unimported]
        self, input_dir: Optional[Union[Path, str]], input_docs: Optional[Sequence[Union[Path, str]]]
    ) -> Sequence["LlamaDocument"]:
        """
        Load documents from a directory and/or a sequence of file paths.

        Default to uses LlamaIndex's SimpleDirectoryReader that supports multiple file[formats](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/#supported-file-types).

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
        loaded_documents: list["LlamaDocument"] = []  # type: ignore[no-any-unimported]
        if input_dir:
            logger.info(f"Loading docs from directory: {input_dir}")
            if not os.path.exists(input_dir):
                raise ValueError(f"Input directory not found: {input_dir}")
            loaded_documents.extend(self.file_reader_class(input_dir=input_dir).load_data())  # type: ignore[operator]

        if input_docs:
            for doc in input_docs:
                logger.info(f"Loading input doc: {doc}")
                if not os.path.exists(doc):
                    raise ValueError(f"Document file not found: {doc}")
            loaded_documents.extend(self.file_reader_class(input_files=input_docs).load_data())  # type: ignore[operator, arg-type]

        if not input_dir and not input_docs:
            raise ValueError("No input directory or docs provided!")

        return loaded_documents


# mypy will fail if LlamaIndexQueryEngine does not implement RAGQueryEngine protocol
if TYPE_CHECKING:
    from .query_engine import RAGQueryEngine

    def _check_implement_protocol(o: LlamaIndexQueryEngine) -> RAGQueryEngine:
        return o
