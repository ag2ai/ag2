# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Optional

from autogen.import_utils import optional_import_block

with optional_import_block():
    import chromadb
    import chromadb.utils.embedding_functions as ef
    from chromadb.api.types import EmbeddingFunction
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.llms import LLM
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.chroma import ChromaVectorStore

DEFAULT_COLLECTION_NAME = "docling-parsed-docs"


class DoclingQueryEngine:
    """
    Leverage llamaIndex VectorStoreIndex and Chromadb to query docling parsed md file
    """

    def __init__(  # type: ignore
        self,
        db_path: Optional[str] = "./chroma",
        embedding_function: Optional[EmbeddingFunction[Any]] = ef.DefaultEmbeddingFunction(),
        metadata: Optional[dict[Any, Any]] = {"hnsw:space": "ip", "hnsw:construction_ef": 30, "hnsw:M": 32},
        llm: Optional["LLM"] = OpenAI(model="gpt-4o", temperature=0.0),
    ) -> None:
        """
        Initializes the DoclingQueryEngine with db_path
        metadata, and embedding function and llm.
        Args:
            db_path: the path to save chromadb data
            embedding_function: The embedding function to use.
            metadata: The metadata of the vector database.
        """
        self.llm = llm
        self.embedding_function = embedding_function
        self.metadata = metadata
        if db_path:
            self.client = chromadb.PersistentClient(path=db_path)
        else:
            self.client = chromadb.PersistentClient()

    def init_db(
        self,
        input_doc_paths: list[str],
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """
        Initinalize vectordb by putting input docs into given collection
        """

        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata=self.metadata,
            get_or_create=True,  # If collection already exists, get the collection
        )

        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        documents = self._load_doc(input_doc_paths)
        self.index = VectorStoreIndex.from_documents(documents, storage_context=self.storage_context)

    def query(self, question: str) -> str:
        self.query_engine = self.index.as_query_engine(llm=self.llm)
        response = self.query_engine.query(question)

        return str(response)

    def add_docs(self, new_doc_paths: list[str]) -> None:
        new_docs = self._load_doc(new_doc_paths)
        for doc in new_docs:
            self.index.insert(doc)

    def _load_doc(self, input_docs: list[str]) -> list["LlamaDocument"]:  # type: ignore
        """
        Load documents from the input files
        """
        for doc in input_docs:
            if not os.path.exists(doc):
                raise ValueError(f"Document file not found: {doc}")

        loaded_documents = []
        loaded_documents.extend(SimpleDirectoryReader(input_files=input_docs).load_data())

        return loaded_documents
