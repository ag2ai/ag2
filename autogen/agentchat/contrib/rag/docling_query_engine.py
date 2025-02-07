import os
from typing import Callable

import chromadb
import chromadb.utils.embedding_functions as ef
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.schema import Document as LlamaDocument
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

DEFAULT_COLLECTION_NAME = "docling-parsed-docs"


class DoclingQueryEngine:
    """
    Leverage llamaIndex VectorStoreIndex and Chromadb to query docling parsed md file
    """

    def __init__(
        self,
        input_doc_paths: list[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_function: Callable = ef.DefaultEmbeddingFunction(),
        metadata: dict = {"hnsw:space": "ip", "hnsw:construction_ef": 30, "hnsw:M": 32},
        llm=OpenAI(model="gpt-4o", temperature=0.0),
    ):
        """
        Initializes the DoclingQueryEngine with the specified document paths, database type, and embedding function.
        Args:
            input_doc_paths: The paths to the input documents.
            database: The type of database to use. Defaults to "chroma".
            embedding_function: The embedding function to use. Defaults to None.
            metadata: The metadata of the vector database. Default is None.
        """
        self.collection_name = collection_name
        self.llm = llm
        self.client = chromadb.PersistentClient()
        self.collection = self.client.create_collection(
            name=collection_name, embedding_function=embedding_function, metadata=metadata, get_or_create=True
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

    def _load_doc(self, input_docs: list[str]) -> list["LlamaDocument"]:
        """
        Load documents from the input files
        """
        for doc in input_docs:
            if not os.path.exists(doc):
                raise ValueError(f"Document file not found: {doc}")

        loaded_documents = []
        loaded_documents.extend(SimpleDirectoryReader(input_files=input_docs).load_data())

        return loaded_documents
