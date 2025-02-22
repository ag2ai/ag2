# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from autogen.agentchat.contrib.rag.query_engine import VectorDbQueryEngine
from autogen.agentchat.contrib.vectordb.base import VectorDBFactory
from autogen.agentchat.contrib.vectordb.mongodb import MongoDBAtlasVectorDB
from autogen.import_utils import optional_import_block, require_optional_import

with optional_import_block():
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.node_parser.docling import DoclingNodeParser
    from llama_index.readers.docling import DoclingReader
    from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch


DEFAULT_COLLECTION_NAME = "docling-parsed-docs"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@require_optional_import(["pymongo", "llama_index"], "rag")
class MongoDBQueryEngine(VectorDbQueryEngine):
    def __init__(
        self,
        connection_string: str = "",
        database_name: str = "vector_db",
        embedding_function: Optional[Callable[..., Any]] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        index_name: str = "vector_index",
    ):
        super().__init__()

        self.vector_db: MongoDBAtlasVectorDB = VectorDBFactory.create_vector_db(  # type: ignore[assignment]
            db_type="mongodb",
            connection_string=connection_string,
            database_name=database_name,
            index_name=index_name,
            embedding_function=embedding_function,
            collection_name=collection_name,
        )
        self.vector_search_engine = MongoDBAtlasVectorSearch(
            mongodb_client=self.vector_db.client, db_name=database_name, collection_name=collection_name
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_search_engine)
        self.indexer = None

    def init_db(  # type: ignore[no-untyped-def]
        self,
        new_doc_dir: Optional[str | Path] = None,
        new_doc_paths: Optional[list[str | Path]] = None,
        *args,
        **kwargs,
    ) -> bool:  # type: ignore[no-untyped-def]
        """Initialize the database with the input documents or records.

        This method initializes database with the input documents or records.
        Usually, it takes the following steps,
        1. connecting to a database.
        2. insert records
        3. build indexes etc.

        Args:
            new_doc_dir: a dir of input documents that are used to create the records in database.
            new_doc_paths:
                a list of input documents that are used to create the records in database.
                a document can be a path to a file or a url.
            *args: Any additional arguments
            **kwargs: Any additional keyword arguments

        Returns:
            bool: True if initialization is successful, False otherwise
        """
        if not self.connect_db():
            return False

        if new_doc_dir or new_doc_paths:
            self.add_records(new_doc_dir, new_doc_paths)  # type: ignore[no-untyped-call]

        self.indexer = VectorStoreIndex.from_vector_store(
            self.vector_search_engine, storage_context=self.storage_context
        )
        return True

    def connect_db(self, *args, **kwargs) -> bool:  # type: ignore[no-untyped-def]
        """
        Connect to the MongoDB database by issuing a ping.

        Returns:
            True if the connection is successful; False otherwise.
        """
        try:
            self.vector_db.client.admin.command("ping")
            return True
        except Exception as error:
            logger.error("Failed to connect to MongoDB: %s", error)
            return False

    def add_records(self, new_doc_dir=None, new_doc_paths_or_urls=None, *args, **kwargs):  # type: ignore[no-untyped-def]
        document_list = []  # type: ignore[var-annotated]
        if new_doc_dir:
            document_list.extend(Path(new_doc_dir).glob("**/*"))
        if new_doc_paths_or_urls:
            document_list.append(new_doc_paths_or_urls)

        reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
        node_parser = DoclingNodeParser()

        documents = reader.load_data(document_list)
        parser_nodes = node_parser._parse_nodes(documents)

        # print("Parser data: ", parser)

        # # document_reader = SimpleDirectoryReader(input_files=document_list).load_data()
        # self.vector_db.insert_docs([
        #     Document(  # type: ignore[typeddict-item, typeddict-unknown-key]
        #         id=document.id_,
        #         content=document.get_content(),
        #         metadata=document.metadata,
        #     )
        #     for document in parser
        # ])

        docs_to_insert = []
        for node in parser_nodes:
            doc_dict = {
                "id": node.id_,  # Ensure the key is 'id'
                "content": node.get_content(),
                "metadata": node.metadata,
            }
            docs_to_insert.append(doc_dict)

        # Insert documents into vector DB.
        self.vector_db.insert_docs(docs_to_insert)  # type: ignore[arg-type]

    def query(self, question, *args, **kwargs):  # type: ignore[no-untyped-def]
        response = self.indexer.as_chat_engine().query(question)  # type: ignore[attr-defined]

        return response
