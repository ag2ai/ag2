# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Replace 'your_module' with the actual module name where MongoDBQueryEngine is defined.
from autogen.agentchat.contrib.rag.mongodb_query_engine import MongoDBQueryEngine


# A dummy client to simulate MongoDB's behavior.
class DummyCollection:
    def __init__(self):
        self.list_collection_names = MagicMock(return_value=[])


class DummyDatabase:
    def __init__(self):
        self._dummy_collection = DummyCollection()

    def __getitem__(self, key):
        # When accessing a collection via subscripting, return the dummy collection.
        return self._dummy_collection

    def list_collection_names(self):
        # Return an empty list or a list of dummy collection names.
        return []

    def create_collection(self, name, *args, **kwargs):
        # Simulate creation of a new collection by returning a dummy collection.
        return DummyCollection()


class DummyClient:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.admin = MagicMock()

    def command(self, command):
        if self.should_fail:
            raise Exception("Ping failed")
        return "ok"

    def __getitem__(self, key):
        # Return a dummy database when subscripting the client.
        return DummyDatabase()


# A dummy chat engine to simulate query responses.
class DummyChatEngine:
    def query(self, question):
        return f"Response to: {question}"


class TestMongoDBQueryEngine(unittest.TestCase):
    def setUp(self):
        # Patch the vector database factory and search engine constructors.
        patcher_factory = patch("autogen.agentchat.contrib.vectordb.base.VectorDBFactory.create_vector_db")
        self.mock_create_vector_db = patcher_factory.start()
        self.addCleanup(patcher_factory.stop)

        patcher_search = patch("llama_index.vector_stores.mongodb.MongoDBAtlasVectorSearch")
        self.mock_mongo_search = patcher_search.start()
        self.addCleanup(patcher_search.stop)

        # Set up a dummy MongoDB client.
        self.dummy_client = DummyClient()
        dummy_vector_db = MagicMock()
        dummy_vector_db.client = self.dummy_client
        self.mock_create_vector_db.return_value = dummy_vector_db

        # Instantiate the engine with dummy parameters.
        self.engine = MongoDBQueryEngine(connection_string="dummy", database_name="test_db")
        # Pre-assign a dummy indexer (used in query and add_records).
        self.engine.indexer = MagicMock()

    def test_connect_db_success(self):
        # Simulate a successful ping.
        self.dummy_client.should_fail = False
        result = self.engine.connect_db()
        self.assertTrue(result)
        # Verify that the ping command was called.
        self.dummy_client.admin.command.assert_called_with("ping")

    def test_connect_db_failure(self):
        # Simulate a failure during the ping.
        self.dummy_client.should_fail = True
        self.dummy_client.admin.command.side_effect = Exception("Ping failed")
        result = self.engine.connect_db()
        self.assertFalse(result)

    @patch("llama_index.core.SimpleDirectoryReader")
    @patch("your_module.VectorStoreIndex.from_documents")
    def test_init_db_with_documents(self, mock_from_documents, mock_simple_dir_reader):
        # Create dummy documents.
        dummy_documents = ["doc1", "doc2"]
        reader_instance = MagicMock()
        reader_instance.load_data.return_value = dummy_documents
        mock_simple_dir_reader.return_value = reader_instance

        # Patch Path.glob to return a list with at least one file.
        with patch.object(Path, "glob", return_value=[Path("dummy.txt")]):
            result = self.engine.init_db(new_doc_dir="dummy_dir")
            self.assertTrue(result)
            # Ensure that the index was built with the dummy documents.
            mock_from_documents.assert_called_with(dummy_documents, storage_context=self.engine.storage_context)

    def test_init_db_no_documents(self):
        # Without any document directory or file paths, the method should return False.
        result = self.engine.init_db()
        self.assertFalse(result)

    @patch("your_module.SimpleDirectoryReader")
    def test_add_records_no_documents(self, mock_simple_dir_reader):
        # When no document paths or directory are provided, expect a warning and no processing.
        with self.assertLogs(level="WARNING") as cm:
            self.engine.add_records()
            self.assertTrue(any("No documents found for adding records." in msg for msg in cm.output))

    @patch("your_module.SimpleDirectoryReader")
    def test_add_records_with_documents(self, mock_simple_dir_reader):
        # Create dummy documents to be loaded.
        dummy_documents = ["doc1", "doc2", "doc3"]
        reader_instance = MagicMock()
        reader_instance.load_data.return_value = dummy_documents
        mock_simple_dir_reader.return_value = reader_instance

        # Ensure that the indexer's insert method is callable.
        self.engine.indexer.insert = MagicMock()

        # Provide a list of dummy document paths.
        self.engine.add_records(new_doc_paths_or_urls=["dummy1.txt", "dummy2.txt"])
        # Verify that insert was called for each dummy document.
        self.assertEqual(self.engine.indexer.insert.call_count, len(dummy_documents))

    def test_query_success(self):
        # Simulate a working chat engine.
        dummy_chat_engine = DummyChatEngine()
        self.engine.indexer.as_chat_engine.return_value = dummy_chat_engine

        response = self.engine.query("Test question")
        self.assertEqual(response, "Response to: Test question")

    def test_query_failure(self):
        # Simulate failure by having as_chat_engine raise an exception.
        self.engine.indexer.as_chat_engine.side_effect = Exception("Query failed")
        response = self.engine.query("Test question")
        self.assertIsNone(response)


if __name__ == "__main__":
    unittest.main()
