# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock, patch

# Import the classes and functions from the module under test.
from autogen.agentchat.contrib.rag.docling_query_engine import (
    DoclingChromaMdQueryEngine,
    DoclingMongoAtlasMdQueryEngine,
    DoclingQueryEngine,
    load_documents,
)


# A dummy document class to simulate LlamaDocument instances.
class DummyDocument:
    def __init__(self, text):  # type: ignore[no-untyped-def]
        self.text = text


# A dummy query engine to simulate the behavior of the vector index query engine.
class DummyQueryEngine:
    def query(self, question: str) -> str:
        return f"Dummy answer to: {question}"


class TestLoadDocuments(unittest.TestCase):
    @patch("os.path.exists", return_value=False)
    def test_nonexistent_directory(self, mock_exists):  # type: ignore[no-untyped-def]
        # When directory does not exist, should raise ValueError.
        with self.assertRaises(ValueError):
            load_documents("nonexistent_dir", None)

    @patch("os.path.exists", return_value=False)
    def test_nonexistent_file(self, mock_exists):  # type: ignore[no-untyped-def]
        # When file does not exist, should raise ValueError.
        with self.assertRaises(ValueError):
            load_documents(None, ["nonexistent_file.md"])

    @patch("os.path.exists", return_value=True)
    @patch("autogen.agentchat.contrib.rag.docling_query_engine.SimpleDirectoryReader")
    def test_successful_load_from_dir(self, mock_reader, mock_exists):  # type: ignore[no-untyped-def]
        # Setup the dummy reader to return dummy documents.
        dummy_docs = [DummyDocument("doc1"), DummyDocument("doc2")]  # type: ignore[no-untyped-def, no-untyped-call]
        instance = MagicMock()
        instance.load_data.return_value = dummy_docs
        mock_reader.return_value = instance

        docs = load_documents("some_dir", None)
        self.assertEqual(docs, dummy_docs)
        mock_reader.assert_called_with(input_dir="some_dir")

    @patch("os.path.exists", return_value=True)
    @patch("autogen.agentchat.contrib.rag.docling_query_engine.SimpleDirectoryReader")
    def test_successful_load_from_files(self, mock_reader, mock_exists):  # type: ignore[no-untyped-def]
        dummy_docs = [DummyDocument("doc1")]  # type: ignore[no-untyped-def, no-untyped-call]
        instance = MagicMock()
        instance.load_data.return_value = dummy_docs
        mock_reader.return_value = instance

        docs = load_documents(None, ["file1.md"])
        self.assertEqual(docs, dummy_docs)
        mock_reader.assert_called_with(input_files=["file1.md"])


class TestDoclingChromaMdQueryEngine(unittest.TestCase):
    def setUp(self):  # type: ignore[no-untyped-def]
        # Create an instance of the engine with a dummy db_path.
        self.engine = DoclingChromaMdQueryEngine(db_path="./tmp/chroma")

    @patch("autogen.agentchat.contrib.rag.docling_query_engine.load_documents", return_value=[DummyDocument("doc")])  # type: ignore[no-untyped-def, no-untyped-call]
    @patch.object(DoclingChromaMdQueryEngine, "_create_index", return_value=MagicMock())
    def test_init_db(self, mock_create_index, mock_load_documents):  # type: ignore[no-untyped-def]
        # Test that init_db sets the collection name and calls _create_index.
        self.engine.init_db(input_dir="dummy_dir", collection_name="test_collection")
        self.assertEqual(self.engine.collection_name, "test_collection")
        mock_create_index.assert_called()

    def test_query_without_init(self):  # type: ignore[no-untyped-def]
        # Calling query before init_db should raise a ValueError.
        with self.assertRaises(ValueError):
            self.engine.query("What is test?")

    @patch("autogen.agentchat.contrib.rag.docling_query_engine.load_documents", return_value=[DummyDocument("doc")])  # type: ignore[no-untyped-def, no-untyped-call]
    @patch.object(DoclingChromaMdQueryEngine, "_create_index")
    def test_query_after_init(self, mock_create_index, mock_load_documents):  # type: ignore[no-untyped-def]
        # Setup a dummy index whose as_query_engine returns our DummyQueryEngine.
        dummy_index = MagicMock()
        dummy_index.as_query_engine.return_value = DummyQueryEngine()
        mock_create_index.return_value = dummy_index

        self.engine.init_db(input_dir="dummy_dir", collection_name="test_collection")
        answer = self.engine.query("What is test?")
        self.assertEqual(answer, "Dummy answer to: What is test?")

    @patch("autogen.agentchat.contrib.rag.docling_query_engine.load_documents", return_value=[DummyDocument("doc")])  # type: ignore[no-untyped-def, no-untyped-call]
    @patch.object(DoclingChromaMdQueryEngine, "_create_index")
    def test_add_docs_without_init(self, mock_create_index, mock_load_documents):  # type: ignore[no-untyped-def]
        # Calling add_docs before init_db should raise a ValueError.
        with self.assertRaises(ValueError):
            self.engine.add_docs(new_doc_dir="dummy_dir")

    @patch("autogen.agentchat.contrib.rag.docling_query_engine.load_documents", return_value=[DummyDocument("doc_new")])  # type: ignore[no-untyped-def, no-untyped-call]
    @patch.object(DoclingChromaMdQueryEngine, "_create_index")
    def test_add_docs_after_init(self, mock_create_index, mock_load_documents):  # type: ignore[no-untyped-def]
        # Setup a dummy index with an insert method.
        dummy_index = MagicMock()
        dummy_index.insert = MagicMock()
        mock_create_index.return_value = dummy_index

        self.engine.init_db(input_dir="dummy_dir", collection_name="test_collection")
        self.engine.add_docs(new_doc_dir="dummy_dir")
        dummy_index.insert.assert_called()  # Ensure that insert was called.


class TestDoclingMongoAtlasMdQueryEngine(unittest.TestCase):
    def setUp(self):  # type: ignore[no-untyped-def]
        # Patch MongoClient so that no real connection is made.
        patcher = patch("autogen.agentchat.contrib.rag.docling_query_engine.MongoClient")
        self.addCleanup(patcher.stop)
        patcher.start()

        self.engine = DoclingMongoAtlasMdQueryEngine(
            connection_string="dummy_conn", database_name="db", collection_name="test_collection"
        )

    @patch("autogen.agentchat.contrib.rag.docling_query_engine.load_documents", return_value=[DummyDocument("doc")])  # type: ignore[no-untyped-def, no-untyped-call]
    @patch.object(DoclingMongoAtlasMdQueryEngine, "_create_index", return_value=MagicMock())
    def test_init_db(self, mock_create_index, mock_load_documents):  # type: ignore[no-untyped-def]
        self.engine.init_db(input_dir="dummy_dir", collection_name="test_collection")
        self.assertEqual(self.engine.collection_name, "test_collection")
        mock_create_index.assert_called()

    def test_query_without_init(self):  # type: ignore[no-untyped-def]
        with self.assertRaises(ValueError):
            self.engine.query("What is test?")

    @patch("autogen.agentchat.contrib.rag.docling_query_engine.load_documents", return_value=[DummyDocument("doc")])  # type: ignore[no-untyped-def, no-untyped-call]
    @patch.object(DoclingMongoAtlasMdQueryEngine, "_create_index")
    def test_query_after_init(self, mock_create_index, mock_load_documents):  # type: ignore[no-untyped-def]
        dummy_index = MagicMock()
        dummy_index.as_query_engine.return_value = DummyQueryEngine()
        mock_create_index.return_value = dummy_index

        self.engine.init_db(input_dir="dummy_dir", collection_name="test_collection")
        answer = self.engine.query("What is test?")
        self.assertEqual(answer, "Dummy answer to: What is test?")

    @patch("autogen.agentchat.contrib.rag.docling_query_engine.load_documents", return_value=[DummyDocument("doc")])  # type: ignore[no-untyped-def, no-untyped-call]
    @patch.object(DoclingMongoAtlasMdQueryEngine, "_create_index")
    def test_add_docs_after_init(self, mock_create_index, mock_load_documents):  # type: ignore[no-untyped-def]
        dummy_index = MagicMock()
        dummy_index.insert = MagicMock()
        mock_create_index.return_value = dummy_index

        self.engine.init_db(input_dir="dummy_dir", collection_name="test_collection")
        self.engine.add_docs(new_doc_dir="dummy_dir")
        dummy_index.insert.assert_called()


class TestDoclingQueryEngineFactory(unittest.TestCase):
    def test_factory_chroma(self):  # type: ignore[no-untyped-def]
        engine = DoclingQueryEngine(db_type="chroma", db_path="./tmp/chroma")
        from autogen.agentchat.contrib.rag.docling_query_engine import DoclingChromaMdQueryEngine  # ensure correct type

        self.assertIsInstance(engine.engine, DoclingChromaMdQueryEngine)

    @patch("autogen.agentchat.contrib.rag.docling_query_engine.MongoClient")
    def test_factory_mongodb(self, mock_mongo_client):  # type: ignore[no-untyped-def]
        # Ensure that a dummy MongoClient is returned so that no real connection is attempted.
        mock_mongo_client.return_value = MagicMock()
        engine = DoclingQueryEngine(db_type="mongodb", connection_string="dummy_conn")
        from autogen.agentchat.contrib.rag.docling_query_engine import (
            DoclingMongoAtlasMdQueryEngine,  # ensure correct type
        )

        self.assertIsInstance(engine.engine, DoclingMongoAtlasMdQueryEngine)

    def test_factory_invalid(self):  # type: ignore[no-untyped-def]
        with self.assertRaises(ValueError):
            DoclingQueryEngine(db_type="unsupported")


if __name__ == "__main__":
    unittest.main()
