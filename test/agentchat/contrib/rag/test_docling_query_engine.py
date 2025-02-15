# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock, patch

from autogen.agentchat.contrib.rag.docling_query_engine import (
    DEFAULT_COLLECTION_NAME,
    DoclingChromaMdQueryEngine,
    DoclingMongoAtlasMdQueryEngine,
)


class TestDoclingChromaMdQueryEngine(unittest.TestCase):
    def setUp(self) -> None:
        # Initialize with test parameters; use a temporary directory for the db_path.
        self.engine = DoclingChromaMdQueryEngine(db_path="./test_chroma")

    @patch.object(DoclingChromaMdQueryEngine, "_load_doc", return_value=["doc1", "doc2"])
    @patch.object(DoclingChromaMdQueryEngine, "_create_index", return_value=MagicMock())
    def test_init_db(self, mock_create_index: MagicMock, mock_load_doc: MagicMock) -> None:
        self.engine.init_db(
            input_dir="dummy_dir",
            input_doc_paths=["file1.md"],
            collection_name="test_collection",
        )
        self.assertEqual(self.engine.collection_name, "test_collection")
        mock_load_doc.assert_called_once_with("dummy_dir", ["file1.md"])
        mock_create_index.assert_called_once()

    def test_query(self) -> None:
        # Create a fake query engine with a query method.
        fake_query_engine = MagicMock()
        fake_query_engine.query.return_value = "Test Answer"
        # Set the engine's index so that as_query_engine returns the fake query engine.
        self.engine.index = MagicMock()
        self.engine.index.as_query_engine.return_value = fake_query_engine

        response = self.engine.query("What is testing?")
        self.assertEqual(response, "Test Answer")
        self.engine.index.as_query_engine.assert_called_once_with(llm=self.engine.llm)
        fake_query_engine.query.assert_called_once_with("What is testing?")

    @patch.object(DoclingChromaMdQueryEngine, "_load_doc", return_value=["doc"])
    def test_add_docs_calls_load_doc(self, mock_load_doc: MagicMock) -> None:
        # Even though the method is incomplete, we can at least test that _load_doc gets called.
        self.engine.index = MagicMock()
        self.engine.add_docs(new_doc_dir="dummy_dir", new_doc_paths=["file.md"])
        mock_load_doc.assert_called_once_with("dummy_dir", ["file.md"])


class TestDoclingMongoAtlasMdQueryEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.mongo_engine = DoclingMongoAtlasMdQueryEngine(
            connection_string="mongodb://test_conn", database_name="test_db"
        )

    def test_instantiation(self) -> None:
        self.assertIsNotNone(self.mongo_engine)

    @unittest.skip("init_db not implemented, skipping test for now")
    def test_init_db(self) -> None:
        # This test would be implemented once init_db is complete.
        self.mongo_engine.init_db(input_dir="dummy_dir", input_doc_paths=["file.md"])
        self.assertEqual(self.mongo_engine.collection_name, DEFAULT_COLLECTION_NAME)

    @unittest.skip("query method not implemented, skipping test for now")
    def test_query(self) -> None:
        # This test will be expanded once query is implemented.
        response = self.mongo_engine.query("Test query")
        self.assertIsInstance(response, str)


if __name__ == "__main__":
    unittest.main()
