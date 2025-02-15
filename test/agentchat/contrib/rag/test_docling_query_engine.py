# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest
from typing import Any, List
from unittest.mock import patch

# Import the engines from the module.
from autogen.agentchat.contrib.rag.docling_query_engine import (
    DoclingChromaMdQueryEngine,
    DoclingMongoAtlasMdQueryEngine,
)

# =============================================================================
# Dummy implementations to override external dependencies.
# =============================================================================


class DummyQueryEngine:
    def query(self, question: str) -> str:
        # Simply return a dummy answer that includes the question
        return f"dummy answer: {question}"


class DummyIndex:
    def __init__(self) -> None:
        self.inserted_docs: List[Any] = []  # track inserted docs

    def as_query_engine(self, llm: Any) -> DummyQueryEngine:
        return DummyQueryEngine()

    def insert(self, doc: Any) -> None:
        self.inserted_docs.append(doc)


# Dummy classes for simulating MongoDB client behavior.
class DummyMongoCollection:
    # You can add more dummy methods if needed.
    pass


class DummyMongoDatabase:
    def __getitem__(self, collection_name: str) -> DummyMongoCollection:
        # Return a dummy collection regardless of the name.
        return DummyMongoCollection()

    def list_collection_names(self) -> List[str]:
        # Return a list with a dummy collection name.
        return ["dummy_collection"]


class DummyMongoClient:
    def __getitem__(self, db_name: str) -> DummyMongoDatabase:
        # Return a dummy database regardless of the db name.
        return DummyMongoDatabase()


# =============================================================================
# Tests for DoclingChromaMdQueryEngine
# =============================================================================


class TestDoclingChromaMdQueryEngine(unittest.TestCase):
    def setUp(self) -> None:
        # Create a temporary directory and a dummy markdown file.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.doc_path = os.path.join(self.temp_dir.name, "dummy.md")
        with open(self.doc_path, "w") as f:
            f.write("This is a dummy document for Chroma engine testing. Nvidia spent $1B on R&D.")

        # Create an instance of the Chroma engine.
        # (db_path points to a temporary location so no real persistent data is written.)
        self.engine = DoclingChromaMdQueryEngine(db_path="./tmp/chroma")
        # Override the _create_index method so that it returns a DummyIndex.
        # Use type: ignore to bypass assignment-to-method warnings.
        self.engine._create_index = lambda collection, docs: DummyIndex()  # type: ignore

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_init_db_with_directory(self) -> None:
        # Test that initializing the DB with a directory returns a valid collection name.
        self.engine.init_db(input_dir=self.temp_dir.name)
        collection_name = self.engine.get_collection_name()
        self.assertIsNotNone(collection_name)
        # By default, if no collection name is provided, the engine should use the default.
        self.assertEqual(collection_name, self.engine.collection_name)

    def test_query(self) -> None:
        # After initializing the index, querying should return our dummy answer.
        self.engine.init_db(input_dir=self.temp_dir.name)
        question = "How much did Nvidia invest in R&D?"
        answer = self.engine.query(question)
        self.assertIn(question, answer)  # dummy answer includes the question text

    def test_add_docs(self) -> None:
        # Test that calling add_docs leads to new documents being inserted into the dummy index.
        self.engine.init_db(input_dir=self.temp_dir.name)
        dummy_index = self.engine.index
        # Initially, no docs should have been inserted via add_docs.
        self.assertEqual(len(dummy_index.inserted_docs), 0)

        # Create another dummy markdown file for adding documents.
        new_doc_path = os.path.join(self.temp_dir.name, "dummy2.md")
        with open(new_doc_path, "w") as f:
            f.write("Additional document about Toast financial report.")

        # Call add_docs with the new file.
        self.engine.add_docs(new_doc_paths=[new_doc_path])
        # Verify that our dummy index recorded the inserted document(s).
        self.assertGreater(len(dummy_index.inserted_docs), 0)

    def test_load_doc_no_input(self) -> None:
        # Calling _load_doc with no input should raise a ValueError.
        with self.assertRaises(ValueError):
            self.engine._load_doc("", [])


# =============================================================================
# Tests for DoclingMongoAtlasMdQueryEngine
# =============================================================================


class TestDoclingMongoAtlasMdQueryEngine(unittest.TestCase):
    def setUp(self) -> None:
        # Patch MongoClient to return a dummy client that supports subscripting.
        self.mongo_patch = patch(
            "autogen.agentchat.contrib.rag.docling_query_engine.MongoClient", return_value=DummyMongoClient()
        )
        self.mongo_patch.start()

        # Create a temporary directory with a dummy markdown file.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.doc_path = os.path.join(self.temp_dir.name, "dummy.md")
        with open(self.doc_path, "w") as f:
            f.write("This is a dummy document for MongoAtlas testing. Nvidia spent $2B on R&D.")

        # Create an instance of the MongoAtlas engine with dummy connection parameters.
        self.engine = DoclingMongoAtlasMdQueryEngine(
            connection_string="dummy_connection_string", database_name="dummy_db", collection_name="dummy_collection"
        )
        # Override _create_index to return a DummyIndex.
        self.engine._create_index = lambda collection, docs: DummyIndex()  # type: ignore

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        self.mongo_patch.stop()

    def test_init_db_with_directory(self) -> None:
        self.engine.init_db(input_dir=self.temp_dir.name)
        collection_name = self.engine.get_collection_name()
        self.assertIsNotNone(collection_name)
        self.assertEqual(collection_name, self.engine.collection_name)

    def test_query(self) -> None:
        self.engine.init_db(input_dir=self.temp_dir.name)
        question = "What did Nvidia invest in R&D?"
        answer = self.engine.query(question)
        self.assertIn(question, answer)

    def test_add_docs(self) -> None:
        self.engine.init_db(input_dir=self.temp_dir.name)
        dummy_index = self.engine.index
        self.assertEqual(len(dummy_index.inserted_docs), 0)  # type: ignore[attr-defined]

        new_doc_path = os.path.join(self.temp_dir.name, "dummy2.md")
        with open(new_doc_path, "w") as f:
            f.write("Additional document for MongoAtlas engine.")
        self.engine.add_docs(new_doc_paths=[new_doc_path])
        self.assertGreater(len(dummy_index.inserted_docs), 0)  # type: ignore[attr-defined]

    def test_load_doc_no_input(self) -> None:
        with self.assertRaises(ValueError):
            self.engine._load_doc("", [])


# =============================================================================
# Run the tests
# =============================================================================

if __name__ == "__main__":
    unittest.main()
