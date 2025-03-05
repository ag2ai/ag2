# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest
import logging
import os
from pathlib import Path

import pytest

from autogen.agentchat.contrib.rag import MongoDBQueryEngine, RAGQueryEngine
from autogen.import_utils import skip_on_missing_imports

logger = logging.getLogger(__name__)
reason = "do not run on unsupported platforms or if dependencies are missing"

# Use the connection string from an environment variable or default to localhost.
MONGO_CONN_STR = os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017")


@pytest.fixture(scope="module")
@pytest.mark.openai
@skip_on_missing_imports(["pymongo", "llama_index", "langchain_openai"], "mongodb_query_engine")
def mongodb_query_engine(tmp_path: Path) -> MongoDBQueryEngine:
    """
    Fixture that creates a MongoDBQueryEngine instance and initializes it with a dummy document.

    The dummy document contains known content used to verify query responses.
    """
    # Create a temporary document file with known content.
    doc_file = tmp_path / "dummy_doc.md"
    doc_file.write_text("Company X spent 45.3 billion on research and development.")

    engine = MongoDBQueryEngine(
        connection_string=MONGO_CONN_STR,
        database_name="test_db",
        collection_name="docling-parsed-docs",
    )
    ret = engine.init_db(new_doc_paths_or_urls=[str(doc_file)])
    assert ret is True
    return engine


@pytest.mark.openai
def test_get_collection_name(mongodb_query_engine: MongoDBQueryEngine) -> None:
    """Test getting the default collection name of the MongoDBQueryEngine."""
    logger.info("Testing MongoDBQueryEngine get_collection_name")
    collection_name = mongodb_query_engine.get_collection_name()
    logger.info("Default collection name: %s", collection_name)
    assert collection_name == "docling-parsed-docs"


@pytest.mark.openai
def test_mongodb_query_engine_query(mongodb_query_engine: MongoDBQueryEngine) -> None:
    """Test the querying functionality of the MongoDBQueryEngine."""
    question = "How much did Company X spend on research and development?"
    answer = mongodb_query_engine.query(question)
    logger.info("Query answer: %s", answer)
    # The dummy document contains "45.3 billion", so expect that substring in the answer.
    assert "45.3 billion" in answer


@pytest.mark.openai
def test_mongodb_query_engine_connect_db(tmp_path: Path) -> None:
    """Test connecting to an existing collection using connect_db."""
    # Create a temporary document file for connection test.
    doc_file = tmp_path / "dummy_doc_connect.md"
    doc_file.write_text("Company X spent 45.3 billion on research and development.")

    engine = MongoDBQueryEngine(
        connection_string=MONGO_CONN_STR,
        database_name="test_db",
        collection_name="docling-parsed-docs",
    )
    ret = engine.connect_db()
    assert ret is True

    question = "How much did Company X spend on research and development?"
    answer = engine.query(question)
    logger.info("Query answer: %s", answer)
    assert "45.3 billion" in answer


@pytest.mark.openai
def test_mongodb_query_engine_add_docs(tmp_path: Path, mongodb_query_engine: MongoDBQueryEngine) -> None:
    """Test adding new documents with add_docs to the existing collection."""
    # Create a temporary additional document file with known content.
    new_doc_file = tmp_path / "dummy_doc_add.md"
    new_doc_file.write_text("Company Y earned $56 million in Q1 2024.")

    mongodb_query_engine.add_docs(new_doc_paths_or_urls=[str(new_doc_file)])

    question = "How much did Company Y earn in Q1 2024?"
    answer = mongodb_query_engine.query(question)
    logger.info("Query answer: %s", answer)
    assert "$56 million" in answer


def test_implements_protocol() -> None:
    """Test that MongoDBQueryEngine implements the RAGQueryEngine protocol."""
    assert issubclass(MongoDBQueryEngine, RAGQueryEngine)
