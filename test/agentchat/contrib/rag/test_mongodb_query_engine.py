# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# !/usr/bin/env python3 -m pytest

import logging
import os

import pytest

from autogen.agentchat.contrib.rag import MongoDBQueryEngine, RAGQueryEngine
from autogen.import_utils import skip_on_missing_imports

logger = logging.getLogger(__name__)
reason = "do not run on unsupported platforms or if dependencies are missing"

# Real file paths provided for testing.
input_dir = "test/agents/experimental/document_agent/pdf_parsed/"
input_docs = [os.path.join(input_dir, "Toast_financial_report.md")]
docs_to_add = [os.path.join(input_dir, "nvidia_10k_2024.md")]

# Use the connection string from an environment variable or fallback to the given connection string.
MONGO_CONN_STR = "mongodb://localhost:27017/?directConnection=true"


@pytest.fixture(scope="module")
@pytest.mark.openai
@skip_on_missing_imports(["pymongo", "llama_index"], "mongodb_query_engine")
def mongodb_query_engine() -> MongoDBQueryEngine:
    """
    Fixture that creates a MongoDBQueryEngine instance and initializes it using real document files.
    """
    engine = MongoDBQueryEngine(
        connection_string=MONGO_CONN_STR,
        database_name="test_db",
        collection_name="docling-parsed-docs",
    )
    ret = engine.init_db(new_doc_paths_or_urls=input_docs)
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
    mongodb_query_engine.add_docs(new_doc_paths_or_urls=input_docs)
    question = "What is the name of the exchange on which TOST has registered"
    answer = mongodb_query_engine.query(question)
    logger.info("Query answer: %s", answer)
    assert "New York Stock Exchange" in answer


@pytest.mark.openai
def test_mongodb_query_engine_connect_db() -> None:
    """Test connecting to an existing collection using connect_db."""
    engine = MongoDBQueryEngine(
        connection_string=MONGO_CONN_STR,
        database_name="test_db",
        collection_name="docling-parsed-docs",
    )
    ret = engine.connect_db()
    assert ret is True


@pytest.mark.openai
def test_mongodb_query_engine_add_docs(mongodb_query_engine: MongoDBQueryEngine) -> None:
    """Test adding new documents with add_docs to the existing collection."""
    mongodb_query_engine.add_docs(new_doc_paths_or_urls=docs_to_add)
    # After adding docs, query for information expected to be in the added document.
    question = (
        "What is the maximum percentage of earnings that may be withheld under the 2012 Plan to purchase common stock?"
    )
    answer = mongodb_query_engine.query(question)
    logger.info("Query answer: %s", answer)
    assert "maximum" in answer


def test_implements_protocol() -> None:
    """Test that MongoDBQueryEngine implements the RAGQueryEngine protocol."""
    assert issubclass(MongoDBQueryEngine, RAGQueryEngine)
