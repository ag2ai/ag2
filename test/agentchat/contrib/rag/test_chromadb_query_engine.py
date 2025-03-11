# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import sys

import pytest

from autogen.agentchat.contrib.rag import ChromaDBQueryEngine, RAGQueryEngine
from autogen.import_utils import skip_on_missing_imports

from ....conftest import reason

"""
This test file contains tests for the ChromaDBQueryEngine class in the ChromaDBQueryEngine module.
Please set your OPENAI_API_KEY in your environment variables before running these tests.
"""

logger = logging.getLogger(__name__)
reason = "do not run on MacOS or windows OR dependency is not installed OR " + reason

input_dir = "/workspaces/ag2/test/agents/experimental/document_agent/pdf_parsed/"
input_docs = [input_dir + "nvidia_10k_2024.md"]
docs_to_add = [input_dir + "Toast_financial_report.md"]


@pytest.fixture(scope="module")
@pytest.mark.openai
@skip_on_missing_imports(["chromadb", "llama_index"], "rag")
def chroma_query_engine() -> ChromaDBQueryEngine:
    # For testing purposes, use a host and port that point to your running ChromaDB.
    # Adjust these if necessary.
    engine = ChromaDBQueryEngine(
        host="172.17.03",  #
        port=8000,
    )
    ret = engine.init_db(new_doc_paths_or_urls=input_docs)  # type: ignore[arg-type]
    assert ret is True
    return engine


@pytest.mark.skipif(sys.platform in ["darwin", "win32"], reason=reason)
@pytest.mark.openai
def test_get_collection_name(chroma_query_engine: ChromaDBQueryEngine) -> None:
    """Test getting the default collection name of the ChromaDBQueryEngine."""
    logger.info("Testing ChromaDBQueryEngine construction")
    collection_name = chroma_query_engine.get_collection_name()
    logger.info("Default collection name: %s", collection_name)
    assert collection_name == "docling-parsed-docs"


@pytest.mark.openai
@pytest.mark.skipif(sys.platform in ["darwin", "win32"], reason=reason)
def test_chroma_db_query_engine_query(chroma_query_engine: ChromaDBQueryEngine) -> None:
    """Test the querying functionality of the ChromaDBQueryEngine."""
    question = "How much money did Nvidia spend in research and development?"
    answer = chroma_query_engine.query(question)
    logger.info("Query answer: %s", answer)
    # If no meaningful answer is produced, the engine returns a default reply.
    assert answer.find("45.3 billion") != -1


@pytest.mark.openai
@pytest.mark.skipif(sys.platform in ["darwin", "win32"], reason=reason)
def test_chroma_db_query_engine_connect_db() -> None:
    """Test connecting to an existing collection using connect_db."""
    logger.info("Testing connect_db of ChromaDBQueryEngine")
    # Initialize first to create a collection
    engine = ChromaDBQueryEngine(
        host="172.17.03",  #
        port=8000,
    )
    ret = engine.connect_db()
    assert ret is True

    question = "How much money did Nvidia spend in research and development?"
    answer = engine.query(question)
    logger.info("Query answer: %s", answer)
    assert answer.find("45.3 billion") != -1


@pytest.mark.openai
@pytest.mark.skipif(sys.platform in ["darwin", "win32"], reason=reason)
def test_chroma_db_query_engine_add_docs(chroma_query_engine: ChromaDBQueryEngine) -> None:
    """Test adding records with add_docs to the existing collection."""
    logger.info("Testing add_records of ChromaDBQueryEngine")
    chroma_query_engine.add_docs(new_doc_paths_or_urls=docs_to_add)

    question = "How much money did Toast earn in 2024?"
    answer = chroma_query_engine.query(question)
    logger.info("Query answer: %s", answer)
    assert answer.find("$56 million") != -1 or answer.find("$13 million") != -1


def test_implements_protocol() -> None:
    assert issubclass(ChromaDBQueryEngine, RAGQueryEngine)
