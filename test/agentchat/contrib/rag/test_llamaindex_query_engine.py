# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import sys

import pytest

from autogen.agentchat.contrib.rag import LlamaIndexQueryEngine, RAGQueryEngine
from autogen.import_utils import optional_import_block, skip_on_missing_imports

from ....conftest import reason

with optional_import_block():
    from chromadb import HttpClient
    from llama_index.vector_stores.chroma import ChromaVectorStore

"""
This test file contains tests for the LlamaIndexQueryEngine class in the rag module.
Please set your OPENAI_API_KEY in your environment variables before running these tests.
"""


logger = logging.getLogger(__name__)
reason = "do not run on MacOS or windows OR dependency is not installed OR " + reason

input_dir = "/workspaces/ag2/test/agents/experimental/document_agent/pdf_parsed/"
input_docs = [input_dir + "nvidia_10k_2024.md"]
docs_to_add = [input_dir + "Toast_financial_report.md"]


@pytest.fixture(scope="module")
@pytest.mark.openai
@pytest.mark.skip(
    """Currently this test is not being run in CI.
    In test optional dependencies workflow, 'rag' is not added yet.
    https://github.com/ag2ai/ag2/issues/1383 issue is created to track this.
    """
)
@skip_on_missing_imports(["chromadb", "llama_index"], "rag")
def chroma_query_engine() -> LlamaIndexQueryEngine:
    # For testing purposes, use a host and port that point to your running ChromaDB.
    # Adjust these if necessary.
    chroma_client = HttpClient(
        host="172.17.0.3",
        port=8000,
    )
    # use get_collection to get an existing collection
    chroma_collection = chroma_client.get_collection("default_collection")
    chroma_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    chroma_query_engine = LlamaIndexQueryEngine(vector_store=chroma_vector_store)  # type: ignore[arg-type]
    return chroma_query_engine


@pytest.mark.openai
@pytest.mark.skipif(sys.platform in ["darwin", "win32"], reason=reason)
def test_lllamindex_query_engine_query(chroma_query_engine: LlamaIndexQueryEngine) -> None:
    """Test initialization and querying LlamaIndexQueryEngine instance."""
    chroma_query_engine.init_db(new_doc_paths_or_urls=input_docs)
    question = "How much money did Nvidia spend in research and development?"
    answer = chroma_query_engine.query(question)
    logger.info("Query answer: %s", answer)
    # If no meaningful answer is produced, the engine returns a default reply.
    assert answer.find("45.3 billion") != -1


@pytest.mark.openai
@pytest.mark.skipif(sys.platform in ["darwin", "win32"], reason=reason)
def test_llamaindex_query_engine_connect_db(chroma_query_engine: LlamaIndexQueryEngine) -> None:
    """Test connecting to an existing collection using connect_db."""
    logger.info("Testing connect_db of llamaindexQueryEngine")
    ret = chroma_query_engine.connect_db()
    assert ret is True


@pytest.mark.openai
@pytest.mark.skipif(sys.platform in ["darwin", "win32"], reason=reason)
def test_llamaindex_query_engine_add_docs(chroma_query_engine: LlamaIndexQueryEngine) -> None:
    """Test adding docs."""
    logger.info("Testing add_records of LlamaIndexQueryEngine")
    chroma_query_engine.add_docs(new_doc_paths_or_urls=docs_to_add)

    question = "How much money did Toast earn in 2024?"
    answer = chroma_query_engine.query(question)
    logger.info("Query answer: %s", answer)
    assert answer.find("$56 million") != -1 or answer.find("$13 million") != -1


def test_implements_protocol() -> None:
    assert issubclass(LlamaIndexQueryEngine, RAGQueryEngine)
