# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import logging
import sys

import pytest

from ....conftest import reason, skip_openai  # noqa: E402

try:
    from autogen.agentchat.contrib.graph_rag.document import Document, DocumentType
    from autogen.agentchat.contrib.graph_rag.neo4j_native_graph_query_engine import (
        GraphStoreQueryResult,
        Neo4jNativeGraphQueryEngine,
    )

except ImportError:
    skip = True
else:
    skip = False

# Configure the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

reason = "do not run on MacOS or windows OR dependency is not installed OR " + reason


# Test fixture for creating and initializing a query engine
@pytest.fixture(scope="module")
def neo4j_native_query_engine():
    input_path = "./test/agentchat/contrib/graph_rag/BUZZ_Employee_Handbook.txt"
    input_document = [Document(doctype=DocumentType.TEXT, path_or_url=input_path)]

    # best practice to use upper-case
    entities = ["EMPLOYEE", "EMPLOYER", "POLICY", "BENEFIT", "POSITION", "DEPARTMENT", "CONTRACT", "RESPONSIBILITY"]
    relations = [
        "FOLLOWS",
        "PROVIDES",
        "APPLIES_TO",
        "DEFINED_AS",
        "ASSIGNED_TO",
        "PART_OF",
        "MANAGES",
        "REQUIRES",
        "ENTITLED_TO",
        "REPORTS_TO",
    ]

    potential_schema = [
        ("EMPLOYEE", "FOLLOWS", "POLICY"),
        ("EMPLOYEE", "APPLIES_TO", "CONTRACT"),
        ("EMPLOYEE", "ASSIGNED_TO", "POSITION"),
        ("EMPLOYEE", "ENTITLED_TO", "BENEFIT"),
        ("EMPLOYEE", "REPORTS_TO", "EMPLOYER"),
        ("EMPLOYEE", "REPORTS_TO", "DEPARTMENT"),
        ("EMPLOYER", "PROVIDES", "BENEFIT"),
        ("EMPLOYER", "MANAGES", "DEPARTMENT"),
        ("EMPLOYER", "REQUIRES", "RESPONSIBILITY"),
        ("POLICY", "APPLIES_TO", "EMPLOYEE"),
        ("POLICY", "APPLIES_TO", "CONTRACT"),
        ("POLICY", "DEFINED_AS", "RESPONSIBILITY"),
        ("POLICY", "REQUIRES", "RESPONSIBILITY"),
        ("BENEFIT", "PROVIDES", "EMPLOYEE"),
        ("BENEFIT", "ENTITLED_TO", "EMPLOYEE"),
        ("POSITION", "PART_OF", "DEPARTMENT"),
        ("POSITION", "ASSIGNED_TO", "EMPLOYEE"),
        ("DEPARTMENT", "PART_OF", "EMPLOYER"),
        ("DEPARTMENT", "MANAGES", "EMPLOYEE"),
        ("CONTRACT", "PROVIDES", "EMPLOYEE"),
        ("CONTRACT", "REQUIRES", "RESPONSIBILITY"),
        ("CONTRACT", "APPLIES_TO", "EMPLOYEE"),
        ("RESPONSIBILITY", "ASSIGNED_TO", "POSITION"),
        ("RESPONSIBILITY", "DEFINED_AS", "POLICY"),
    ]

    query_engine = Neo4jNativeGraphQueryEngine(
        host="bolt://172.17.0.3",  # Change
        port=7687,  # if needed
        username="neo4j",  # Change if you reset username
        password="password",  # Change if you reset password
        entities=entities,
        relations=relations,
        potential_schema=potential_schema,
    )

    # Ingest data and initialize the database
    query_engine.init_db(input_doc=input_document)
    return query_engine


# Test fixture for auto generated knowledge graph
@pytest.fixture(scope="module")
def neo4j_native_query_engine_auto():
    input_path = "./test/agentchat/contrib/graph_rag/BUZZ_Employee_Handbook.txt"

    input_document = [Document(doctype=DocumentType.TEXT, path_or_url=input_path)]

    query_engine = Neo4jNativeGraphQueryEngine(
        host="bolt://172.17.0.3",  # Change
        port=7687,  # if needed
        username="neo4j",  # Change if you reset username
        password="password",  # Change if you reset password
    )

    # Ingest data and initialize the database
    query_engine.init_db(input_doc=input_document)
    return query_engine


@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"] or skip or skip_openai,
    reason=reason,
)
def test_neo4j_native_query_engine(neo4j_native_query_engine):
    """
    Test querying with initialized knowledge graph
    """
    question = "Which company is the employer?"
    query_result: GraphStoreQueryResult = neo4j_native_query_engine.query(question=question)

    logger.info(query_result.answer)
    assert query_result.answer.find("BUZZ") >= 0


@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"] or skip or skip_openai,
    reason=reason,
)
def test_neo4j_native_query_auto(neo4j_native_query_engine_auto):
    """
    Test querying with auto-generated property graph
    """
    question = "Which company is the employer?"
    query_result: GraphStoreQueryResult = neo4j_native_query_engine_auto.query(question=question)

    logger.info(query_result.answer)
    assert query_result.answer.find("BUZZ") >= 0


def test_neo4j_add_records(neo4j_native_query_engine):
    """
    Test the add_records functionality of the Neo4j Query Engine.
    """
    input_path = "./test/agentchat/contrib/graph_rag/the_matrix.txt"
    input_documents = [Document(doctype=DocumentType.TEXT, path_or_url=input_path)]

    # Add records to the existing graph
    _ = neo4j_native_query_engine.add_records(input_documents)

    # Verify the new data is in the graph
    question = "Who acted in 'The Matrix'?"
    query_result: GraphStoreQueryResult = neo4j_native_query_engine.query(question=question)

    logger.info(query_result.answer)

    assert query_result.answer.find("Keanu Reeves") >= 0
