# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import logging
import sys
from typing import Literal

import pytest

from autogen.agentchat.contrib.graph_rag import Document, DocumentType, GraphStoreQueryResult
from autogen.agentchat.contrib.graph_rag.neo4j_graph_query_engine import Neo4jGraphQueryEngine
from autogen.import_utils import run_for_optional_imports

from ....conftest import reason

# Configure the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

reason = "do not run on MacOS or windows OR dependency is not installed OR " + reason


# Test fixture for creating and initializing a query engine with a JSON input file
@pytest.fixture(scope="module")
@run_for_optional_imports(["llama_index"], "neo4j")
def neo4j_query_engine_with_json() -> Neo4jGraphQueryEngine:
    input_path = "./test/agentchat/contrib/graph_rag/layout_parser_paper_parsed_elements.json"
    input_documents = [Document(doctype=DocumentType.JSON, path_or_url=input_path)]
    # Create Neo4jGraphQueryEngine
    query_engine = Neo4jGraphQueryEngine(
        username="neo4j",  # Change if you reset username
        password="password",  # Change if you reset password
        host="bolt://127.0.0.1",  # Change
        port=7687,  # if needed
        database="neo4j",  # Change if you want to store the graphh in your custom database
    )

    # Ingest data and create a new property graph
    query_engine.init_db(input_doc=input_documents)
    return query_engine


# Test fixture for creating and initializing a query engine
@pytest.fixture(scope="module")
def neo4j_query_engine() -> Neo4jGraphQueryEngine:
    input_path = "./test/agentchat/contrib/graph_rag/BUZZ_Employee_Handbook.docx"
    input_documents = [Document(doctype=DocumentType.TEXT, path_or_url=input_path)]

    # best practice to use upper-case
    entities = Literal[
        "EMPLOYEE", "EMPLOYER", "POLICY", "BENEFIT", "POSITION", "DEPARTMENT", "CONTRACT", "RESPONSIBILITY"
    ]
    relations = Literal[
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

    # define which entities can have which relations
    schema = {
        "EMPLOYEE": ["FOLLOWS", "APPLIES_TO", "ASSIGNED_TO", "ENTITLED_TO", "REPORTS_TO"],
        "EMPLOYER": ["PROVIDES", "DEFINED_AS", "MANAGES", "REQUIRES"],
        "POLICY": ["APPLIES_TO", "DEFINED_AS", "REQUIRES"],
        "BENEFIT": ["PROVIDES", "ENTITLED_TO"],
        "POSITION": ["DEFINED_AS", "PART_OF", "ASSIGNED_TO"],
        "DEPARTMENT": ["PART_OF", "MANAGES", "REQUIRES"],
        "CONTRACT": ["PROVIDES", "REQUIRES", "APPLIES_TO"],
        "RESPONSIBILITY": ["ASSIGNED_TO", "REQUIRES", "DEFINED_AS"],
    }

    # Create Neo4jGraphQueryEngine
    query_engine = Neo4jGraphQueryEngine(
        username="neo4j",  # Change if you reset username
        password="password",  # Change if you reset password
        host="bolt://127.0.0.1",  # Change
        port=7687,  # if needed
        database="neo4j",  # Change if you want to store the graphh in your custom database
        entities=entities,  # possible entities
        relations=relations,  # possible relations
        schema=schema,  # type: ignore[arg-type]
        strict=True,  # enofrce the extracted triplets to be in the schema
    )

    # Ingest data and initialize the database
    query_engine.init_db(input_doc=input_documents)
    return query_engine


# Test fixture to test auto-generation without given schema
@pytest.fixture(scope="module")
def neo4j_query_engine_auto() -> Neo4jGraphQueryEngine:
    """Test the engine with auto-generated property graph"""
    input_path = "./test/agentchat/contrib/graph_rag/BUZZ_Employee_Handbook.txt"

    input_document = [Document(doctype=DocumentType.TEXT, path_or_url=input_path)]

    query_engine = Neo4jGraphQueryEngine(
        username="neo4j",
        password="password",
        host="bolt://127.0.0.1",
        port=7687,
        database="neo4j",
    )
    query_engine.init_db(input_doc=input_document)
    return query_engine


@run_for_optional_imports("openai", "openai")
@pytest.mark.neo4j
@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"],
    reason=reason,
)
@run_for_optional_imports(["llama_index"], "neo4j")
def test_neo4j_query_engine(neo4j_query_engine: Neo4jGraphQueryEngine) -> None:
    """Test querying functionality of the Neo4j Query Engine."""
    question = "Which company is the employer?"

    # Query the database
    query_result: GraphStoreQueryResult = neo4j_query_engine.query(question=question)

    logger.info(query_result.answer)

    assert query_result.answer.find("BUZZ") >= 0  # type: ignore[union-attr]


@run_for_optional_imports("openai", "openai")
@pytest.mark.neo4j
@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"],
    reason=reason,
)
@run_for_optional_imports(["llama_index"], "neo4j")
def test_neo4j_add_records(neo4j_query_engine: Neo4jGraphQueryEngine) -> None:
    """Test the add_records functionality of the Neo4j Query Engine."""
    input_path = "./test/agentchat/contrib/graph_rag/the_matrix.txt"
    input_documents = [Document(doctype=DocumentType.TEXT, path_or_url=input_path)]

    # Add records to the existing graph
    _ = neo4j_query_engine.add_records(input_documents)

    # Verify the new data is in the graph
    question = "Who acted in 'The Matrix'?"
    query_result: GraphStoreQueryResult = neo4j_query_engine.query(question=question)

    logger.info(query_result.answer)

    assert query_result.answer.find("Keanu Reeves") >= 0  # type: ignore[union-attr]


@run_for_optional_imports("openai", "openai")
@pytest.mark.neo4j
@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"],
    reason=reason,
)
@run_for_optional_imports(["llama_index"], "neo4j")
def test_neo4j_auto(neo4j_query_engine_auto: Neo4jGraphQueryEngine) -> None:
    """Test querying with auto-generated property graph"""
    question = "Which company is the employer?"
    query_result: GraphStoreQueryResult = neo4j_query_engine_auto.query(question=question)

    logger.info(query_result.answer)
    assert query_result.answer.find("BUZZ") >= 0  # type: ignore[union-attr]


@run_for_optional_imports("openai", "openai")
@pytest.mark.neo4j
@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"],
    reason=reason,
)
@run_for_optional_imports(["llama_index"], "neo4j")
def test_neo4j_json_auto(neo4j_query_engine_with_json: Neo4jGraphQueryEngine) -> None:
    """Test querying with auto-generated property graph from a JSON file."""
    question = "What are current layout detection models in the LayoutParser model zoo?"
    query_result: GraphStoreQueryResult = neo4j_query_engine_with_json.query(question=question)

    logger.info(query_result.answer)
    assert query_result.answer.find("PRImA") >= 0  # type: ignore[union-attr]
