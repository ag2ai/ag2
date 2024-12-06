# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import sys
from typing import Literal

import pytest
from conftest import reason, skip_openai  # noqa: E402

try:
    from autogen.agentchat.contrib.graph_rag.document import Document, DocumentType
    from autogen.agentchat.contrib.graph_rag.neo4j_graph_query_engine import (
        GraphStoreQueryResult,
        Neo4jGraphQueryEngine,
    )

except ImportError:
    skip = True
else:
    skip = False

reason = "do not run on MacOS or windows OR dependency is not installed OR " + reason


# Test fixture for creating and initializing a query engine
@pytest.fixture(scope="module")
def neo4j_query_engine():
    input_path = "./test/agentchat/contrib/graph_rag/paul_graham_essay.txt"
    input_documents = [Document(doctype=DocumentType.TEXT, path_or_url=input_path)]

    # best practice to use upper-case
    entities = Literal["PERSON", "PLACE", "ORGANIZATION"]  #
    relations = Literal["HAS", "PART_OF", "WORKED_ON", "WORKED_WITH", "WORKED_AT"]

    # define which entities can have which relations
    validation_schema = {
        "PERSON": ["HAS", "PART_OF", "WORKED_ON", "WORKED_WITH", "WORKED_AT"],
        "PLACE": ["HAS", "PART_OF", "WORKED_AT"],
        "ORGANIZATION": ["HAS", "PART_OF", "WORKED_WITH"],
    }

    # Create Neo4jGraphQueryEngine
    query_engine = Neo4jGraphQueryEngine(
        username="neo4j",  # Change if you reset username
        password="password",  # Change if you reset password
        host="bolt://172.17.0.3",  # Change
        port=7687,  # if needed
        database="neo4j",  # Change if you want to store the graph in your custom database
        entities=entities,  # possible entities
        relations=relations,  # possible relations
        validation_schema=validation_schema,  # schema to validate the extracted triplets
        strict=True,  # enforce the extracted triplets to be in the schema
    )

    # Initialize the database
    query_engine.init_db(input_doc=input_documents)
    return query_engine


@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"] or skip or skip_openai,
    reason=reason,
)
def test_neo4j_query_engine(neo4j_query_engine):
    """
    Test querying functionality of the Neo4j Query Engine.
    """
    question = "Which companies did Paul Graham work for?"

    # Query the database
    query_result: GraphStoreQueryResult = neo4j_query_engine.query(question=question)

    print(query_result.answer)

    assert query_result.answer.find("Y Combinator") >= 0


@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"] or skip or skip_openai,
    reason=reason,
)
def test_neo4j_add_records(neo4j_query_engine):
    """
    Test the add_records functionality of the Neo4j Query Engine.
    """
    input_path = "./test/agentchat/contrib/graph_rag/the_matrix.txt"
    input_documents = [Document(doctype=DocumentType.TEXT, path_or_url=input_path)]

    # Add records to the existing graph
    _ = neo4j_query_engine.add_records(input_documents)

    # Verify the new data is in the graph
    question = "Who acted in 'The Matrix'?"
    query_result: GraphStoreQueryResult = neo4j_query_engine.query(question=question)

    print(query_result.answer)

    assert query_result.answer.find("Keanu Reeves") >= 0
