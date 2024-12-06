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


# if you are not running the test in home directory, please change input file path.
# You also need to have an OpenAI key in your environment variable `OPENAI_API_KEY`.
# If you see an Assertion error inside neo4j query engine, please just rerun the test
@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"] or skip or skip_openai,
    reason=reason,
)
def test_neo4j_query_engine():
    """
    Test Neo4j Query Engine.
    1. create a test Neo4j Query Engine with a predefined schema.
    2. Initialize it with an input txt file.
    3. Query it with a question and verify the result contains the critical information.
    """

    #
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

    # Create FalkorGraphQueryEngine
    query_engine = Neo4jGraphQueryEngine(
        username="neo4j",  # Change if you reset username
        password="password",  # Change if you reset password
        host="bolt://172.17.0.2",  # Change
        port=7687,  # if needed
        database="neo4j",  # Change if you want to store the graphh in your custom database
        entities=entities,  # possible entities
        relations=relations,  # possible relations
        validation_schema=validation_schema,  # schema to validate the extracted triplets
        strict=True,  # enofrce the extracted triplets to be in the schema
    )

    # Ingest data and initialize the database
    query_engine.init_db(input_doc=input_documents)

    question = "Which companies did Paul Graham work for?"

    # Query the database
    query_result: GraphStoreQueryResult = query_engine.query(question=question)

    print(query_result.answer)

    assert query_result.answer.find("Y Combinator") >= 0
