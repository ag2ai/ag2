# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import sys

import pytest
from graphrag_sdk import Attribute, AttributeType, Entity, Ontology, Relation

from ....conftest import reason, skip_openai  # noqa: E402

try:
    from autogen.agentchat.contrib.graph_rag.document import Document, DocumentType
    from autogen.agentchat.contrib.graph_rag.falkor_graph_query_engine import (
        FalkorGraphQueryEngine,
        GraphStoreQueryResult,
    )
except ImportError:
    skip = True
else:
    skip = False

reason = "do not run on MacOS or windows OR dependency is not installed OR " + reason


@pytest.mark.skipif(
    sys.platform in ["darwin", "win32"] or skip or skip_openai,
    reason=reason,
)
def test_falkor_db_query_engine():
    """
    Test FalkorDB Query Engine.
    1. create a test FalkorDB Query Engine with a schema.
    2. Initialize it with an input txt file.
    3. Query it with a question and verify the result contains the critical information.
    """
    # Arrange
    movie_ontology = Ontology()
    movie_ontology.add_entity(
        Entity(label="Actor", attributes=[Attribute(name="name", attr_type=AttributeType.STRING, unique=True)])
    )
    movie_ontology.add_entity(
        Entity(label="Movie", attributes=[Attribute(name="title", attr_type=AttributeType.STRING, unique=True)])
    )
    movie_ontology.add_relation(Relation(label="ACTED", source="Actor", target="Movie"))

    query_engine = FalkorGraphQueryEngine(
        name="IMDB",
        # host="192.168.0.115",     # Change
        # port=6379,                # if needed
        ontology=movie_ontology,
    )

    source_file = "test/agentchat/contrib/graph_rag/the_matrix.txt"
    input_docs = [Document(doctype=DocumentType.TEXT, path_or_url=source_file)]

    question = "Name a few actors who've played in 'The Matrix'"

    # Act
    query_engine.init_db(input_doc=input_docs)

    query_result: GraphStoreQueryResult = query_engine.query(question=question)

    # Assert
    assert query_result.answer.find("Keanu Reeves") >= 0
