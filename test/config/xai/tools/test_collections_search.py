# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context
from ag2.config.xai.mappers import tool_to_api
from ag2.tools.builtin.collections_search import CollectionsSearchTool


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = CollectionsSearchTool(collection_ids=["c1", "c2"])

    [schema] = await tool.schemas(context)
    api = tool_to_api(schema)

    assert api.HasField("collections_search")
    assert list(api.collections_search.collection_ids) == ["c1", "c2"]
    assert api.collections_search.WhichOneof("retrieval_mode") is None


@pytest.mark.asyncio
async def test_all_options(context: Context) -> None:
    tool = CollectionsSearchTool(
        collection_ids=["c1"],
        limit=5,
        instructions="rank by recency",
        retrieval_mode="semantic",
    )

    [schema] = await tool.schemas(context)
    api = tool_to_api(schema)

    cs = api.collections_search
    assert list(cs.collection_ids) == ["c1"]
    assert cs.limit == 5
    assert cs.instructions == "rank by recency"
    assert cs.WhichOneof("retrieval_mode") == "semantic_retrieval"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "oneof"),
    [
        ("hybrid", "hybrid_retrieval"),
        ("semantic", "semantic_retrieval"),
        ("keyword", "keyword_retrieval"),
    ],
)
async def test_retrieval_modes(context: Context, mode: str, oneof: str) -> None:
    tool = CollectionsSearchTool(collection_ids=["c1"], retrieval_mode=mode)

    [schema] = await tool.schemas(context)
    api = tool_to_api(schema)

    assert api.collections_search.WhichOneof("retrieval_mode") == oneof
