# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context
from ag2.config.zai.mappers import tool_to_api
from ag2.tools.builtin.retrieval import RetrievalTool


@pytest.mark.asyncio
async def test_defaults(context: Context) -> None:
    tool = RetrievalTool("kb_123")

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {"type": "retrieval", "retrieval": {"knowledge_id": "kb_123"}}


@pytest.mark.asyncio
async def test_with_prompt_template(context: Context) -> None:
    tool = RetrievalTool("kb_123", prompt_template="Use {{ knowledge }} to answer {{ question }}.")

    [schema] = await tool.schemas(context)

    assert tool_to_api(schema) == {
        "type": "retrieval",
        "retrieval": {
            "knowledge_id": "kb_123",
            "prompt_template": "Use {{ knowledge }} to answer {{ question }}.",
        },
    }
