# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config.openai import OpenAIResponsesConfig
from ag2.events import ModelRequest, TextInput
from ag2.tools.builtin.file_search import FileSearchTool
from ag2.tools.builtin.shell import ShellTool
from ag2.tools.builtin.skills import Skill, SkillsTool


def _capturing_config(captured: dict[str, object]) -> OpenAIResponsesConfig:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "model": "m",
                "status": "completed",
                "output": [],
                "usage": None,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "instructions": None,
                "metadata": {},
                "text": {"format": {"type": "text"}},
            },
        )

    return OpenAIResponsesConfig(
        model="m",
        api_key="test",
        base_url="http://test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def _request_body(captured: dict[str, object], tools: list) -> dict[str, object]:
    context = Context(stream=MemoryStream())
    schemas = []
    for t in tools:
        schemas.extend(await t.schemas(context))
    client = _capturing_config(captured).create()
    await client(
        messages=[ModelRequest([TextInput("hi")])],
        context=context,
        tools=schemas,
        response_schema=None,
        serializer=SerializerCls,
    )
    return captured["body"]  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_skills_merge_into_existing_shell() -> None:
    captured: dict[str, object] = {}

    body = await _request_body(captured, [ShellTool(), SkillsTool("skill_abc", Skill("skill_def", version="2"))])

    assert body["tools"] == [
        {
            "type": "shell",
            "environment": {
                "type": "container_auto",
                "skills": [
                    {"type": "skill_reference", "skill_id": "skill_abc"},
                    {"type": "skill_reference", "skill_id": "skill_def", "version": "2"},
                ],
            },
        },
    ]


@pytest.mark.asyncio
async def test_skills_auto_add_shell() -> None:
    captured: dict[str, object] = {}

    body = await _request_body(captured, [SkillsTool("skill_abc")])

    assert body["tools"] == [
        {
            "type": "shell",
            "environment": {
                "type": "container_auto",
                "skills": [{"type": "skill_reference", "skill_id": "skill_abc"}],
            },
        },
    ]


@pytest.mark.asyncio
async def test_file_search_include_results_sets_include() -> None:
    captured: dict[str, object] = {}

    body = await _request_body(captured, [FileSearchTool(vector_store_ids=["vs_1"], include_results=True)])

    assert body["include"] == ["file_search_call.results"]
    assert {"type": "file_search", "vector_store_ids": ["vs_1"]} in body["tools"]
