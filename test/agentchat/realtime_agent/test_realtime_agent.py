# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

from autogen.agentchat.realtime.experimental import RealtimeAgent
from autogen.import_utils import skip_on_missing_imports
from autogen.tools.tool import Tool

from ...conftest import Credentials


def f(a: int, b: int = 3) -> int:
    return a + b


async def f_async(a: int, b: int = 3) -> int:
    return a + b


class A:
    def f(self, a: int, b: int = 3) -> int:
        return a + b

    async def f_async(self, a: int, b: int = 3) -> int:
        return a + b

    @staticmethod
    def f_static(a: int, b: int = 3) -> int:
        return a + b

    @staticmethod
    async def f_static_async(a: int, b: int = 3) -> int:
        return a + b


@skip_on_missing_imports("openai", "openai")
class TestRealtimeAgent:
    @pytest.fixture
    def agent(self, mock_credentials: Credentials) -> RealtimeAgent:
        return RealtimeAgent(
            name="realtime_agent",
            llm_config=mock_credentials.llm_config,
            audio_adapter=MagicMock(),
        )

    @pytest.fixture
    def expected_tools(self) -> dict[str, Any]:
        return {
            "type": "function",
            "description": "Example function",
            "name": "f",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "a"},
                    "b": {"type": "integer", "description": "b", "default": 3},
                },
                "required": ["a"],
            },
        }

    @pytest.mark.parametrize(
        ("func", "func_name", "is_async", "expected"),
        [
            (f, "f", False, 4),
            (f_async, "f_async", True, 4),
            (A().f, "f", False, 4),
            (A().f_async, "f_async", True, 4),
            (A.f_static, "f_static", False, 4),
            (A.f_static_async, "f_static_async", True, 4),
        ],
    )
    @pytest.mark.asyncio
    async def test_register_tools(
        self,
        agent: RealtimeAgent,
        expected_tools: dict[str, Any],
        func: Callable[..., Any],
        func_name: str,
        is_async: bool,
        expected: str,
    ) -> None:
        agent.register_realtime_function(description="Example function")(func)

        assert isinstance(agent._registered_realtime_tools[func_name], Tool)

        expected_tools["name"] = func_name
        assert agent._registered_realtime_tools[func_name].realtime_tool_schema == expected_tools

        retval = agent._registered_realtime_tools[func_name].func(1)
        actual = await retval if is_async else retval

        assert actual == expected
