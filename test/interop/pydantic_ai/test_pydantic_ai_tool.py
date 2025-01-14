# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
from pydantic_ai.tools import Tool as PydanticAITool

from autogen import AssistantAgent
from autogen.interop.pydantic_ai.pydantic_ai_tool import PydanticAITool as AG2PydanticAITool


# skip if python version is not >= 3.9
@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Only Python 3.9 and above are supported for LangchainInteroperability"
)
class TestPydanticAITool:
    def test_register_for_llm(self) -> None:
        def foobar(a: int, b: str, c: dict[str, list[float]]) -> str:  # type: ignore[misc]
            """Get me foobar.

            Args:
                a: apple pie
                b: banana cake
                c: carrot smoothie
            """
            return f"{a} {b} {c}"

        tool = PydanticAITool(foobar)  # type: ignore[var-annotated]
        ag2_tool = AG2PydanticAITool(
            name=tool.name,
            description=tool.description,
            func=tool.function,
            parameters_json_schema=tool._parameters_json_schema,
        )
        config_list = [{"model": "gpt-4o", "api_key": "abc"}]
        chatbot = AssistantAgent(
            name="chatbot",
            llm_config={"config_list": config_list},
        )
        ag2_tool.register_for_llm(chatbot)
        expected_tools = [
            {
                "type": "function",
                "function": {
                    "name": "foobar",
                    "description": "Get me foobar.",
                    "parameters": {
                        "properties": {
                            "a": {"description": "apple pie", "title": "A", "type": "integer"},
                            "b": {"description": "banana cake", "title": "B", "type": "string"},
                            "c": {
                                "additionalProperties": {"items": {"type": "number"}, "type": "array"},
                                "description": "carrot smoothie",
                                "title": "C",
                                "type": "object",
                            },
                        },
                        "required": ["a", "b", "c"],
                        "type": "object",
                        "additionalProperties": False,
                    },
                },
            }
        ]
        assert chatbot.llm_config["tools"] == expected_tools  # type: ignore[index]
