# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen import AssistantAgent
from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.tools import Tool

with optional_import_block():
    from pydantic_ai.tools import Tool as PydanticAITool


@pytest.mark.interop
@run_for_optional_imports("pydantic_ai", "interop-pydantic-ai")
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
        ag2_tool = Tool(
            name=tool.name,
            description=tool.description,
            func_or_tool=tool.function,
            parameters_json_schema=tool._parameters_json_schema,
        )
        config_list = [{"api_type": "openai", "model": "gpt-4o", "api_key": "abc"}]
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
