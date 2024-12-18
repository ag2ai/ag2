# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import os
import random
import sys
import unittest
from inspect import signature
from typing import Any, Dict, Optional

import pytest
from conftest import reason, skip_openai
from pydantic import BaseModel

from autogen import AssistantAgent, UserProxyAgent
from autogen.interop import Interoperable

if sys.version_info >= (3, 9):
    from pydantic_ai import RunContext
    from pydantic_ai.tools import Tool as PydanticAITool

    from autogen.interop.pydantic_ai import PydanticAIInteroperability
else:
    RunContext = unittest.mock.MagicMock()
    PydanticAITool = unittest.mock.MagicMock()
    PydanticAIInteroperability = unittest.mock.MagicMock()


# skip if python version is not >= 3.9
@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Only Python 3.9 and above are supported for LangchainInteroperability"
)
class TestPydanticAIInteroperabilityWithotContext:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        def roll_dice() -> str:
            """Roll a six-sided dice and return the result."""
            return str(random.randint(1, 6))

        self.pydantic_ai_interop = PydanticAIInteroperability()
        pydantic_ai_tool = PydanticAITool(roll_dice)
        self.tool = self.pydantic_ai_interop.convert_tool(pydantic_ai_tool)

    def test_type_checks(self) -> None:
        # mypy should fail if the type checks are not correct
        interop: Interoperable = self.pydantic_ai_interop
        # runtime check
        assert isinstance(interop, Interoperable)

    def test_init(self) -> None:
        assert isinstance(self.pydantic_ai_interop, Interoperable)

    def test_convert_tool(self) -> None:
        assert self.tool.name == "roll_dice"
        assert self.tool.description == "Roll a six-sided dice and return the result."
        assert self.tool.func() in ["1", "2", "3", "4", "5", "6"]

    @pytest.mark.skipif(skip_openai, reason=reason)
    def test_with_llm(self) -> None:
        config_list = [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]
        user_proxy = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
        )

        chatbot = AssistantAgent(
            name="chatbot",
            llm_config={"config_list": config_list},
        )

        self.tool.register_for_execution(user_proxy)
        self.tool.register_for_llm(chatbot)

        user_proxy.initiate_chat(recipient=chatbot, message="roll a dice", max_turns=2)

        for message in user_proxy.chat_messages[chatbot]:
            if "tool_responses" in message:
                assert message["tool_responses"][0]["content"] in ["1", "2", "3", "4", "5", "6"]
                return

        assert False, "No tool response found in chat messages"


class TestPydanticAIInteroperabilityWithContext:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        class Player(BaseModel):
            name: str
            age: int

        def get_player(ctx: RunContext[Player], additional_info: Optional[str] = None) -> str:  # type: ignore[valid-type]
            """Get the player's name.

            Args:
                additional_info: Additional information which can be used.
            """
            return f"Name: {ctx.deps.name}, Age: {ctx.deps.age}, Additional info: {additional_info}"  # type: ignore[attr-defined]

        self.pydantic_ai_interop = PydanticAIInteroperability()
        pydantic_ai_tool = PydanticAITool(get_player, takes_ctx=True)
        player = Player(name="Luka", age=25)
        self.tool = self.pydantic_ai_interop.convert_tool(tool=pydantic_ai_tool, deps=player)

    def test_dependency_injection(self) -> None:
        def f(
            city: str,
            date: str,
            ctx: int,
        ) -> str:
            return f"{city} {date} {ctx}"

        ctx = 123
        g = PydanticAIInteroperability.inject_params(f=f, ctx=ctx)
        assert list(signature(g).parameters.keys()) == ["city", "date"]
        kwargs: Dict[str, Any] = {"city": "Zagreb", "date": "2021-01-01"}
        assert g(**kwargs) == "Zagreb 2021-01-01 123"

    def test_signature(self) -> None:
        assert str(self.tool.func.__signature__) == "(additional_info: Optional[str] = None) -> str"

    @pytest.mark.skipif(skip_openai, reason=reason)
    def test_with_llm(self) -> None:
        config_list = [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]
        user_proxy = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
        )

        chatbot = AssistantAgent(
            name="chatbot",
            llm_config={"config_list": config_list},
        )

        self.tool.register_for_execution(user_proxy)
        self.tool.register_for_llm(chatbot)

        expected_tools = [
            {
                "type": "function",
                "function": {
                    "description": "Get the player's name.",
                    "name": "get_player",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "additional_info": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                                "description": "additional_info",
                            }
                        },
                        "required": [],
                    },
                },
            }
        ]

        assert chatbot.llm_config["tools"] == expected_tools  # type: ignore[index]

        user_proxy.initiate_chat(
            recipient=chatbot, message="Get player, for additional information use 'goal keeper'", max_turns=3
        )

        for message in user_proxy.chat_messages[chatbot]:
            if "tool_responses" in message:
                assert message["tool_responses"][0]["content"] == "Name: Luka, Age: 25, Additional info: goal keeper"
                return

        assert False, "No tool response found in chat messages"
