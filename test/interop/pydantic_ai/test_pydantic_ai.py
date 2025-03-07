# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import random
from inspect import signature
from typing import Any, Optional

import pytest
from pydantic import BaseModel

from autogen import AssistantAgent, UserProxyAgent
from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.interop import Interoperable
from autogen.interop.pydantic_ai import PydanticAIInteroperability

from ...conftest import Credentials

with optional_import_block():
    from pydantic_ai import RunContext
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import Tool as PydanticAITool
    from pydantic_ai.usage import Usage


@pytest.mark.interop
@run_for_optional_imports("pydantic_ai", "interop-pydantic-ai")
class TestPydanticAIInteroperabilityWithotContext:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        def roll_dice() -> str:
            """Roll a six-sided dice and return the result."""
            return str(random.randint(1, 6))

        pydantic_ai_tool = PydanticAITool(roll_dice, max_retries=3)  # type: ignore[var-annotated]
        self.tool = PydanticAIInteroperability.convert_tool(pydantic_ai_tool)

    def test_type_checks(self) -> None:
        # mypy should fail if the type checks are not correct
        interop: Interoperable = PydanticAIInteroperability()
        # runtime check
        assert isinstance(interop, Interoperable)

    def test_convert_tool(self) -> None:
        assert self.tool.name == "roll_dice"
        assert self.tool.description == "Roll a six-sided dice and return the result."
        assert self.tool.func() in ["1", "2", "3", "4", "5", "6"]

    @run_for_optional_imports("openai", "openai")
    def test_with_llm(self, credentials_gpt_4o: Credentials, user_proxy: UserProxyAgent) -> None:
        chatbot = AssistantAgent(name="chatbot", llm_config=credentials_gpt_4o.llm_config)

        self.tool.register_for_execution(user_proxy)
        self.tool.register_for_llm(chatbot)

        user_proxy.initiate_chat(recipient=chatbot, message="roll a dice", max_turns=2)

        for message in user_proxy.chat_messages[chatbot]:
            if "tool_responses" in message:
                assert message["tool_responses"][0]["content"] in ["1", "2", "3", "4", "5", "6"]
                return

        assert False, "No tool response found in chat messages"


@pytest.mark.interop
@run_for_optional_imports("pydantic_ai", "interop-pydantic-ai")
class TestPydanticAIInteroperabilityDependencyInjection:
    def test_dependency_injection(self) -> None:
        def f(  # type: ignore[no-any-unimported]
            ctx: RunContext[int],  # type: ignore[valid-type]
            city: str,
            date: str,
        ) -> str:
            """Random function for testing."""
            return f"{city} {date} {ctx.deps}"  # type: ignore[attr-defined]

        ctx = RunContext(
            model=TestModel(),
            usage=Usage(),
            prompt="",
            deps=123,
            retry=0,
            messages=None,  # type: ignore[arg-type]
            tool_name=f.__name__,
        )
        pydantic_ai_tool = PydanticAITool(f, takes_ctx=True)  # type: ignore[var-annotated]
        g = PydanticAIInteroperability.inject_params(
            ctx=ctx,
            tool=pydantic_ai_tool,
        )
        assert list(signature(g).parameters.keys()) == ["city", "date"]
        kwargs: dict[str, Any] = {"city": "Zagreb", "date": "2021-01-01"}
        assert g(**kwargs) == "Zagreb 2021-01-01 123"

    def test_dependency_injection_with_retry(self) -> None:
        def f(  # type: ignore[no-any-unimported]
            ctx: RunContext[int],  # type: ignore[valid-type]
            city: str,
            date: str,
        ) -> str:
            """Random function for testing."""
            raise ValueError("Retry")

        ctx = RunContext(
            model=TestModel(),
            usage=Usage(),
            prompt="",
            deps=123,
            retry=0,
            messages=None,  # type: ignore[arg-type]
            tool_name=f.__name__,
        )

        pydantic_ai_tool = PydanticAITool(f, takes_ctx=True, max_retries=3)  # type: ignore[var-annotated]
        g = PydanticAIInteroperability.inject_params(
            ctx=ctx,
            tool=pydantic_ai_tool,
        )

        for i in range(3):
            with pytest.raises(ValueError, match="Retry"):
                g(city="Zagreb", date="2021-01-01")
                assert pydantic_ai_tool.current_retry == i + 1
                assert ctx.retry == i

        with pytest.raises(ValueError, match="f failed after 3 retries"):
            g(city="Zagreb", date="2021-01-01")
            assert pydantic_ai_tool.current_retry == 3


@pytest.mark.interop
@run_for_optional_imports("pydantic_ai", "interop-pydantic-ai")
class TestPydanticAIInteroperabilityWithContext:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        class Player(BaseModel):
            name: str
            age: int

        def get_player(ctx: RunContext[Player], additional_info: Optional[str] = None) -> str:  # type: ignore[valid-type,no-any-unimported]
            """Get the player's name.

            Args:
                ctx: The context object.
                additional_info: Additional information which can be used.
            """
            return f"Name: {ctx.deps.name}, Age: {ctx.deps.age}, Additional info: {additional_info}"  # type: ignore[attr-defined]

        self.pydantic_ai_tool = PydanticAITool(get_player, takes_ctx=True)  # type: ignore[var-annotated]
        player = Player(name="Luka", age=25)
        self.tool = PydanticAIInteroperability.convert_tool(tool=self.pydantic_ai_tool, deps=player)

    def test_convert_tool_raises_error_if_take_ctx_is_true_and_deps_is_none(self) -> None:
        with pytest.raises(ValueError, match="If the tool takes a context, the `deps` argument must be provided"):
            PydanticAIInteroperability.convert_tool(tool=self.pydantic_ai_tool, deps=None)

    def test_expected_tools(self) -> None:
        config_list = [{"api_type": "openai", "model": "gpt-4o", "api_key": "abc"}]
        chatbot = AssistantAgent(
            name="chatbot",
            llm_config={"config_list": config_list},
        )
        self.tool.register_for_llm(chatbot)

        expected_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_player",
                    "description": "Get the player's name.",
                    "parameters": {
                        "properties": {
                            "additional_info": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "description": "Additional information which can be used.",
                                "title": "Additional Info",
                            }
                        },
                        "type": "object",
                        "additionalProperties": False,
                    },
                },
            }
        ]

        assert chatbot.llm_config["tools"] == expected_tools  # type: ignore[index]

    @run_for_optional_imports("openai", "openai")
    def test_with_llm(self, credentials_gpt_4o: Credentials, user_proxy: UserProxyAgent) -> None:
        chatbot = AssistantAgent(
            name="chatbot",
            llm_config=credentials_gpt_4o.llm_config,
        )

        self.tool.register_for_execution(user_proxy)
        self.tool.register_for_llm(chatbot)

        user_proxy.initiate_chat(
            recipient=chatbot, message="Get player, for additional information use 'goal keeper'", max_turns=3
        )

        for message in user_proxy.chat_messages[chatbot]:
            if "tool_responses" in message:
                assert message["tool_responses"][0]["content"] == "Name: Luka, Age: 25, Additional info: goal keeper"
                return

        assert False, "No tool response found in chat messages"
