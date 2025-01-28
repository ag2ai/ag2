# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any, Callable, Optional
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from autogen.agentchat import ConversableAgent, UserProxyAgent
from autogen.tools import BaseContext, ChatContext, Depends

from ..conftest import Credentials, credentials_all_llms, suppress_gemini_resource_exhausted


class MyContext(BaseContext, BaseModel):
    b: int


def f_with_annotated(
    a: int,
    ctx: Annotated[MyContext, Depends(MyContext(b=2))],
    chat_ctx: Annotated[ChatContext, "Chat context"],
    c: Annotated[int, "c description"] = 3,
    d: Annotated[Optional[int], "d description"] = None,
) -> int:
    assert isinstance(chat_ctx, ChatContext)
    return a + ctx.b + c


async def f_with_annotated_async(
    a: int,
    ctx: Annotated[MyContext, Depends(MyContext(b=2))],
    chat_ctx: ChatContext,
    c: Annotated[int, "c description"] = 3,
    d: Annotated[Optional[int], "d description"] = None,
) -> int:
    assert isinstance(chat_ctx, ChatContext)
    return a + ctx.b + c


def f_without_annotated(
    a: int,
    chat_ctx: ChatContext,
    ctx: MyContext = Depends(MyContext(b=3)),
    c: Annotated[int, "c description"] = 3,
    d: Annotated[Optional[int], "d description"] = None,
) -> int:
    return a + ctx.b + c


async def f_without_annotated_async(
    a: int,
    ctx: MyContext = Depends(MyContext(b=3)),
    c: Annotated[int, "c description"] = 3,
    d: Annotated[Optional[int], "d description"] = None,
) -> int:
    return a + ctx.b + c


def f_with_annotated_and_depends(
    a: int,
    ctx: MyContext = MyContext(b=4),
    c: Annotated[int, "c description"] = 3,
    d: Annotated[Optional[int], "d description"] = None,
) -> int:
    return a + ctx.b + c


async def f_with_annotated_and_depends_async(
    a: int,
    ctx: MyContext = MyContext(b=4),
    c: Annotated[int, "c description"] = 3,
    d: Annotated[Optional[int], "d description"] = None,
) -> int:
    return a + ctx.b + c


def f_with_multiple_depends(
    a: int,
    ctx: Annotated[MyContext, Depends(MyContext(b=2))],
    ctx2: Annotated[MyContext, Depends(MyContext(b=3))],
    c: Annotated[int, "c description"] = 3,
    d: Annotated[Optional[int], "d description"] = None,
) -> int:
    return a + ctx.b + ctx2.b + c


async def f_with_multiple_depends_async(
    a: int,
    ctx: Annotated[MyContext, Depends(MyContext(b=2))],
    ctx2: Annotated[MyContext, Depends(MyContext(b=3))],
    c: Annotated[int, "c description"] = 3,
    d: Annotated[Optional[int], "d description"] = None,
) -> int:
    return a + ctx.b + ctx2.b + c


def f_wihout_base_context(
    a: int,
    ctx: Annotated[int, Depends(lambda a: a + 2)],
    c: Annotated[int, "c description"] = 3,
    d: Annotated[Optional[int], "d description"] = None,
) -> int:
    return a + ctx + c


async def f_wihout_base_context_async(
    a: int,
    ctx: Annotated[int, Depends(lambda a: a + 2)],
    c: Annotated[int, "c description"] = 3,
    d: Annotated[Optional[int], "d description"] = None,
) -> int:
    return a + ctx + c


def f_with_default_depends(
    a: int,
    ctx: int = Depends(lambda a: a + 2),
    c: Annotated[int, "c description"] = 3,
    d: Annotated[Optional[int], "d description"] = None,
) -> int:
    return a + ctx + c


async def f_with_default_depends_async(
    a: int,
    ctx: int = Depends(lambda a: a + 2),
    c: Annotated[int, "c description"] = 3,
    d: Annotated[Optional[int], "d description"] = None,
) -> int:
    return a + ctx + c


class TestDependencyInjection:
    @pytest.fixture
    def expected_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "description": "Example function",
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer", "description": "a"},
                            "c": {"type": "integer", "description": "c description", "default": 3},
                            "d": {
                                "anyOf": [{"type": "integer"}, {"type": "null"}],
                                "description": "d description",
                                "default": None,
                            },
                        },
                        "required": ["a"],
                    },
                },
            }
        ]

    @pytest.mark.parametrize(
        ("func", "func_name", "is_async", "expected"),
        [
            (f_with_annotated, "f_with_annotated", False, "6"),
            (f_with_annotated_async, "f_with_annotated_async", True, "6"),
            (f_without_annotated, "f_without_annotated", False, "7"),
            (f_without_annotated_async, "f_without_annotated_async", True, "7"),
            (f_with_annotated_and_depends, "f_with_annotated_and_depends", False, "8"),
            (f_with_annotated_and_depends_async, "f_with_annotated_and_depends_async", True, "8"),
            (f_with_multiple_depends, "f_with_multiple_depends", False, "9"),
            (f_with_multiple_depends_async, "f_with_multiple_depends_async", True, "9"),
            (f_wihout_base_context, "f_wihout_base_context", False, "7"),
            (f_wihout_base_context_async, "f_wihout_base_context_async", True, "7"),
            (f_with_default_depends, "f_with_default_depends", False, "7"),
            (f_with_default_depends_async, "f_with_default_depends_async", True, "7"),
        ],
    )
    @pytest.mark.asyncio
    async def test_register_tools(
        self,
        mock_credentials: Credentials,
        expected_tools: list[dict[str, Any]],
        func: Callable[..., Any],
        func_name: str,
        is_async: bool,
        expected: str,
    ) -> None:
        agent = ConversableAgent(name="agent", llm_config=mock_credentials.llm_config)
        agent.register_for_llm(description="Example function")(func)
        agent.register_for_execution()(func)

        expected_tools[0]["function"]["name"] = func_name
        assert agent.llm_config["tools"] == expected_tools

        assert func_name in agent.function_map

        retval = agent.function_map[func_name](1)
        actual = await retval if is_async else retval

        assert actual == expected

    async def _test_end2end(self, credentials: Credentials, is_async: bool) -> None:
        class UserContext(BaseContext, BaseModel):
            username: str
            password: str

        agent = ConversableAgent(name="agent", llm_config=credentials.llm_config)
        user = UserContext(username="user23", password="password23")
        users = [user]

        user_proxy = UserProxyAgent(
            name="user_proxy_1",
            human_input_mode="NEVER",
        )

        mock = MagicMock()

        def _login(user: UserContext) -> str:
            if user in users:
                mock(user, "Login successful.")
                return "Login successful."
            mock(user, "Login failed.")
            return "Login failed"

        if is_async:

            @user_proxy.register_for_execution()
            @agent.register_for_llm(description="Login function")
            async def login(
                user: Annotated[UserContext, Depends(user)],
                chat_ctx: ChatContext,
            ) -> str:
                expected = {"arguments": "{}", "name": "login"}
                assert chat_ctx.last_message["tool_calls"][0]["function"] == expected

                return _login(user)

            await user_proxy.a_initiate_chat(agent, message="Please login", max_turns=2)
        else:

            @user_proxy.register_for_execution()
            @agent.register_for_llm(description="Login function")
            def login(
                user: Annotated[UserContext, Depends(user)],
            ) -> str:
                return _login(user)

            user_proxy.initiate_chat(agent, message="Please login", max_turns=2)

        mock.assert_called_once_with(
            UserContext(username="user23", password="password23"),
            "Login successful.",
        )

    @pytest.mark.parametrize("credentials_from_test_param", credentials_all_llms, indirect=True)
    @suppress_gemini_resource_exhausted
    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.asyncio
    async def test_end2end(
        self,
        credentials_from_test_param: Credentials,
        is_async: bool,
    ) -> None:
        await self._test_end2end(credentials_from_test_param, is_async)
