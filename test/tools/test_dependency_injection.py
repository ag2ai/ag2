# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import sys
from typing import Annotated, Any, Callable, Optional, get_type_hints

import pytest
from pydantic import BaseModel

from autogen.tools.dependency_injection import (
    BaseContext,
    ChatContext,
    Depends,
    Field,
    _is_context_param,
    _is_depends_param,
    _remove_injected_params_from_signature,
    _set_return_annotation_to_any,
    _string_metadata_to_description_field,
    get_context_params,
)


class TestRemoveInjectedParamsFromSignature:
    class MyContext(BaseContext, BaseModel):
        b: int

    def f_with_annotated(  # type: ignore[misc]
        a: int,  # noqa: N805
        ctx: Annotated[MyContext, Depends(MyContext(b=2))],
        chat_ctx: Annotated[ChatContext, "Chat context"],
    ) -> int:
        assert isinstance(chat_ctx, ChatContext)
        return a + ctx.b

    async def f_with_annotated_async(  # type: ignore[misc]
        a: int,  # noqa: N805
        ctx: Annotated[MyContext, Depends(MyContext(b=2))],
        chat_ctx: Annotated[ChatContext, "Chat context"],
    ) -> int:
        assert isinstance(chat_ctx, ChatContext)
        return a + ctx.b

    @staticmethod
    def f_without_annotated(
        a: int,
        chat_ctx: ChatContext,
        ctx: MyContext = Depends(MyContext(b=3)),
    ) -> int:
        assert isinstance(chat_ctx, ChatContext)
        return a + ctx.b

    @staticmethod
    async def f_without_annotated_async(
        a: int,
        chat_ctx: ChatContext,
        ctx: MyContext = Depends(MyContext(b=3)),
    ) -> int:
        assert isinstance(chat_ctx, ChatContext)
        return a + ctx.b

    @staticmethod
    def f_without_annotated_and_depends(
        a: int,
        ctx: MyContext = MyContext(b=4),
    ) -> int:
        return a + ctx.b

    @staticmethod
    async def f_without_annotated_and_depends_async(
        a: int,
        ctx: MyContext = MyContext(b=4),
    ) -> int:
        return a + ctx.b

    @staticmethod
    def f_without_MyContext(  # noqa: N802
        a: int,
        ctx: Annotated[int, Depends(lambda a: a + 2)],
    ) -> int:
        return a + ctx

    @staticmethod
    def f_without_MyContext_async(  # noqa: N802
        a: int,
        ctx: Annotated[int, Depends(lambda a: a + 2)],
    ) -> int:
        return a + ctx

    @staticmethod
    def f_with_default_depends(
        a: int,
        ctx: int = Depends(lambda a: a + 2),
    ) -> int:
        return a + ctx

    @staticmethod
    def f_with_default_depends_async(
        a: int,
        ctx: int = Depends(lambda a: a + 2),
    ) -> int:
        return a + ctx

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.expected_tools = [
            {
                "type": "function",
                "function": {
                    "description": "Example function",
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "integer", "description": "a"}},
                        "required": ["a"],
                    },
                },
            }
        ]

    @staticmethod
    def f_all_params(
        a: int,
        b: Annotated[int, "b description"],
        ctx1: Annotated[MyContext, Depends(MyContext(b=2))],
        ctx2: Annotated[int, Depends(lambda a: a + 2)],
        ctx6: ChatContext,
        ctx7: Annotated[ChatContext, "ctx7 description"],
        ctx3: MyContext = Depends(MyContext(b=3)),
        ctx4: MyContext = MyContext(b=4),
        ctx5: int = Depends(lambda a: a + 2),
    ) -> int:
        return a

    def test_is_base_context_param(self) -> None:
        sig = inspect.signature(self.f_all_params)
        assert _is_context_param(sig.parameters["a"]) is False
        assert _is_context_param(sig.parameters["b"]) is False
        assert _is_context_param(sig.parameters["ctx1"]) is True
        assert _is_context_param(sig.parameters["ctx2"]) is False
        assert _is_context_param(sig.parameters["ctx3"]) is True
        assert _is_context_param(sig.parameters["ctx4"]) is True
        assert _is_context_param(sig.parameters["ctx5"]) is False
        assert _is_context_param(sig.parameters["ctx6"]) is True
        assert _is_context_param(sig.parameters["ctx7"]) is True

    def test_is_chat_context_param(self) -> None:
        sig = inspect.signature(self.f_all_params)
        assert _is_context_param(sig.parameters["ctx1"], subclass=ChatContext) is False
        assert _is_context_param(sig.parameters["ctx3"], subclass=ChatContext) is False
        assert _is_context_param(sig.parameters["ctx4"], subclass=ChatContext) is False
        assert _is_context_param(sig.parameters["ctx6"], subclass=ChatContext) is True
        assert _is_context_param(sig.parameters["ctx7"], subclass=ChatContext) is True

    def test_get_chat_context_params(self) -> None:
        chat_context_params = get_context_params(self.f_all_params, subclass=ChatContext)
        assert chat_context_params == ["ctx6", "ctx7"]

    def test_is_depends_param(self) -> None:
        sig = inspect.signature(self.f_all_params)
        assert _is_depends_param(sig.parameters["a"]) is False
        assert _is_depends_param(sig.parameters["b"]) is False
        # Whenever a parameter is annotated with Depends, it is considered also as Depends parameter
        assert _is_depends_param(sig.parameters["ctx1"]) is True
        assert _is_depends_param(sig.parameters["ctx2"]) is True
        assert _is_depends_param(sig.parameters["ctx3"]) is True
        assert _is_depends_param(sig.parameters["ctx5"]) is True

        assert _is_depends_param(sig.parameters["ctx4"]) is False

    @pytest.mark.parametrize(
        "test_func",
        [
            f_with_annotated,
            f_without_annotated,
            f_without_annotated_and_depends,
            f_with_annotated_async,
            f_without_annotated_async,
            f_without_annotated_and_depends_async,
            f_without_MyContext,
            f_without_MyContext_async,
            f_with_default_depends,
            f_with_default_depends_async,
        ],
    )
    def test_remove_injected_params_from_signature(self, test_func: Callable[..., int]) -> None:
        f = _remove_injected_params_from_signature(test_func)
        assert str(inspect.signature(f)) == "(a: int) -> int"


class TestHelperFunctions:
    def f_sync(  # type: ignore[misc]
        a: int,  # noqa: N805
        b: Annotated[int, "b description"],
        c: Annotated[Optional[int], "c description"] = None,
    ) -> int:
        return a + b

    def f_async(  # type: ignore[misc]
        a: int,  # noqa: N805
        b: Annotated[int, "b description"],
        c: Annotated[Optional[int], "c description"] = None,
    ) -> int:
        return a + b

    @pytest.mark.parametrize(
        "test_func",
        [
            f_sync,
            f_async,
        ],
    )
    def test_string_metadata_to_description_field(self, test_func: Callable[..., int]) -> None:
        f = _string_metadata_to_description_field(test_func)
        type_hints = get_type_hints(f, include_extras=True)

        field_info = type_hints["b"].__metadata__[0]
        assert isinstance(field_info, Field)
        assert field_info.description == "b description"

        if sys.version_info < (3, 11):
            field_info = type_hints["c"].__args__[0].__metadata__[0]
        else:
            field_info = type_hints["c"].__metadata__[0]

        assert isinstance(field_info, Field)
        assert field_info.description == "c description"

    @pytest.mark.parametrize(
        "test_func",
        [
            f_sync,
            f_async,
        ],
    )
    def test_set_return_annotation_to_any(self, test_func: Callable[..., int]) -> None:
        f = _set_return_annotation_to_any(test_func)
        assert inspect.signature(f).return_annotation == Any
