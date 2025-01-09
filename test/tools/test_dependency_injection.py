# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Annotated, Callable, get_type_hints

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
    _string_metadata_to_description_field,
    get_chat_context_params,
)


class TestRemoveInjectedParamsFromSignature:
    class MyContext(BaseContext, BaseModel):
        b: int

    def f_with_annotated(  # type: ignore[misc]
        a: int,
        ctx: Annotated[MyContext, Depends(MyContext(b=2))],
        chat_ctx: Annotated[ChatContext, "Chat context"],
    ) -> int:
        assert isinstance(chat_ctx, ChatContext)
        return a + ctx.b

    async def f_with_annotated_async(  # type: ignore[misc]
        a: int,
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
    def f_without_MyContext(
        a: int,
        ctx: Annotated[int, Depends(lambda a: a + 2)],
    ) -> int:
        return a + ctx

    @staticmethod
    def f_without_MyContext_async(
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
        chat_context_params = get_chat_context_params(self.f_all_params)
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


def test_string_metadata_to_description_field() -> None:
    def f(a: int, b: Annotated[int, "b description"]) -> int:
        return a + b

    type_hints = get_type_hints(f, include_extras=True)

    params_with_string_metadata = []
    for param, annotation in type_hints.items():
        if hasattr(annotation, "__metadata__"):
            metadata = annotation.__metadata__
            if metadata and isinstance(metadata[0], str):
                params_with_string_metadata.append(param)

    assert params_with_string_metadata == ["b"]

    f = _string_metadata_to_description_field(f)
    type_hints = get_type_hints(f, include_extras=True)
    for param, annotation in type_hints.items():
        if hasattr(annotation, "__metadata__"):
            metadata = annotation.__metadata__
            if metadata and isinstance(metadata[0], str):
                raise AssertionError("The string metadata should have been replaced with Pydantic's Field")

    field_info = type_hints["b"].__metadata__[0]
    assert isinstance(field_info, Field)
    assert field_info.description == "b description"
