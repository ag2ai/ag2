# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Annotated, Callable, get_type_hints

import pytest
from pydantic import BaseModel

from autogen.tools.dependency_injection import (
    BaseContext,
    Depends,
    Field,
    _remove_injected_params_from_signature,
    _string_metadata_to_description_field,
)


class TestRemoveInjectedParamsFromSignature:
    class MyContext(BaseContext, BaseModel):
        b: int

    @staticmethod
    def f_with_annotated(
        a: int,
        ctx: Annotated[MyContext, Depends(MyContext(b=2))],
    ) -> int:
        return a + ctx.b

    @staticmethod
    async def f_with_annotated_async(
        a: int,
        ctx: Annotated[MyContext, Depends(MyContext(b=2))],
    ) -> int:
        return a + ctx.b

    @staticmethod
    def f_without_annotated(
        a: int,
        ctx: MyContext = Depends(MyContext(b=3)),
    ) -> int:
        return a + ctx.b

    @staticmethod
    async def f_without_annotated_async(
        a: int,
        ctx: MyContext = Depends(MyContext(b=3)),
    ) -> int:
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
        ],
    )
    def test_remove_injected_params_from_signature(self, test_func: Callable[..., int]) -> None:
        _remove_injected_params_from_signature(test_func)
        assert str(inspect.signature(test_func)) == "(a: int) -> int"


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
