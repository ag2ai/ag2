# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Annotated, Callable

import pytest
from pydantic import BaseModel

from autogen.tools.dependency_injection import BaseContext, Depends, remove_injected_params_from_signature


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
    def f_without_annotated(
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

    @pytest.mark.parametrize("test_func", [f_with_annotated, f_without_annotated, f_without_annotated_and_depends])
    def test_remove_injected_params_from_signature(self, test_func: Callable[..., int]) -> None:
        remove_injected_params_from_signature(test_func)
        assert str(inspect.signature(test_func)) == "(a: int) -> int"
