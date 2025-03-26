# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/https://github.com/Lancetnik/FastDepends are under the MIT License.
# SPDX-License-Identifier: MIT

from typing import Iterator

import pytest
from annotated_types import Ge
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import Annotated

from autogen.fast_depends import inject
from test.fast_depends.marks import pydanticV2


def test_not_annotated():
    @inject
    def some_func(a, b):
        return a + b

    assert isinstance(some_func("1", "2"), str)


def test_annotated_partial():
    @inject
    def some_func(a, b: int):
        assert isinstance(b, int)
        return a + b

    assert isinstance(some_func(1, "2"), int)


def test_arbitrary_args():
    class ArbitraryType:
        def __init__(self):
            self.value = "value"

    @inject
    def some_func(a: ArbitraryType):
        return a

    assert isinstance(some_func(ArbitraryType()), ArbitraryType)


def test_arbitrary_response():
    class ArbitraryType:
        def __init__(self):
            self.value = "value"

    @inject
    def some_func(a: ArbitraryType) -> ArbitraryType:
        return a

    assert isinstance(some_func(ArbitraryType()), ArbitraryType)


def test_validation_error():
    @inject
    def some_func(a, b: str = Field(..., max_length=1)):  # pragma: no cover
        pass

    with pytest.raises(ValidationError):
        assert some_func()

    with pytest.raises(ValidationError):
        assert some_func(1, "dsdas")


def test_types_casting():
    @inject
    def some_func(a: int, b: int) -> float:
        assert isinstance(a, int)
        assert isinstance(b, int)
        r = a + b
        assert isinstance(r, int)
        return r

    assert isinstance(some_func("1", "2"), float)


def test_types_casting_from_str():
    @inject
    def some_func(a: "int") -> float:
        return a

    assert isinstance(some_func("1"), float)


def test_pydantic_types_casting():
    class SomeModel(BaseModel):
        field: int

    @inject
    def some_func(a: SomeModel):
        return a.field

    assert isinstance(some_func({"field": "31"}), int)


def test_pydantic_field_types_casting():
    @inject
    def some_func(a: int = Field(..., alias="b")) -> float:
        assert isinstance(a, int)
        return a

    @inject
    def another_func(a=Field(..., alias="b")) -> float:
        assert isinstance(a, str)
        return a

    assert isinstance(some_func(b="2"), float)
    assert isinstance(another_func(b="2"), float)


def test_wrong_incoming_types():
    @inject
    def some_func(a: int):  # pragma: no cover
        return a

    with pytest.raises(ValidationError):
        some_func({"key", 1})


def test_wrong_return_types():
    @inject
    def some_func(a: int) -> dict:
        return a

    with pytest.raises(ValidationError):
        some_func("2")


def test_annotated():
    A = Annotated[int, Field(..., alias="b")]  # noqa: N806

    @inject
    def some_func(a: A) -> float:
        assert isinstance(a, int)
        return a

    assert isinstance(some_func(b="2"), float)


def test_args_kwargs_1():
    @inject
    def simple_func(
        a: int,
        *args: tuple[float, ...],
        b: int,
        **kwargs: dict[str, int],
    ):
        return a, args, b, kwargs

    assert simple_func(1.0, 2.0, 3, b=3.0, key=1.0) == (1, (2.0, 3.0), 3, {"key": 1})


def test_args_kwargs_2():
    @inject
    def simple_func(
        a: int,
        *args: tuple[float, ...],
        b: int,
    ):
        return a, args, b

    assert simple_func(
        1.0,
        2.0,
        3,
        b=3.0,
    ) == (1, (2.0, 3.0), 3)


def test_args_kwargs_3():
    @inject
    def simple_func(a: int, *, b: int):
        return a, b

    assert simple_func(
        1.0,
        b=3.0,
    ) == (1, 3)


def test_generator():
    @inject
    def simple_func(a: str) -> int:
        for _ in range(2):
            yield a

    for i in simple_func("1"):
        assert i == 1


def test_generator_iterator_type():
    @inject
    def simple_func(a: str) -> Iterator[int]:
        for _ in range(2):
            yield a

    for i in simple_func("1"):
        assert i == 1


@pydanticV2
def test_multi_annotated():
    from pydantic.functional_validators import AfterValidator

    @inject()
    def f(a: Annotated[int, Ge(10), AfterValidator(lambda x: x + 10)]) -> int:
        return a

    with pytest.raises(ValidationError):
        f(1)

    assert f(10) == 20
