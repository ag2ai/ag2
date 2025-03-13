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


@pytest.mark.anyio
async def test_not_annotated():
    @inject
    async def some_func(a, b):
        return a + b

    assert isinstance(await some_func("1", "2"), str)


@pytest.mark.anyio
async def test_annotated_partial():
    @inject
    async def some_func(a, b: int):
        assert isinstance(b, int)
        return a + b

    assert isinstance(await some_func(1, "2"), int)


@pytest.mark.anyio
async def test_arbitrary_args():
    class ArbitraryType:
        def __init__(self):
            self.value = "value"

    @inject
    async def some_func(a: ArbitraryType):
        return a

    assert isinstance(await some_func(ArbitraryType()), ArbitraryType)


@pytest.mark.anyio
async def test_arbitrary_response():
    class ArbitraryType:
        def __init__(self):
            self.value = "value"

    @inject
    async def some_func(a: ArbitraryType) -> ArbitraryType:
        return a

    assert isinstance(await some_func(ArbitraryType()), ArbitraryType)


@pytest.mark.anyio
async def test_types_casting():
    @inject
    async def some_func(a: int, b: int) -> float:
        assert isinstance(a, int)
        assert isinstance(b, int)
        r = a + b
        assert isinstance(r, int)
        return r

    assert isinstance(await some_func("1", "2"), float)


@pytest.mark.anyio
async def test_types_casting_from_str():
    @inject
    async def some_func(a: "int") -> float:
        return a

    assert isinstance(await some_func("1"), float)


@pytest.mark.anyio
async def test_pydantic_types_casting():
    class SomeModel(BaseModel):
        field: int

    @inject
    async def some_func(a: SomeModel):
        return a.field

    assert isinstance(await some_func({"field": "31"}), int)


@pytest.mark.anyio
async def test_pydantic_field_types_casting():
    @inject
    async def some_func(a: int = Field(..., alias="b")) -> float:
        assert isinstance(a, int)
        return a

    @inject
    async def another_func(a=Field(..., alias="b")) -> float:
        assert isinstance(a, str)
        return a

    assert isinstance(await some_func(b="2", c=3), float)
    assert isinstance(await another_func(b="2"), float)


@pytest.mark.anyio
async def test_wrong_incoming_types():
    @inject
    async def some_func(a: int):  # pragma: no cover
        return a

    with pytest.raises(ValidationError):
        await some_func({"key", 1})


@pytest.mark.anyio
async def test_wrong_return_types():
    @inject
    async def some_func(a: int) -> dict:
        return a

    with pytest.raises(ValidationError):
        await some_func("2")


@pytest.mark.anyio
async def test_annotated():
    A = Annotated[int, Field(..., alias="b")]  # noqa: N806

    @inject
    async def some_func(a: A) -> float:
        assert isinstance(a, int)
        return a

    assert isinstance(await some_func(b="2"), float)


@pytest.mark.anyio
async def test_args_kwargs_1():
    @inject
    async def simple_func(
        a: int,
        *args: tuple[float, ...],
        b: int,
        **kwargs: dict[str, int],
    ):
        return a, args, b, kwargs

    assert await simple_func(1.0, 2.0, 3, b=3.0, key=1.0) == (1, (2.0, 3.0), 3, {"key": 1})


@pytest.mark.anyio
async def test_args_kwargs_2():
    @inject
    async def simple_func(
        a: int,
        *args: tuple[float, ...],
        b: int,
    ):
        return a, args, b

    assert await simple_func(
        1.0,
        2.0,
        3,
        b=3.0,
    ) == (1, (2.0, 3.0), 3)


@pytest.mark.anyio
async def test_args_kwargs_3():
    @inject
    async def simple_func(a: int, *, b: int):
        return a, b

    assert await simple_func(
        1.0,
        b=3.0,
    ) == (1, 3)


@pytest.mark.anyio
async def test_generator():
    @inject
    async def simple_func(a: str) -> int:
        for _ in range(2):
            yield a

    async for i in simple_func("1"):
        assert i == 1


@pytest.mark.anyio
async def test_generator_iterator_type():
    @inject
    async def simple_func(a: str) -> Iterator[int]:
        for _ in range(2):
            yield a

    async for i in simple_func("1"):
        assert i == 1


@pytest.mark.anyio
@pydanticV2
async def test_multi_annotated():
    from pydantic.functional_validators import AfterValidator

    @inject()
    async def f(a: Annotated[int, Ge(10), AfterValidator(lambda x: x + 10)]) -> int:
        return a

    with pytest.raises(ValidationError):
        await f(1)

    assert await f(10) == 20
