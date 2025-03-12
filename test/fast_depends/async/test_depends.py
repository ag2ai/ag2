# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/https://github.com/Lancetnik/FastDepends are under the MIT License.
# SPDX-License-Identifier: MIT

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from typing_extensions import Annotated

from autogen.fast_depends import Depends, inject


@pytest.mark.anyio
async def test_depends():
    async def dep_func(b: int, a: int = 3) -> float:
        return a + b

    @inject
    async def some_func(b: int, c=Depends(dep_func)) -> int:
        assert isinstance(c, float)
        return b + c

    assert (await some_func("2")) == 7


@pytest.mark.anyio
async def test_empty_main_body():
    async def dep_func(a: int) -> float:
        return a

    @inject
    async def some_func(c=Depends(dep_func)):
        assert isinstance(c, float)
        assert c == 1.0

    await some_func("1")


@pytest.mark.anyio
async def test_sync_depends():
    def sync_dep_func(a: int) -> float:
        return a

    @inject
    async def some_func(a: int, b: int, c=Depends(sync_dep_func)) -> float:
        assert isinstance(c, float)
        return a + b + c

    assert await some_func("1", "2")


@pytest.mark.anyio
async def test_depends_response_cast():
    async def dep_func(a):
        return a

    @inject
    async def some_func(a: int, b: int, c: int = Depends(dep_func)) -> float:
        assert isinstance(c, int)
        return a + b + c

    assert await some_func("1", "2")


@pytest.mark.anyio
async def test_depends_error():
    async def dep_func(b: dict, a: int = 3) -> float:  # pragma: no cover
        return a + b

    async def another_dep_func(b: int, a: int = 3) -> dict:  # pragma: no cover
        return a + b

    @inject
    async def some_func(b: int, c=Depends(dep_func), d=Depends(another_dep_func)) -> int:  # pragma: no cover
        assert c is None
        return b

    with pytest.raises(ValidationError):
        assert await some_func("2")


@pytest.mark.anyio
async def test_depends_annotated():
    async def dep_func(a):
        return a

    D = Annotated[int, Depends(dep_func)]  # noqa: N806

    @inject
    async def some_func(a: int, b: int, c: D = None) -> float:
        assert isinstance(c, int)
        return a + b + c

    @inject
    async def another_func(a: int, c: D):
        return a + c

    assert await some_func("1", "2")
    assert (await another_func(3)) == 6.0


@pytest.mark.anyio
@pytest.mark.skip("Currently failing.")
async def test_async_depends_annotated_str():
    async def dep_func(a):
        return a

    @inject
    async def some_func(
        a: int,
        b: int,
        c: "Annotated[int, Depends(dep_func)]",
    ) -> float:
        assert isinstance(c, int)
        return a + b + c

    @inject
    async def another_func(
        a: int,
        c: "Annotated[int, Depends(dep_func)]",
    ):
        return a + c

    assert await some_func("1", "2")
    assert await another_func("3") == 6.0


@pytest.mark.anyio
async def test_async_depends_annotated_str_partial():
    async def adep_func(a):
        return a

    @inject
    async def some_func(
        a: int,
        b: int,
        c: Annotated["float", Depends(adep_func)],
    ) -> float:
        assert isinstance(c, float)
        return a + b + c

    @inject
    async def another_func(
        a: int,
        c: Annotated["float", Depends(adep_func)],
    ):
        return a + c

    assert await some_func("1", "2")
    assert await another_func("3") == 6.0


@pytest.mark.anyio
async def test_cache():
    mock = Mock()

    async def nested_dep_func():
        mock()
        return 1000

    async def dep_func(a=Depends(nested_dep_func)):
        return a

    @inject
    async def some_func(a=Depends(dep_func), b=Depends(nested_dep_func)):
        assert a is b
        return a + b

    assert await some_func()
    mock.assert_called_once()


@pytest.mark.anyio
async def test_not_cache():
    mock = Mock()

    async def nested_dep_func():
        mock()
        return 1000

    async def dep_func(a=Depends(nested_dep_func, use_cache=False)):
        return a

    @inject
    async def some_func(
        a=Depends(dep_func, use_cache=False),
        b=Depends(nested_dep_func, use_cache=False),
    ):
        assert a is b
        return a + b

    assert await some_func()
    assert mock.call_count == 2


@pytest.mark.anyio
async def test_yield():
    mock = Mock()

    async def dep_func():
        mock()
        yield 1000
        mock.exit()

    @inject
    async def some_func(a=Depends(dep_func)):
        assert mock.called
        assert not mock.exit.called
        return a

    assert await some_func()
    mock.assert_called_once()
    mock.exit.assert_called_once()


@pytest.mark.anyio
async def test_sync_yield():
    mock = Mock()

    def sync_dep_func():
        mock()
        yield 1000
        mock.exit()

    @inject
    async def some_func(a=Depends(sync_dep_func)):
        assert mock.called
        assert not mock.exit.called
        return a

    assert await some_func()
    mock.assert_called_once()
    mock.exit.assert_called_once()


@pytest.mark.anyio
async def test_sync_yield_exception():
    mock = Mock()

    def sync_dep_func():
        mock()
        yield 1000
        raise ValueError()

    @inject
    async def some_func(a=Depends(sync_dep_func)):
        assert mock.called
        assert not mock.exit.called
        return a

    with pytest.raises(ValueError):
        await some_func()

    mock.assert_called_once()
    assert not mock.exit.called


@pytest.mark.anyio
async def test_sync_yield_exception_start():
    mock = Mock()

    def sync_dep_func():
        raise ValueError()

    @inject
    async def some_func(a=Depends(sync_dep_func)):  # pragma: no cover
        mock()
        return a

    with pytest.raises(ValueError):
        await some_func()

    assert not mock.called


@pytest.mark.anyio
async def test_sync_yield_exception_main():
    mock = Mock()

    def sync_dep_func():
        mock()
        try:
            yield 1000
        finally:
            mock.exit()

    @inject
    async def some_func(a=Depends(sync_dep_func)):
        assert mock.called
        assert not mock.exit.called
        raise ValueError()

    with pytest.raises(ValueError):
        await some_func()

    mock.assert_called_once()
    mock.exit.assert_called_once()


@pytest.mark.anyio
async def test_class_depends():
    class MyDep:
        def __init__(self, a: int):
            self.a = a

    @inject
    async def some_func(a=Depends(MyDep)):
        assert isinstance(a, MyDep)
        assert a.a == 3
        return a

    await some_func(3)


@pytest.mark.anyio
async def test_callable_class_depends():
    class MyDep:
        def __init__(self, a: int):
            self.a = a

        def __call__(self) -> int:
            return self.a

    @inject
    async def some_func(a=Depends(MyDep(3))):  # noqa: B008
        assert a == 3
        return a

    await some_func()


@pytest.mark.anyio
async def test_async_callable_class_depends():
    class MyDep:
        def __init__(self, a: int):
            self.a = a

        async def call(self) -> int:
            return self.a

    @inject
    async def some_func(a=Depends(MyDep(3).call)):  # noqa: B008
        assert a == 3
        return a

    await some_func()


@pytest.mark.anyio
async def test_not_cast():
    @dataclass
    class A:
        a: int

    async def dep() -> A:
        return A(a=1)

    async def get_logger() -> logging.Logger:
        return logging.getLogger(__file__)

    @inject
    async def some_func(
        b,
        a: A = Depends(dep, cast=False),
        logger: logging.Logger = Depends(get_logger, cast=False),
    ):
        assert a.a == 1
        assert logger
        return b

    assert (await some_func(1)) == 1


@pytest.mark.anyio
@pytest.mark.skip("Currently failing.")
async def test_not_cast_main():
    @dataclass
    class A:
        a: int

    async def dep() -> A:
        return A(a=1)

    async def get_logger() -> logging.Logger:
        return logging.getLogger(__file__)

    @inject(cast=False)
    async def some_func(
        b: str,
        a: A = Depends(dep),
        logger: logging.Logger = Depends(get_logger),
    ) -> str:
        assert a.a == 1
        assert logger
        return b

    assert (await some_func(1)) == 1


@pytest.mark.anyio
async def test_extra():
    mock = Mock()

    async def dep():
        mock.async_call()

    def sync_dep():
        mock.sync_call()

    @inject(extra_dependencies=(Depends(dep), Depends(sync_dep)))
    async def some_func():
        mock()

    await some_func(1)
    mock.assert_called_once()
    mock.async_call.assert_called_once()
    mock.sync_call.assert_called_once()


@pytest.mark.anyio
async def test_generator():
    mock = Mock()

    async def func():
        mock.async_call()

    async def gen_func():
        mock.start()
        yield
        mock.end()

    @inject
    async def simple_func(
        a: str,
        d=Depends(gen_func),
        d2=Depends(func),
    ) -> int:
        for _ in range(2):
            yield a

    async for i in simple_func("1"):
        mock.start.assert_called_once()
        assert not mock.end.called
        assert i == 1

    mock.async_call.assert_called_once()
    mock.end.assert_called_once()


@pytest.mark.anyio
async def test_partial():
    async def dep(a):
        return a

    @inject
    async def func(a=Depends(partial(dep, 10))):  # noqa D008
        return a

    assert await func() == 10


@pytest.mark.anyio
@pytest.mark.skip("Currently failing.")
async def test_default_key_value():
    async def dep(a: str = "a"):
        return a

    @inject(cast=False)
    async def func(a=Depends(dep)):
        return a

    assert await func() == "a"


@pytest.mark.anyio
async def test_asynccontextmanager():
    async def dep(a: str):
        return a

    @asynccontextmanager
    @inject
    async def func(a: str, b: str = Depends(dep)):
        yield a == b

    async with func("a") as is_equal:
        assert is_equal
