import json
from typing import Annotated
from unittest.mock import MagicMock

import pytest
from fast_depends import Depends
from pydantic import BaseModel

from autogen.beta import Context, tool


@pytest.mark.asyncio
async def test_execute(mock: MagicMock) -> None:
    @tool
    def my_func(a: str, b: int) -> str:
        mock(a=a, b=b)
        return "tool executed"

    assert (
        await my_func.execute(
            json.dumps({"a": "1", "b": "1"}),
            ctx=Context(mock),
        )
        == b'"tool executed"'
    )

    mock.assert_called_once_with(a="1", b=1)


@pytest.mark.asyncio
async def test_execute_sync_without_thread(mock: MagicMock) -> None:
    @tool(sync_to_thread=False)
    def my_func(a: str, b: int) -> str:
        mock(a=a, b=b)
        return "tool executed"

    assert (
        await my_func.execute(
            json.dumps({"a": "1", "b": "1"}),
            ctx=Context(mock),
        )
        == b'"tool executed"'
    )

    mock.assert_called_once_with(a="1", b=1)


@pytest.mark.asyncio
async def test_execute_async(mock: MagicMock) -> None:
    @tool
    async def my_func(a: str, b: int) -> str:
        mock(a=a, b=b)
        return "tool executed"

    assert (
        await my_func.execute(
            json.dumps({"a": "1", "b": "1"}),
            ctx=Context(mock),
        )
        == b'"tool executed"'
    )

    mock.assert_called_once_with(a="1", b=1)


@pytest.mark.asyncio
async def test_return_model(mock: MagicMock) -> None:
    class Result(BaseModel):
        a: str

    @tool
    def my_func(a: str, b: int) -> Result:
        return Result(a=a)

    assert (
        await my_func.execute(
            json.dumps({"a": "1", "b": "1"}),
            ctx=Context(mock),
        )
        == b'{"a":"1"}'
    )


@pytest.mark.asyncio
async def test_tool_with_depends(mock: MagicMock) -> None:
    def dep(a: str) -> str:
        return a * 2

    @tool
    def my_func(a: str, b: Annotated[str, Depends(dep)]) -> str:
        return a + b

    assert (
        await my_func.execute(
            json.dumps({"a": "1"}),
            ctx=Context(mock),
        )
        == b'"111"'
    )


@pytest.mark.asyncio
async def test_tool_get_context(mock: MagicMock) -> None:
    @tool
    def my_func(a: str, ctx: Context) -> str:
        return "".join(ctx.prompt)

    assert (
        await my_func.execute(
            json.dumps({"a": "1"}),
            ctx=Context(mock, prompt=["1"]),
        )
        == b'"1"'
    )
