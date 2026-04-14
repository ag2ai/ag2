# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from autogen.beta import Agent, Context
from autogen.beta.events import ToolCallEvent
from autogen.beta.testing import TestConfig, TrackingConfig
from autogen.beta.tools import FilesystemToolset


@pytest.mark.asyncio
async def test_path_traversal_blocked(tmp_path: Path) -> None:
    from autogen.beta.tools.toolkits.filesystem import _resolve_path

    with pytest.raises(PermissionError, match="escapes base directory"):
        _resolve_path(tmp_path, "../../etc/passwd")


@pytest.mark.asyncio
async def test_schemas(async_mock: AsyncMock) -> None:
    toolset = FilesystemToolset()
    schemas = list(await toolset.schemas(Context(async_mock)))

    names = {s.function.name for s in schemas}
    assert names == {"read_file", "write_file", "update_file", "delete_file", "find_files"}


@pytest.mark.asyncio
async def test_read_only(async_mock: AsyncMock) -> None:
    toolset = FilesystemToolset(read_only=True)
    schemas = list(await toolset.schemas(Context(async_mock)))

    names = {s.function.name for s in schemas}
    assert names == {"read_file", "find_files"}


@pytest.mark.asyncio
async def test_read_file(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello world")

    toolset = FilesystemToolset(base_path=tmp_path)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(
                name="read_file",
                arguments=json.dumps({"path": "./hello.txt"}),
            ),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[toolset])
    await agent.ask("read it")

    # Second call receives the tool result; verify the file content was read
    tool_result_msg = tracking.mock.call_args_list[1][0][0]
    assert "hello world" in tool_result_msg.results[0].content


@pytest.mark.asyncio
async def test_read_file_raw(tmp_path: Path) -> None:
    binary_content = bytes(range(256))
    (tmp_path / "binary.bin").write_bytes(binary_content)

    toolset = FilesystemToolset(base_path=tmp_path)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(
                name="read_file",
                arguments=json.dumps({"path": "binary.bin", "raw": True}),
            ),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[toolset])
    await agent.ask("read binary")

    tool_result_msg = tracking.mock.call_args_list[1][0][0]
    result = tool_result_msg.results[0].content
    assert base64.b64decode(result) == binary_content


@pytest.mark.asyncio
async def test_write_file(tmp_path: Path) -> None:
    toolset = FilesystemToolset(base_path=tmp_path)

    config = TestConfig(
        ToolCallEvent(
            name="write_file",
            arguments=json.dumps({"path": "out.txt", "content": "new content"}),
        ),
        "done",
    )
    agent = Agent("", config=config, tools=[toolset])
    await agent.ask("write it")

    assert (tmp_path / "out.txt").read_text() == "new content"


@pytest.mark.asyncio
async def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    toolset = FilesystemToolset(base_path=tmp_path)

    config = TestConfig(
        ToolCallEvent(name="write_file", arguments=json.dumps({"path": "sub/dir/file.txt", "content": "nested"})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolset])
    await agent.ask("write nested")

    assert (tmp_path / "sub" / "dir" / "file.txt").read_text() == "nested"


@pytest.mark.asyncio
async def test_update_file(tmp_path: Path) -> None:
    (tmp_path / "data.txt").write_text("foo bar baz")

    toolset = FilesystemToolset(base_path=tmp_path)

    config = TestConfig(
        ToolCallEvent(
            name="update_file",
            arguments=json.dumps({"path": "data.txt", "old_content": "bar", "new_content": "qux"}),
        ),
        "done",
    )
    agent = Agent("", config=config, tools=[toolset])
    await agent.ask("update it")

    assert (tmp_path / "data.txt").read_text() == "foo qux baz"


@pytest.mark.asyncio
async def test_delete_file(tmp_path: Path) -> None:
    target = tmp_path / "to_delete.txt"
    target.write_text("bye")

    toolset = FilesystemToolset(base_path=tmp_path)

    config = TestConfig(
        ToolCallEvent(name="delete_file", arguments=json.dumps({"path": "to_delete.txt"})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolset])
    await agent.ask("delete it")

    assert not target.exists()


@pytest.mark.asyncio
async def test_find_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.py").write_text("c")
    (sub / "d.txt").write_text("d")
    sub2 = sub / "sub2"
    sub2.mkdir()
    (sub2 / "e.py").write_text("e")

    # tmp_path
    # |-- a.py
    # |-- b.txt
    # |-- sub
    # |   |-- c.py
    # |   |-- d.txt
    # |   |-- sub2
    # |       |-- e.py

    toolset = FilesystemToolset(base_path=tmp_path)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(name="find_files", arguments=json.dumps({"pattern": "**/*.py"})),
            ToolCallEvent(name="find_files", arguments=json.dumps({"pattern": "sub/*"})),
            ToolCallEvent(name="find_files", arguments=json.dumps({"pattern": "sub/**"})),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[toolset])
    await agent.ask("find py files")

    # "**/*.py" — recursive, matches .py files at any depth
    result_1 = json.loads(tracking.mock.call_args_list[1][0][0].results[0].content)
    assert sorted(result_1) == ["a.py", str(Path("sub/c.py")), str(Path("sub/sub2/e.py"))]

    # "sub/*" — non-recursive, matches all files directly in sub/
    result_2 = json.loads(tracking.mock.call_args_list[2][0][0].results[0].content)
    assert sorted(result_2) == [str(Path("sub/c.py")), str(Path("sub/d.txt"))]

    # "sub/**" — recursive, matches all files under sub/ at any depth
    result_3 = json.loads(tracking.mock.call_args_list[3][0][0].results[0].content)
    assert sorted(result_3) == [str(Path("sub/c.py")), str(Path("sub/d.txt")), str(Path("sub/sub2/e.py"))]
