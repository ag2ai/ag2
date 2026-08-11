# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ag2 import Agent, Context
from ag2.events import ToolCallEvent, ToolResultsEvent
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools import FilesystemToolkit
from ag2.tools.toolkits.filesystem import _resolve_path


def test_path_traversal_blocked(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="escapes base directory"):
        _resolve_path(tmp_path, "../../etc/passwd")


def test_absolute_path_blocked(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="escapes base directory"):
        _resolve_path(tmp_path, "/etc/passwd")


def test_allow_only_dir() -> None:
    with pytest.raises(ValueError, match="is not a directory"):
        FilesystemToolkit("file.txt")


@pytest.mark.asyncio
async def test_schemas(async_mock: AsyncMock) -> None:
    toolkit = FilesystemToolkit()
    schemas = list(await toolkit.schemas(Context(async_mock)))

    names = {s.function.name for s in schemas}
    assert names == {
        "read_file",
        "write_file",
        "update_file",
        "delete_file",
        "find_files",
    }


@pytest.mark.asyncio
async def test_read_only(async_mock: AsyncMock) -> None:
    toolkit = FilesystemToolkit(read_only=True)
    schemas = list(await toolkit.schemas(Context(async_mock)))

    names = {s.function.name for s in schemas}
    assert names == {"read_file", "find_files"}


@pytest.mark.asyncio
async def test_read_file(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello world")

    toolkit = FilesystemToolkit(base_path=tmp_path)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(
                name="read_file",
                arguments=json.dumps({"path": "./hello.txt"}),
            ),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[toolkit])
    await agent.ask("read it")

    # Second call receives the tool result; verify the file content was read
    tool_result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    assert "hello world" in tool_result_msg.results[0].result.parts[0].content


@pytest.mark.asyncio
async def test_read_file_raw(tmp_path: Path) -> None:
    binary_content = bytes(range(256))
    (tmp_path / "binary.bin").write_bytes(binary_content)

    toolkit = FilesystemToolkit(base_path=tmp_path)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(
                name="read_file",
                arguments=json.dumps({"path": "binary.bin", "raw": True}),
            ),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[toolkit])
    await agent.ask("read binary")

    tool_result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    assert base64.b64decode(tool_result_msg.results[0].result.parts[0].content) == binary_content


@pytest.mark.asyncio
async def test_write_file(tmp_path: Path) -> None:
    toolkit = FilesystemToolkit(base_path=tmp_path)

    config = TestConfig(
        ToolCallEvent(
            name="write_file",
            arguments=json.dumps({"path": "out.txt", "content": "new content"}),
        ),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("write it")

    assert (tmp_path / "out.txt").read_text() == "new content"


@pytest.mark.asyncio
async def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    toolkit = FilesystemToolkit(base_path=tmp_path)

    config = TestConfig(
        ToolCallEvent(
            name="write_file",
            arguments=json.dumps({"path": "sub/dir/file.txt", "content": "nested"}),
        ),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("write nested")

    assert (tmp_path / "sub" / "dir" / "file.txt").read_text() == "nested"


@pytest.mark.asyncio
async def test_write_file_encodes_as_utf8(tmp_path: Path) -> None:
    """The toolkit must write non-ASCII content as UTF-8 regardless of the
    host's `locale.getpreferredencoding()` (which is `cp1252` on most
    Windows installs and would silently raise `UnicodeEncodeError` mid-write
    for any non-cp1252 glyph). Read the bytes back and compare against the
    UTF-8 encoding directly so the test pins the contract on every platform.
    """
    payload = "Café — Beberenice ☕ — 例 — émoji 🚀"
    toolkit = FilesystemToolkit(base_path=tmp_path)

    config = TestConfig(
        ToolCallEvent(
            name="write_file",
            arguments=json.dumps({"path": "non_ascii.txt", "content": payload}),
        ),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("write it")

    assert (tmp_path / "non_ascii.txt").read_bytes() == payload.encode("utf-8")


@pytest.mark.asyncio
async def test_read_file_decodes_as_utf8(tmp_path: Path) -> None:
    """The toolkit must read files written as UTF-8 regardless of the
    host's default encoding. On Windows, `Path.read_text()` without an
    explicit encoding decodes as `cp1252` and raises `UnicodeDecodeError`
    for any non-cp1252 byte sequence.
    """
    payload = "Café — Beberenice ☕ — 例 — émoji 🚀"
    (tmp_path / "non_ascii.txt").write_bytes(payload.encode("utf-8"))

    toolkit = FilesystemToolkit(base_path=tmp_path)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(
                name="read_file",
                arguments=json.dumps({"path": "./non_ascii.txt"}),
            ),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[toolkit])
    await agent.ask("read it")

    tool_result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    assert tool_result_msg.results[0].result.parts[0].content == payload


@pytest.mark.asyncio
async def test_update_file_preserves_utf8(tmp_path: Path) -> None:
    """`update_file` reads, edits, and rewrites the file. Both endpoints
    must use UTF-8 so a round-trip preserves non-ASCII byte content.
    """
    target = tmp_path / "data.txt"
    target.write_bytes("Café old Beberenice".encode())

    toolkit = FilesystemToolkit(base_path=tmp_path)
    config = TestConfig(
        ToolCallEvent(
            name="update_file",
            arguments=json.dumps({"path": "data.txt", "old_content": "old", "new_content": "新"}),
        ),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("update it")

    assert target.read_bytes() == "Café 新 Beberenice".encode()


@pytest.mark.asyncio
async def test_update_file(tmp_path: Path) -> None:
    (tmp_path / "data.txt").write_text("foo bar baz")

    toolkit = FilesystemToolkit(base_path=tmp_path)

    config = TestConfig(
        ToolCallEvent(
            name="update_file",
            arguments=json.dumps({"path": "data.txt", "old_content": "bar", "new_content": "qux"}),
        ),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("update it")

    assert (tmp_path / "data.txt").read_text() == "foo qux baz"


@pytest.mark.asyncio
async def test_delete_file(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    target = sub / "to_delete.txt"
    target.write_text("bye")

    toolkit = FilesystemToolkit(base_path=tmp_path)

    config = TestConfig(
        ToolCallEvent(name="delete_file", arguments=json.dumps({"path": "sub/to_delete.txt"})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("delete it")

    assert not target.exists()


@pytest.mark.asyncio
async def test_delete_directory(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    target = sub / "to_delete.txt"
    target.write_text("bye")

    toolkit = FilesystemToolkit(base_path=tmp_path)

    config = TestConfig(
        ToolCallEvent(name="delete_file", arguments=json.dumps({"path": "sub/"})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
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

    toolkit = FilesystemToolkit()

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(name="find_files", arguments=json.dumps({"pattern": "**/*.py"})),
            ToolCallEvent(name="find_files", arguments=json.dumps({"pattern": "sub/*"})),
            ToolCallEvent(name="find_files", arguments=json.dumps({"pattern": "sub/**"})),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[toolkit.find_files(tmp_path)])
    await agent.ask("find py files")

    # "**/*.py" — recursive, matches .py files at any depth
    tool_result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    result_1 = tool_result_msg.results[0].result.parts[0].data
    assert sorted(result_1) == ["a.py", str(Path("sub/c.py")), str(Path("sub/sub2/e.py"))]

    # "sub/*" — non-recursive, matches all files directly in sub/
    tool_result_msg: ToolResultsEvent = tracking.mock.call_args_list[2][0][0]
    result_2 = tool_result_msg.results[0].result.parts[0].data
    assert sorted(result_2) == [str(Path("sub/c.py")), str(Path("sub/d.txt"))]

    # "sub/**" — recursive, matches all files under sub/ at any depth
    tool_result_msg: ToolResultsEvent = tracking.mock.call_args_list[3][0][0]
    result_3 = tool_result_msg.results[0].result.parts[0].data
    assert sorted(result_3) == [str(Path("sub/c.py")), str(Path("sub/d.txt")), str(Path("sub/sub2/e.py"))]
