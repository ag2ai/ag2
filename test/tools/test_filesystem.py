# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ag2 import Agent, Context, DataInput
from ag2.events import ToolCallEvent, ToolErrorEvent, ToolResultsEvent
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools import FilesystemToolkit
from ag2.tools.toolkits.filesystem import _check_pattern, _resolve_path


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
async def test_update_file_rejects_empty_old_content_on_non_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "data.txt"
    target.write_text("original", encoding="utf-8")

    toolkit = FilesystemToolkit(base_path=tmp_path)
    config = TestConfig(
        ToolCallEvent(
            name="update_file",
            arguments=json.dumps({"path": "data.txt", "old_content": "", "new_content": "unexpected"}),
        ),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])

    with pytest.raises(ValueError, match="old_content must not be empty"):
        await agent.ask("update it")

    assert target.read_text(encoding="utf-8") == "original"


@pytest.mark.asyncio
async def test_update_file_empty_old_content_fills_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "data.txt"
    target.write_text("", encoding="utf-8")

    toolkit = FilesystemToolkit(base_path=tmp_path)
    config = TestConfig(
        ToolCallEvent(
            name="update_file",
            arguments=json.dumps({"path": "data.txt", "old_content": "", "new_content": "filled"}),
        ),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])

    await agent.ask("fill it")

    assert target.read_text(encoding="utf-8") == "filled"


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


def _symlink(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError:
        pytest.skip("symlinks are not available")


@pytest.mark.asyncio
async def test_delete_symlink_to_file_keeps_target(tmp_path: Path) -> None:
    target = tmp_path / "guide.md"
    target.write_text("keep me")
    link = tmp_path / "legacy.md"
    _symlink(link, target)

    toolkit = FilesystemToolkit(base_path=tmp_path)

    config = TestConfig(
        ToolCallEvent(name="delete_file", arguments=json.dumps({"path": "legacy.md"})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("delete it")

    assert not link.is_symlink()
    assert target.read_text() == "keep me"


@pytest.mark.asyncio
# A trailing slash still names the link, never the directory it points to.
@pytest.mark.parametrize("path", ["alias", "alias/"])
async def test_delete_symlink_to_directory_keeps_target(tmp_path: Path, path: str) -> None:
    target = tmp_path / "real"
    target.mkdir()
    (target / "data.txt").write_text("keep me")
    link = tmp_path / "alias"
    _symlink(link, target, target_is_directory=True)

    toolkit = FilesystemToolkit(base_path=tmp_path)

    config = TestConfig(
        ToolCallEvent(name="delete_file", arguments=json.dumps({"path": path})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("delete it")

    assert not link.is_symlink()
    assert (target / "data.txt").read_text() == "keep me"


@pytest.mark.asyncio
@pytest.mark.parametrize("target_name", ["missing", "../missing"])
async def test_delete_dangling_symlink(tmp_path: Path, target_name: str) -> None:
    base = tmp_path / "base"
    base.mkdir()
    link = base / "link"
    _symlink(link, Path(target_name))

    toolkit = FilesystemToolkit(base_path=base)

    config = TestConfig(
        ToolCallEvent(name="delete_file", arguments=json.dumps({"path": "link"})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("delete it")

    assert not link.is_symlink()


@pytest.mark.asyncio
async def test_delete_symlink_pointing_out_of_base_dir_keeps_target(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("keep me")
    link = base / "link"
    _symlink(link, target)

    toolkit = FilesystemToolkit(base_path=base)

    config = TestConfig(
        ToolCallEvent(name="delete_file", arguments=json.dumps({"path": "link"})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("delete it")

    assert not link.is_symlink()
    assert target.read_text() == "keep me"


@pytest.mark.asyncio
async def test_delete_symlink_through_escaping_parent_blocked(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    target = base / "target.txt"
    target.write_text("keep me")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = outside / "link"
    _symlink(link, target)
    _symlink(base / "escape", outside, target_is_directory=True)

    toolkit = FilesystemToolkit(base_path=base)

    config = TestConfig(
        ToolCallEvent(name="delete_file", arguments=json.dumps({"path": "escape/link"})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    with pytest.raises(PermissionError, match="escapes base directory"):
        await agent.ask("delete it")

    assert link.is_symlink()
    assert target.read_text() == "keep me"


@pytest.fixture
def junction(tmp_path: Path) -> Path:
    if sys.platform != "win32":
        pytest.skip("directory junctions are Windows-only")
    target = tmp_path / "real"
    target.mkdir()
    (target / "data.txt").write_text("keep me")
    link = tmp_path / "alias"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    return link


@pytest.mark.asyncio
async def test_delete_junction_keeps_target(tmp_path: Path, junction: Path) -> None:
    target = tmp_path / "real"

    toolkit = FilesystemToolkit(base_path=tmp_path)

    config = TestConfig(
        ToolCallEvent(name="delete_file", arguments=json.dumps({"path": "alias"})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    await agent.ask("delete it")

    assert not junction.exists()
    assert (target / "data.txt").read_text() == "keep me"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [".", "", "sub/.."])
async def test_delete_base_dir_blocked(tmp_path: Path, path: str) -> None:
    (tmp_path / "sub").mkdir()
    kept = tmp_path / "kept.txt"
    kept.write_text("keep me")

    toolkit = FilesystemToolkit(base_path=tmp_path)

    config = TestConfig(
        ToolCallEvent(name="delete_file", arguments=json.dumps({"path": path})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    with pytest.raises(PermissionError, match="base directory itself"):
        await agent.ask("delete it")

    assert kept.read_text() == "keep me"


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


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        (
            "packages/*/src/**",
            ["packages/api/src/main.py", "packages/api/src/util/helper.py", "packages/web/src/app.py"],
        ),
        ("packages/a?i/src/**", ["packages/api/src/main.py", "packages/api/src/util/helper.py"]),
        (
            "packages/[aw]*/src/**",
            ["packages/api/src/main.py", "packages/api/src/util/helper.py", "packages/web/src/app.py"],
        ),
        (
            "packages/**/src/**",
            [
                "packages/api/src/main.py",
                "packages/api/src/util/helper.py",
                "packages/nested/worker/src/job.py",
                "packages/web/src/app.py",
            ],
        ),
        (
            "**/src/**",
            [
                "packages/api/src/main.py",
                "packages/api/src/util/helper.py",
                "packages/nested/worker/src/job.py",
                "packages/web/src/app.py",
                "src/root.py",
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_find_files_recursive_wildcard_prefix(tmp_path: Path, pattern: str, expected: list[str]) -> None:
    for name in [
        "packages/api/src/main.py",
        "packages/api/src/util/helper.py",
        "packages/web/src/app.py",
        "packages/nested/worker/src/job.py",
        "packages/api/tests/test_main.py",
        "src/root.py",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source", encoding="utf-8")
    (tmp_path / "packages/api/src/empty").mkdir()

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(name="find_files", arguments=json.dumps({"pattern": pattern})),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[FilesystemToolkit(tmp_path).find_files()])
    await agent.ask("find source files")

    tool_result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    [result] = tool_result_msg.results[0].result.parts
    assert isinstance(result, DataInput)
    assert result.data == [str(Path(name)) for name in expected]


@pytest.mark.parametrize("pattern", ["../**/*", "sub/../../*", "/etc/*"])
def test_pattern_traversal_blocked(pattern: str) -> None:
    with pytest.raises(PermissionError, match="escapes base directory"):
        _check_pattern(pattern)


@pytest.mark.asyncio
async def test_find_files_pattern_cannot_escape_base_dir(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "a.txt").write_text("a")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")

    toolkit = FilesystemToolkit(base_path=base)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(name="find_files", arguments=json.dumps({"pattern": "../**/*"})),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[toolkit.find_files()])
    with pytest.raises(PermissionError, match="escapes base directory"):
        await agent.ask("find files")

    tool_result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    [result] = tool_result_msg.results
    assert isinstance(result, ToolErrorEvent)
    assert "secret.txt" not in result.result.parts[0].content


@pytest.mark.parametrize(("pattern", "expected"), [("*/*", []), ("*/**", ["link"]), ("link/**", ["link"])])
@pytest.mark.asyncio
async def test_find_files_skips_symlink_target_out_of_base_dir(
    tmp_path: Path, pattern: str, expected: list[str]
) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "a.txt").write_text("a")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    try:
        (base / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not available")

    toolkit = FilesystemToolkit(base_path=base)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(name="find_files", arguments=json.dumps({"pattern": pattern})),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[toolkit.find_files()])
    await agent.ask("find files")

    tool_result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    assert tool_result_msg.results[0].result.parts[0].data == expected


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("alias/**", ["alias", "alias/.hidden.txt", "alias/nested/data.txt"]),
        (
            "*/**",
            ["alias", "alias/.hidden.txt", "alias/nested/data.txt", "real/.hidden.txt", "real/nested/data.txt"],
        ),
    ],
)
@pytest.mark.asyncio
async def test_find_files_recursive_pattern_preserves_symlink(
    tmp_path: Path, pattern: str, expected: list[str]
) -> None:
    target = tmp_path / "real"
    (target / "nested").mkdir(parents=True)
    (target / ".hidden.txt").write_text("hidden")
    (target / "nested/data.txt").write_text("nested")
    _symlink(tmp_path / "alias", target, target_is_directory=True)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(name="find_files", arguments=json.dumps({"pattern": pattern})),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[FilesystemToolkit(tmp_path).find_files()])
    await agent.ask("find files")

    tool_result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    [result] = tool_result_msg.results[0].result.parts
    assert isinstance(result, DataInput)
    assert result.data == [str(Path(name)) for name in expected]


@pytest.mark.asyncio
async def test_find_files_lists_symlinks(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "a.txt").write_text("a")
    (base / "real").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink(base / "file_link", base / "a.txt")
    _symlink(base / "dir_link", base / "real", target_is_directory=True)
    _symlink(base / "dangling", base / "missing")
    _symlink(base / "outside_link", outside, target_is_directory=True)

    toolkit = FilesystemToolkit(base_path=base)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(name="find_files", arguments=json.dumps({"pattern": "*"})),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[toolkit.find_files()])
    await agent.ask("find files")

    tool_result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    assert tool_result_msg.results[0].result.parts[0].data == [
        "a.txt",
        "dangling",
        "dir_link",
        "file_link",
        "outside_link",
    ]


@pytest.mark.parametrize(("pattern", "expected"), [("*/*", []), ("*/**", ["escape"])])
@pytest.mark.asyncio
async def test_find_files_skips_link_located_out_of_base_dir(tmp_path: Path, pattern: str, expected: list[str]) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "a.txt").write_text("a")
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink(outside / "link", base / "a.txt")
    _symlink(base / "escape", outside, target_is_directory=True)

    toolkit = FilesystemToolkit(base_path=base)

    tracking = TrackingConfig(
        TestConfig(
            ToolCallEvent(name="find_files", arguments=json.dumps({"pattern": pattern})),
            "done",
        )
    )
    agent = Agent("", config=tracking, tools=[toolkit.find_files()])
    await agent.ask("find files")

    tool_result_msg: ToolResultsEvent = tracking.mock.call_args_list[1][0][0]
    assert tool_result_msg.results[0].result.parts[0].data == expected


@pytest.mark.asyncio
async def test_escape_error_does_not_reveal_base_dir(tmp_path: Path) -> None:
    toolkit = FilesystemToolkit(base_path=tmp_path)

    config = TestConfig(
        ToolCallEvent(name="read_file", arguments=json.dumps({"path": "../secret.txt"})),
        "done",
    )
    agent = Agent("", config=config, tools=[toolkit])
    with pytest.raises(PermissionError) as exc_info:
        await agent.ask("read it")

    assert str(exc_info.value) == "Path '../secret.txt' escapes base directory"
