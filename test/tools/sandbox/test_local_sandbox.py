# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path, PurePosixPath
from threading import Event
from unittest.mock import patch

import pytest

from ag2.tools.sandbox import ExecResult, Sandbox
from ag2.tools.sandbox.local import LocalSandbox


async def _wait_for_marker(path: Path) -> None:
    while not path.exists():  # noqa: ASYNC110 - the child signals through the filesystem
        await asyncio.sleep(0.005)


async def _execute_during_shutdown(sandbox: LocalSandbox, program: str) -> None:
    try:
        await sandbox.exec([sys.executable, "-c", program])
    except asyncio.CancelledError:
        await sandbox.aclose()
        (sandbox.host_workdir / "release").touch()
        raise


async def _start_command_before_shutdown(sandbox: LocalSandbox, program: str) -> None:
    execution = asyncio.create_task(_execute_during_shutdown(sandbox, program))
    await asyncio.wait_for(_wait_for_marker(sandbox.host_workdir / "started"), timeout=2)
    await asyncio.sleep(0.025)
    assert not execution.done()


async def _cancel_queued_command(sandbox: LocalSandbox) -> None:
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    release = Event()
    occupied = loop.run_in_executor(None, release.wait, 3)
    execution = asyncio.create_task(
        sandbox.exec([sys.executable, "-c", "from pathlib import Path; Path('launched').touch()"])
    )
    try:
        await asyncio.sleep(0)
        execution.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(execution, timeout=2)
        await occupied
        # Wait behind every queued command before inspecting its filesystem side effect.
        await loop.run_in_executor(None, str)
    finally:
        release.set()
        execution.cancel()
        with suppress(asyncio.CancelledError):
            await execution
        await sandbox.aclose()


def test_cancelled_queued_command_never_launches(tmp_path: Path) -> None:
    sandbox = LocalSandbox(tmp_path, timeout=2)
    with patch("ag2.tools.sandbox.local.subprocess.Popen", wraps=subprocess.Popen) as launch:
        asyncio.run(_cancel_queued_command(sandbox))
    launch.assert_not_called()
    assert not (tmp_path / "launched").exists()


@pytest.mark.parametrize("attempt", [0, 1, 2])
def test_event_loop_shutdown_stops_running_command(tmp_path: Path, attempt: int) -> None:
    root = tmp_path / str(attempt)
    sandbox = LocalSandbox(root, timeout=3)
    program = (
        "from pathlib import Path; import time; "
        "Path('started').touch(); deadline = time.monotonic() + 2; "
        "\nwhile not Path('release').exists() and time.monotonic() < deadline: time.sleep(0.001)"
        "\nif Path('release').exists(): Path('completed').touch()"
    )
    asyncio.run(_start_command_before_shutdown(sandbox, program))
    assert not (root / "completed").exists()


@pytest.mark.asyncio
class TestLocalSandboxExec:
    async def test_argv_form_runs_command(self) -> None:
        sandbox = LocalSandbox()
        result = await sandbox.exec(["python3", "-c", "print('hi')"])
        assert result == ExecResult(output="hi", exit_code=0)

    async def test_shell_pipeline_via_sh_dash_c(self) -> None:
        sandbox = LocalSandbox()
        result = await sandbox.exec(["sh", "-c", "echo hello && echo world"])
        assert result == ExecResult(output="hello\nworld", exit_code=0)

    async def test_legacy_shell_kwarg_rejected(self) -> None:
        sandbox = LocalSandbox()
        with pytest.raises(TypeError):
            await sandbox.exec(["echo", "x"], shell=True)  # type: ignore[call-arg]

    async def test_workdir_is_used_as_cwd(self, tmp_path: Path) -> None:
        (tmp_path / "marker.txt").write_text("hello")
        sandbox = LocalSandbox(tmp_path)
        result = await sandbox.exec(["cat", "marker.txt"])
        assert result == ExecResult(output="hello", exit_code=0)

    async def test_extra_env_vars_are_merged(self) -> None:
        sandbox = LocalSandbox()
        result = await sandbox.exec(["sh", "-c", "echo $FOO"], env={"FOO": "bar"})
        assert result == ExecResult(output="bar", exit_code=0)

    async def test_nonzero_exit_code_preserved(self) -> None:
        sandbox = LocalSandbox()
        result = await sandbox.exec(["sh", "-c", "exit 7"])
        assert result.exit_code == 7

    async def test_default_timeout_returns_124(self) -> None:
        sandbox = LocalSandbox(timeout=0.1)
        result = await sandbox.exec(["sleep", "5"])
        assert result.exit_code == 124
        assert "timed out" in result.output.lower()

    async def test_per_call_timeout_overrides_default(self) -> None:
        sandbox = LocalSandbox(timeout=10)
        result = await sandbox.exec(["sleep", "5"], timeout=0.1)
        assert result.exit_code == 124

    @pytest.mark.parametrize("cancel_again", [False, True])
    async def test_cancel_stops_running_command(self, tmp_path: Path, cancel_again: bool) -> None:
        program = (
            "from pathlib import Path; import time; "
            "Path('started').touch(); deadline = time.monotonic() + 3; "
            "\nwhile not Path('release').exists() and time.monotonic() < deadline: time.sleep(0.01)"
            "\nif Path('release').exists(): Path('completed').touch()"
        )
        sandbox = LocalSandbox(tmp_path, timeout=5)
        execution = asyncio.create_task(sandbox.exec([sys.executable, "-c", program]))
        try:
            await asyncio.wait_for(_wait_for_marker(tmp_path / "started"), timeout=2)
            execution.cancel()
            if cancel_again:
                asyncio.get_running_loop().call_later(0.01, execution.cancel)
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(execution, timeout=2)
            await sandbox.aclose()
            (tmp_path / "release").touch()
            await asyncio.sleep(0.3)
            assert not (tmp_path / "completed").exists()
        finally:
            (tmp_path / "release").touch()
            execution.cancel()
            with suppress(asyncio.CancelledError):
                await execution
            await sandbox.aclose()

    @pytest.mark.parametrize("max_output", [100, 10])
    async def test_output_survives_multiple_collection_intervals(self, max_output: int) -> None:
        program = (
            "import sys, time; sys.stdout.buffer.write(b'one\\r\\ntwo\\rthree\\n'); "
            "sys.stdout.flush(); time.sleep(0.25); sys.stderr.write('err\\n'); sys.exit(7)"
        )
        async with LocalSandbox(max_output=max_output) as sandbox:
            result = await sandbox.exec([sys.executable, "-c", program])
        output = (
            "one\ntwo\nthree\nerr" if max_output == 100 else "one\ntwo\nth\n[truncated: showing first 10 of 17 chars]"
        )
        assert result == ExecResult(output=output, exit_code=7)

    async def test_output_is_truncated_when_too_long(self) -> None:
        sandbox = LocalSandbox(max_output=10)
        long = "x" * 200
        result = await sandbox.exec(["python3", "-c", f"print('{long}')"])
        assert result.exit_code == 0
        assert "[truncated:" in result.output

    async def test_command_not_found_returns_127(self) -> None:
        sandbox = LocalSandbox()
        result = await sandbox.exec(["this-command-does-not-exist-xyz"])
        assert result.exit_code == 127

    async def test_empty_argv_returns_failure(self) -> None:
        sandbox = LocalSandbox()
        result = await sandbox.exec([])
        assert result == ExecResult(output="", exit_code=2)

    async def test_aclose_blocks_further_exec(self) -> None:
        sandbox = LocalSandbox()
        await sandbox.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await sandbox.exec(["echo", "hi"])

    async def test_aclose_is_idempotent(self) -> None:
        sandbox = LocalSandbox()
        await sandbox.aclose()
        await sandbox.aclose()


@pytest.mark.asyncio
class TestLocalSandboxProperties:
    async def test_workdir_returns_user_path(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        assert sandbox.workdir == PurePosixPath(tmp_path.resolve().as_posix())
        assert sandbox.host_workdir == tmp_path.resolve()

    async def test_workdir_temp_when_path_none(self) -> None:
        sandbox = LocalSandbox()
        assert isinstance(sandbox.workdir, PurePosixPath)
        assert sandbox.host_workdir is not None
        assert sandbox.host_workdir.exists()
        assert "ag2_sandbox_" in sandbox.host_workdir.name

    async def test_workdir_is_created_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "dir"
        sandbox = LocalSandbox(nested)
        assert sandbox.host_workdir == nested.resolve()
        assert sandbox.host_workdir.exists()

    async def test_supported_languages_includes_python_by_default(self) -> None:
        sandbox = LocalSandbox()
        assert "python" in sandbox.supported_languages

    async def test_supported_languages_can_be_overridden(self) -> None:
        sandbox = LocalSandbox(languages=("python",))
        assert sandbox.supported_languages == ("python",)

    async def test_satisfies_sandbox_protocol(self) -> None:
        sandbox = LocalSandbox()
        assert isinstance(sandbox, Sandbox)


@pytest.mark.asyncio
class TestLocalSandboxLifecycle:
    async def test_aenter_returns_self(self) -> None:
        sandbox = LocalSandbox()
        async with sandbox as sb:
            assert sb is sandbox

    async def test_aexit_closes_sandbox(self) -> None:
        sandbox = LocalSandbox()
        async with sandbox:
            pass
        with pytest.raises(RuntimeError, match="closed"):
            await sandbox.exec(["echo", "hi"])


@pytest.mark.asyncio
class TestLocalSandboxFileIO:
    async def test_put_file_writes_relative_to_workdir(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        await sandbox.put_file(PurePosixPath("note.txt"), b"hello")
        assert (tmp_path / "note.txt").read_bytes() == b"hello"

    async def test_put_file_creates_intermediate_dirs(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        await sandbox.put_file(PurePosixPath("sub/dir/x.txt"), b"x")
        assert (tmp_path / "sub" / "dir" / "x.txt").read_bytes() == b"x"

    async def test_absolute_path_rejected(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        with pytest.raises(ValueError, match="Absolute"):
            await sandbox.put_file(PurePosixPath("/etc/passwd"), b"x")

    async def test_path_escape_rejected(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        with pytest.raises(ValueError, match="escapes"):
            await sandbox.put_file(PurePosixPath("../outside.txt"), b"x")

    async def test_remove_file_deletes(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        await sandbox.put_file(PurePosixPath("temp.txt"), b"x")
        assert (tmp_path / "temp.txt").exists()
        await sandbox.remove_file(PurePosixPath("temp.txt"))
        assert not (tmp_path / "temp.txt").exists()

    async def test_remove_file_missing_is_noop(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        # missing_ok=True under the hood — must not raise
        await sandbox.remove_file(PurePosixPath("never_existed.txt"))
