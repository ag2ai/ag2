# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import daytona
import pytest

from ag2.annotations import Variable
from ag2.extensions.daytona.sandbox import DaytonaSandbox
from ag2.tools.sandbox import ExecResult


def _fake_sandbox(result: str = "ok", exit_code: int = 0) -> Any:
    response = SimpleNamespace(result=result, exit_code=exit_code)
    return SimpleNamespace(
        id="sb-1",
        process=SimpleNamespace(
            exec=AsyncMock(return_value=response),
        ),
        fs=SimpleNamespace(
            upload_file=AsyncMock(return_value=None),
            download_file=AsyncMock(return_value=b"contents"),
            delete_file=AsyncMock(return_value=None),
        ),
        delete=AsyncMock(return_value=None),
    )


def _fake_client(sandbox: Any) -> Any:
    return SimpleNamespace(
        create=AsyncMock(return_value=sandbox),
        close=AsyncMock(return_value=None),
    )


class TestConstruction:
    def test_invalid_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            DaytonaSandbox(client=_fake_client(_fake_sandbox()), params={}, timeout=0)

    def test_workdir_is_posix(self) -> None:
        sandbox = DaytonaSandbox(client=_fake_client(_fake_sandbox()), params={}, workdir="/srv")
        assert sandbox.workdir == PurePosixPath("/srv")

    def test_host_workdir_none(self) -> None:
        sandbox = DaytonaSandbox(client=_fake_client(_fake_sandbox()), params={})
        assert sandbox.host_workdir is None

    def test_variable_rejected_in_constructor(self) -> None:
        with pytest.raises(TypeError):
            DaytonaSandbox(client=Variable("c"), params={})  # type: ignore[arg-type]


@pytest.mark.asyncio
class TestExec:
    async def test_argv_is_joined_into_shell_command(self) -> None:
        remote = _fake_sandbox(result="42\n")
        client = _fake_client(remote)
        sandbox = DaytonaSandbox(client=client, params={})
        result = await sandbox.exec(["python", "-c", "print(40+2)"])

        assert result == ExecResult(output="42\n", exit_code=0)
        remote.process.exec.assert_awaited_once()
        cmd = remote.process.exec.await_args.args[0]
        assert cmd.startswith("python -c ")

    async def test_empty_argv_returns_failure(self) -> None:
        sandbox = DaytonaSandbox(client=_fake_client(_fake_sandbox()), params={})
        result = await sandbox.exec([])
        assert result.exit_code == 2

    async def test_timeout_maps_to_exit_124(self) -> None:
        remote = _fake_sandbox()
        remote.process.exec = AsyncMock(side_effect=daytona.DaytonaTimeoutError("slow"))
        sandbox = DaytonaSandbox(client=_fake_client(remote), params={})
        result = await sandbox.exec(["sleep", "5"])
        assert result.exit_code == 124


@pytest.mark.asyncio
class TestFileIO:
    async def test_put_file_uses_upload_file(self) -> None:
        remote = _fake_sandbox()
        sandbox = DaytonaSandbox(client=_fake_client(remote), params={}, workdir="/srv")
        await sandbox.put_file(PurePosixPath("hello.txt"), b"world")
        remote.fs.upload_file.assert_awaited_once_with(b"world", "/srv/hello.txt")

    async def test_absolute_path_rejected(self) -> None:
        sandbox = DaytonaSandbox(client=_fake_client(_fake_sandbox()), params={})
        with pytest.raises(ValueError, match="Absolute"):
            await sandbox.put_file(PurePosixPath("/etc/passwd"), b"x")


@pytest.mark.asyncio
class TestLifecycle:
    async def test_aenter_creates_sandbox(self) -> None:
        remote = _fake_sandbox()
        client = _fake_client(remote)
        async with DaytonaSandbox(client=client, params={}):
            pass
        client.create.assert_awaited_once()
        remote.delete.assert_awaited_once()
        client.close.assert_awaited_once()

    async def test_aclose_idempotent(self) -> None:
        remote = _fake_sandbox()
        client = _fake_client(remote)
        sandbox = DaytonaSandbox(client=client, params={})
        await sandbox.exec(["echo", "hi"])
        await sandbox.aclose()
        await sandbox.aclose()
        remote.delete.assert_awaited_once()
