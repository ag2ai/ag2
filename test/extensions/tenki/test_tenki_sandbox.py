# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from tenki_sandbox import CommandResult

from ag2.annotations import Variable
from ag2.extensions.tenki.sandbox import TenkiSandbox
from ag2.tools.sandbox import ExecResult


def _fake_remote(
    *,
    result: CommandResult | None = None,
    state: str = "RUNNING",
) -> Any:
    command_result = result or CommandResult(argv=["echo", "ok"], exit_code=0, stdout=b"ok\n")
    return SimpleNamespace(
        id="sb-1",
        state=state,
        wait_ready=AsyncMock(return_value=None),
        exec=AsyncMock(return_value=command_result),
        fs=SimpleNamespace(
            write_bytes=AsyncMock(return_value=None),
            remove=AsyncMock(return_value=None),
        ),
        close_if_open=AsyncMock(return_value=None),
    )


def _fake_client(remote: Any) -> Any:
    project = SimpleNamespace(id="project-1")
    identity = SimpleNamespace(workspaces=(SimpleNamespace(projects=(project,)),))
    return SimpleNamespace(
        auth_token="test",  # pragma: allowlist secret
        base_url="https://api.tenki.cloud",
        create=AsyncMock(return_value=remote),
        who_am_i=AsyncMock(return_value=identity),
        close=AsyncMock(return_value=None),
    )


class TestConstruction:
    def test_invalid_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            TenkiSandbox(client=_fake_client(_fake_remote()), create_options={}, timeout=0)

    def test_workdir_is_posix(self) -> None:
        sandbox = TenkiSandbox(
            client=_fake_client(_fake_remote()),
            create_options={},
            workdir="/srv",
        )
        assert sandbox.workdir == PurePosixPath("/srv")

    def test_host_workdir_none(self) -> None:
        sandbox = TenkiSandbox(client=_fake_client(_fake_remote()), create_options={})
        assert sandbox.host_workdir is None

    def test_variable_rejected_in_constructor(self) -> None:
        with pytest.raises(TypeError):
            TenkiSandbox(client=Variable("client"), create_options={})  # type: ignore[arg-type]


@pytest.mark.asyncio
class TestExec:
    async def test_maps_argv_environment_workdir_timeout_and_output(self) -> None:
        result = CommandResult(
            argv=["python", "-c", "print(42)"],
            exit_code=3,
            stdout=b"out\n",
            stderr=b"err\n",
        )
        remote = _fake_remote(result=result)
        sandbox = TenkiSandbox(
            client=_fake_client(remote),
            create_options={"project_id": "project-1"},
            timeout=30,
        )

        actual = await sandbox.exec(["python", "-c", "print(42)"], env={"FOO": "bar"}, timeout=12)

        assert actual == ExecResult(output="out\nerr\n", exit_code=3)
        remote.exec.assert_awaited_once_with(
            "python",
            "-c",
            "print(42)",
            cwd="/home/tenki",
            env={"FOO": "bar"},
            timeout=12,
        )

    async def test_empty_argv_returns_failure(self) -> None:
        sandbox = TenkiSandbox(client=_fake_client(_fake_remote()), create_options={})
        assert await sandbox.exec([]) == ExecResult(output="", exit_code=2)

    async def test_reason_prevents_silent_failure(self) -> None:
        result = CommandResult(argv=["false"], exit_code=1, reason="process exited")
        sandbox = TenkiSandbox(
            client=_fake_client(_fake_remote(result=result)),
            create_options={"project_id": "project-1"},
        )
        assert await sandbox.exec(["false"]) == ExecResult(
            output="Tenki execution ended: process exited",
            exit_code=1,
        )


@pytest.mark.asyncio
class TestFileIO:
    async def test_put_and_remove_file_use_sdk(self) -> None:
        remote = _fake_remote()
        sandbox = TenkiSandbox(
            client=_fake_client(remote),
            create_options={"project_id": "project-1"},
            workdir="/srv",
        )
        await sandbox.put_file(PurePosixPath("hello.txt"), b"world")
        await sandbox.remove_file(PurePosixPath("hello.txt"))
        remote.fs.write_bytes.assert_awaited_once_with("/srv/hello.txt", b"world")
        remote.fs.remove.assert_awaited_once_with("/srv/hello.txt", recursive=False)

    async def test_absolute_paths_rejected(self) -> None:
        sandbox = TenkiSandbox(client=_fake_client(_fake_remote()), create_options={})
        with pytest.raises(ValueError, match="Absolute"):
            await sandbox.put_file(PurePosixPath("/etc/passwd"), b"x")
        with pytest.raises(ValueError, match="Absolute"):
            await sandbox.remove_file(PurePosixPath("/etc/passwd"))


@pytest.mark.asyncio
class TestLifecycle:
    async def test_aenter_creates_without_waiting_in_create_and_aclose_terminates(self) -> None:
        remote = _fake_remote()
        client = _fake_client(remote)
        sandbox = TenkiSandbox(client=client, create_options={"project_id": "project-1"})

        await sandbox.__aenter__()
        await sandbox.aclose()
        await sandbox.aclose()

        client.create.assert_awaited_once_with(wait=False, project_id="project-1")
        remote.close_if_open.assert_awaited_once()
        client.close.assert_awaited_once()

    async def test_project_is_discovered_when_omitted(self) -> None:
        remote = _fake_remote()
        client = _fake_client(remote)
        sandbox = TenkiSandbox(client=client, create_options={})

        await sandbox.__aenter__()

        client.who_am_i.assert_awaited_once()
        client.create.assert_awaited_once_with(wait=False, project_id="project-1")
        await sandbox.aclose()

    async def test_readiness_error_terminates_created_sandbox(self) -> None:
        remote = _fake_remote(state="CREATING")
        remote.wait_ready = AsyncMock(side_effect=RuntimeError("failed"))
        sandbox = TenkiSandbox(
            client=_fake_client(remote),
            create_options={"project_id": "project-1", "max_duration": 900},
        )

        with pytest.raises(RuntimeError, match="failed"):
            await sandbox.__aenter__()

        remote.close_if_open.assert_awaited_once()

    async def test_cancellation_terminates_created_sandbox(self) -> None:
        remote = _fake_remote(state="CREATING")
        remote.wait_ready = AsyncMock(side_effect=asyncio.CancelledError())
        sandbox = TenkiSandbox(
            client=_fake_client(remote),
            create_options={"project_id": "project-1", "max_duration": 900},
        )

        with pytest.raises(asyncio.CancelledError):
            await sandbox.__aenter__()

        remote.close_if_open.assert_awaited_once()
