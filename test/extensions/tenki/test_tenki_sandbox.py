# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Offline unit tests for TenkiSandbox; the Tenki SDK is fully mocked."""

import asyncio
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from tenki import CommandResult

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
    identity = SimpleNamespace(workspaces=(SimpleNamespace(id="workspace-1", name="ws"),))
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
            create_options={"workspace_id": "workspace-1"},
            timeout=30,
        )

        actual = await sandbox.exec(["python", "-c", "print(42)"], env={"FOO": "bar"}, timeout=12)

        # `ExecResult.output` is contracted to arrive already trimmed.
        assert actual == ExecResult(output="out\nerr", exit_code=3)
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
            create_options={"workspace_id": "workspace-1"},
        )
        assert await sandbox.exec(["false"]) == ExecResult(
            output="Tenki execution ended: process exited",
            exit_code=1,
        )

    @pytest.mark.parametrize(
        ("label", "result", "expected"),
        [
            # Every shape below was captured from the live Tenki API: a command
            # that never produced a wait status comes back as exit_code == -1,
            # which is not a POSIX status and must be translated.
            (
                "timeout",
                CommandResult(argv=["sleep", "10"], exit_code=-1, signal="terminated", reason="timeout"),
                124,
            ),
            (
                "not found",
                CommandResult(
                    argv=["nope"],
                    exit_code=-1,
                    reason='exec: "nope": executable file not found in $PATH',
                ),
                127,
            ),
            (
                "permission denied",
                CommandResult(
                    argv=["/home/tenki"],
                    exit_code=-1,
                    errno=13,
                    reason="fork/exec /home/tenki: permission denied",
                ),
                126,
            ),
            (
                "sigkill",
                CommandResult(argv=["sh"], exit_code=-1, signal="killed", reason="signaled"),
                128,
            ),
            (
                "sigterm",
                CommandResult(argv=["sh"], exit_code=-1, signal="terminated", reason="signaled"),
                128,
            ),
            (
                "unknown abnormal end",
                CommandResult(argv=["sh"], exit_code=-1, reason="something else"),
                1,
            ),
            ("clean failure keeps its status", CommandResult(argv=["sh"], exit_code=3, reason="exit"), 3),
        ],
    )
    async def test_abnormal_endings_map_onto_posix_exit_codes(
        self, label: str, result: CommandResult, expected: int
    ) -> None:
        sandbox = TenkiSandbox(
            client=_fake_client(_fake_remote(result=result)),
            create_options={"workspace_id": "workspace-1"},
        )

        actual = await sandbox.exec(["sh"])

        assert actual.exit_code == expected, f"{label}: got {actual.exit_code}"
        assert actual.exit_code >= 0

    async def test_timeout_result_is_reported_with_the_budget(self) -> None:
        # Tenki returns a result for an exec timeout instead of raising, so the
        # message has to come from the result path.
        result = CommandResult(argv=["sleep", "10"], exit_code=-1, signal="terminated", reason="timeout")
        sandbox = TenkiSandbox(
            client=_fake_client(_fake_remote(result=result)),
            create_options={"workspace_id": "workspace-1"},
        )

        assert await sandbox.exec(["sleep", "10"], timeout=2) == ExecResult(
            output="Tenki execution timed out after 2s",
            exit_code=124,
        )

    async def test_silent_success_stays_silent(self) -> None:
        # Tenki reports reason="exit" on a clean finish too, so a command that
        # simply prints nothing must not be dressed up as an abnormal ending.
        result = CommandResult(argv=["touch", "f"], exit_code=0, reason="exit")
        sandbox = TenkiSandbox(
            client=_fake_client(_fake_remote(result=result)),
            create_options={"workspace_id": "workspace-1"},
        )
        assert await sandbox.exec(["touch", "f"]) == ExecResult(output="", exit_code=0)


@pytest.mark.asyncio
class TestFileIO:
    async def test_put_and_remove_file_use_sdk(self) -> None:
        remote = _fake_remote()
        sandbox = TenkiSandbox(
            client=_fake_client(remote),
            create_options={"workspace_id": "workspace-1"},
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
        sandbox = TenkiSandbox(client=client, create_options={"workspace_id": "workspace-1"})

        await sandbox.__aenter__()
        await sandbox.aclose()
        await sandbox.aclose()

        client.create.assert_awaited_once_with(wait=False, workspace_id="workspace-1")
        remote.close_if_open.assert_awaited_once()
        client.close.assert_awaited_once()

    async def test_failed_close_is_not_raised_and_stays_retryable(self) -> None:
        remote = _fake_remote()
        remote.close_if_open = AsyncMock(side_effect=RuntimeError("server unreachable"))
        client = _fake_client(remote)
        sandbox = TenkiSandbox(client=client, create_options={"workspace_id": "workspace-1"})
        await sandbox.__aenter__()

        await sandbox.aclose()

        assert sandbox.closed
        # The retained session handle routes through the client, so a failed
        # close must leave the client open — otherwise the retry below would
        # run against a dead transport and silently do nothing.
        client.close.assert_not_awaited()
        # The session is still alive, so a second close retries it rather than
        # dropping the handle and leaking the sandbox until `max_duration`.
        await sandbox.aclose()
        assert remote.close_if_open.await_count == 2

    async def test_successful_close_releases_the_client(self) -> None:
        remote = _fake_remote()
        client = _fake_client(remote)
        sandbox = TenkiSandbox(client=client, create_options={"workspace_id": "workspace-1"})
        await sandbox.__aenter__()

        await sandbox.aclose()

        remote.close_if_open.assert_awaited_once()
        client.close.assert_awaited_once()

    async def test_workspace_is_discovered_when_omitted(self) -> None:
        remote = _fake_remote()
        client = _fake_client(remote)
        sandbox = TenkiSandbox(client=client, create_options={})

        await sandbox.__aenter__()

        client.who_am_i.assert_awaited_once()
        client.create.assert_awaited_once_with(wait=False, workspace_id="workspace-1")
        await sandbox.aclose()

    async def test_readiness_error_terminates_created_sandbox(self) -> None:
        remote = _fake_remote(state="CREATING")
        remote.wait_ready = AsyncMock(side_effect=RuntimeError("failed"))
        sandbox = TenkiSandbox(
            client=_fake_client(remote),
            create_options={"workspace_id": "workspace-1", "max_duration": 900},
        )

        with pytest.raises(RuntimeError, match="failed"):
            await sandbox.__aenter__()

        remote.close_if_open.assert_awaited_once()

    async def test_cancellation_terminates_created_sandbox(self) -> None:
        remote = _fake_remote(state="CREATING")
        remote.wait_ready = AsyncMock(side_effect=asyncio.CancelledError())
        sandbox = TenkiSandbox(
            client=_fake_client(remote),
            create_options={"workspace_id": "workspace-1", "max_duration": 900},
        )

        with pytest.raises(asyncio.CancelledError):
            await sandbox.__aenter__()

        remote.close_if_open.assert_awaited_once()
