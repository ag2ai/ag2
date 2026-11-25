# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Offline unit tests for BoxdSandbox; the boxd SDK is fully mocked."""

import asyncio
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, call

import pytest
from boxd import APIConnectionError, BoxdError, NotFoundError
from boxd import ExecResult as BoxdExecResult

from ag2.annotations import Variable
from ag2.extensions.boxd.sandbox import BoxdSandbox
from ag2.tools.sandbox import ExecResult


def _result(stdout: str = "", stderr: str = "", exit_code: int = 0) -> BoxdExecResult:
    return BoxdExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code)


def _fake_client(*results: BoxdExecResult) -> Any:
    # The first exec is the `mkdir -p <workdir>` run while the machine is
    # prepared; tests pass the results of their own commands after it.
    return SimpleNamespace(
        base_url="https://boxd.sh:9443",
        region="fra",
        machines=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(id="m-1", name="ag2-test")),
            wait_until_ready=AsyncMock(return_value=None),
            exec=AsyncMock(side_effect=[_result(), *results] if results else None, return_value=_result()),
            delete=AsyncMock(return_value=None),
            fs=SimpleNamespace(write=AsyncMock(return_value=None)),
        ),
        close=AsyncMock(return_value=None),
    )


class TestConstruction:
    def test_invalid_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            BoxdSandbox(client=_fake_client(), create_options={}, timeout=0)

    def test_workdir_is_posix(self) -> None:
        sandbox = BoxdSandbox(client=_fake_client(), create_options={}, workdir="/srv")
        assert sandbox.workdir == PurePosixPath("/srv")

    def test_host_workdir_none(self) -> None:
        sandbox = BoxdSandbox(client=_fake_client(), create_options={})
        assert sandbox.host_workdir is None

    def test_invalid_default_env_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="environment variable name"):
            BoxdSandbox(client=_fake_client(), create_options={}, env={"1BAD": "x"})

    def test_variable_rejected_in_constructor(self) -> None:
        with pytest.raises(TypeError):
            BoxdSandbox(client=Variable("client"), create_options={})


@pytest.mark.asyncio
class TestExec:
    async def test_maps_argv_environment_workdir_timeout_and_output(self) -> None:
        client = _fake_client(_result(stdout="out\n", stderr="err\n", exit_code=3))
        sandbox = BoxdSandbox(client=client, create_options={}, timeout=30)

        actual = await sandbox.exec(["python", "-c", "print(42)"], env={"FOO": "bar"}, timeout=12)

        # `ExecResult.output` is contracted to arrive already trimmed.
        assert actual == ExecResult(output="out\nerr", exit_code=3)
        assert client.machines.exec.await_args_list[-1] == call(
            "m-1",
            ["python", "-c", "print(42)"],
            env={"FOO": "bar"},
            cwd="/home/boxd",
            timeout=12,
        )

    async def test_default_timeout_and_no_env(self) -> None:
        client = _fake_client(_result(stdout="ok"))
        sandbox = BoxdSandbox(client=client, create_options={}, timeout=7)

        await sandbox.exec(["true"])

        assert client.machines.exec.await_args_list[-1].kwargs == {"env": None, "cwd": "/home/boxd", "timeout": 7}

    async def test_empty_argv_returns_failure(self) -> None:
        sandbox = BoxdSandbox(client=_fake_client(), create_options={})
        assert await sandbox.exec([]) == ExecResult(output="", exit_code=2)

    async def test_invalid_env_name_rejected_before_anything_runs(self) -> None:
        # The SDK prefixes `NAME=value` to a shell line, so a name must be an identifier.
        client = _fake_client()
        sandbox = BoxdSandbox(client=client, create_options={})
        with pytest.raises(ValueError, match="environment variable name"):
            await sandbox.exec(["env"], env={"A; rm -rf ~; B": "x"})
        client.machines.create.assert_not_awaited()

    @pytest.mark.parametrize(
        ("label", "exit_code"),
        [
            # Shapes captured from a live boxd machine: commands run through the
            # machine's shell, so these arrive as POSIX codes already.
            ("not found", 127),
            ("permission denied", 126),
            ("sigkill", 137),
            ("clean failure keeps its status", 3),
        ],
    )
    async def test_shell_exit_codes_pass_through(self, label: str, exit_code: int) -> None:
        sandbox = BoxdSandbox(client=_fake_client(_result(stderr="x", exit_code=exit_code)), create_options={})

        actual = await sandbox.exec(["sh"])

        assert actual.exit_code == exit_code, label

    async def test_missing_exit_status_is_a_failure(self) -> None:
        sandbox = BoxdSandbox(client=_fake_client(_result(stdout="partial", exit_code=-1)), create_options={})
        assert await sandbox.exec(["sh"]) == ExecResult(
            output="partial\nboxd execution ended without an exit status",
            exit_code=1,
        )

    async def test_sdk_timeout_is_reported_with_the_budget(self) -> None:
        # The SDK raises a code-less BoxdError when `timeout=` runs out.
        client = _fake_client()
        client.machines.exec = AsyncMock(side_effect=[_result(), BoxdError("exec timed out after 2s")])
        sandbox = BoxdSandbox(client=client, create_options={})

        assert await sandbox.exec(["sleep", "10"], timeout=2) == ExecResult(
            output="boxd execution timed out after 2s",
            exit_code=124,
        )

    async def test_other_sdk_errors_become_a_failed_result(self) -> None:
        client = _fake_client()
        client.machines.exec = AsyncMock(side_effect=[_result(), APIConnectionError("agent gone")])
        sandbox = BoxdSandbox(client=client, create_options={})

        assert await sandbox.exec(["true"]) == ExecResult(output="boxd error: agent gone", exit_code=1)

    async def test_bytes_output_is_decoded(self) -> None:
        sandbox = BoxdSandbox(
            client=_fake_client(BoxdExecResult(stdout=b"a\xff", stderr=b"", exit_code=0)),
            create_options={},
        )
        assert await sandbox.exec(["sh"]) == ExecResult(output="a�", exit_code=0)

    async def test_silent_success_stays_silent(self) -> None:
        sandbox = BoxdSandbox(client=_fake_client(_result()), create_options={})
        assert await sandbox.exec(["touch", "f"]) == ExecResult(output="", exit_code=0)


@pytest.mark.asyncio
class TestFileIO:
    async def test_put_file_writes_under_workdir(self) -> None:
        client = _fake_client()
        sandbox = BoxdSandbox(client=client, create_options={}, workdir="/srv")

        await sandbox.put_file(PurePosixPath("sub/hello.txt"), b"world")

        # `fs.write` creates the missing `sub/` directory.
        client.machines.fs.write.assert_awaited_once_with("m-1", "/srv/sub/hello.txt", b"world")

    async def test_remove_file_runs_rm_f_in_workdir(self) -> None:
        client = _fake_client()
        sandbox = BoxdSandbox(client=client, create_options={}, workdir="/srv")

        await sandbox.remove_file(PurePosixPath("hello.txt"))

        assert client.machines.exec.await_args_list[-1].args == ("m-1", ["rm", "-f", "/srv/hello.txt"])

    async def test_absolute_paths_rejected(self) -> None:
        sandbox = BoxdSandbox(client=_fake_client(), create_options={})
        with pytest.raises(ValueError, match="Absolute"):
            await sandbox.put_file(PurePosixPath("/etc/passwd"), b"x")
        with pytest.raises(ValueError, match="Absolute"):
            await sandbox.remove_file(PurePosixPath("/etc/passwd"))


@pytest.mark.asyncio
class TestLifecycle:
    async def test_scope_creates_prepares_workdir_and_deletes_on_exit(self) -> None:
        client = _fake_client()
        sandbox = BoxdSandbox(client=client, create_options={"name": "n", "isolated": True}, workdir="/home/boxd/w")

        async with sandbox:
            assert sandbox.machine_id == "m-1"
        await sandbox.aclose()

        client.machines.create.assert_awaited_once_with(name="n", isolated=True)
        client.machines.wait_until_ready.assert_awaited_once_with("m-1", timeout=60)
        client.machines.exec.assert_awaited_once_with("m-1", ["mkdir", "-p", "/home/boxd/w"], timeout=60)
        client.machines.delete.assert_awaited_once_with("m-1")
        client.close.assert_awaited_once()
        assert sandbox.machine_id is None

    async def test_machine_is_created_once_across_calls(self) -> None:
        client = _fake_client()
        sandbox = BoxdSandbox(client=client, create_options={})

        await asyncio.gather(sandbox.exec(["true"]), sandbox.exec(["true"]), sandbox.put_file(PurePosixPath("f"), b""))

        client.machines.create.assert_awaited_once()

    async def test_use_after_close_raises(self) -> None:
        sandbox = BoxdSandbox(client=_fake_client(), create_options={})
        await sandbox.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await sandbox.exec(["true"])

    async def test_failed_close_is_not_raised_and_stays_retryable(self) -> None:
        client = _fake_client()
        client.machines.delete = AsyncMock(side_effect=[APIConnectionError("unreachable"), None])
        sandbox = BoxdSandbox(client=client, create_options={})
        await sandbox.exec(["true"])

        await sandbox.aclose()

        assert sandbox.closed
        # The client is still needed to retry the delete.
        client.close.assert_not_awaited()
        assert sandbox.machine_id == "m-1"
        await sandbox.aclose()
        assert client.machines.delete.await_count == 2
        client.close.assert_awaited_once()

    async def test_machine_already_gone_counts_as_deleted(self) -> None:
        client = _fake_client()
        client.machines.delete = AsyncMock(side_effect=NotFoundError("gone", code="not_found"))
        sandbox = BoxdSandbox(client=client, create_options={})
        await sandbox.exec(["true"])

        await sandbox.aclose()

        client.close.assert_awaited_once()
        assert sandbox.machine_id is None

    async def test_readiness_error_deletes_created_machine(self) -> None:
        client = _fake_client()
        client.machines.wait_until_ready = AsyncMock(side_effect=TimeoutError("not ready"))
        sandbox = BoxdSandbox(client=client, create_options={})

        with pytest.raises(TimeoutError, match="not ready"):
            async with sandbox:
                pass

        client.machines.delete.assert_awaited_once_with("m-1")
        assert sandbox.machine_id is None

    async def test_workdir_failure_deletes_created_machine(self) -> None:
        client = _fake_client()
        client.machines.exec = AsyncMock(return_value=_result(stderr="Permission denied", exit_code=1))
        sandbox = BoxdSandbox(client=client, create_options={}, workdir="/root/x")

        with pytest.raises(RuntimeError, match="Could not create workdir /root/x.*Permission denied"):
            async with sandbox:
                pass

        client.machines.delete.assert_awaited_once_with("m-1")

    async def test_cancellation_deletes_created_machine(self) -> None:
        client = _fake_client()
        client.machines.wait_until_ready = AsyncMock(side_effect=asyncio.CancelledError())
        sandbox = BoxdSandbox(client=client, create_options={})

        with pytest.raises(asyncio.CancelledError):
            async with sandbox:
                pass

        client.machines.delete.assert_awaited_once_with("m-1")

    async def test_failed_cleanup_keeps_the_machine_for_aclose(self) -> None:
        client = _fake_client()
        client.machines.wait_until_ready = AsyncMock(side_effect=TimeoutError("not ready"))
        client.machines.delete = AsyncMock(side_effect=[APIConnectionError("unreachable"), None])
        sandbox = BoxdSandbox(client=client, create_options={})

        with pytest.raises(TimeoutError, match="not ready"):
            async with sandbox:
                pass
        assert sandbox.machine_id == "m-1"
        await sandbox.aclose()

        assert client.machines.delete.await_count == 2
        assert sandbox.machine_id is None
        client.close.assert_awaited_once()

    async def test_no_new_machine_while_a_failed_one_is_still_alive(self) -> None:
        client = _fake_client()
        client.machines.create = AsyncMock(
            side_effect=[SimpleNamespace(id="m-1", name="first"), SimpleNamespace(id="m-2", name="second")]
        )
        client.machines.wait_until_ready = AsyncMock(side_effect=[TimeoutError("not ready"), None])
        client.machines.delete = AsyncMock(
            side_effect=[APIConnectionError("unreachable"), APIConnectionError("unreachable"), None, None]
        )
        sandbox = BoxdSandbox(client=client, create_options={})

        with pytest.raises(TimeoutError, match="not ready"):
            await sandbox.exec(["true"])
        with pytest.raises(RuntimeError, match="could not be deleted"):
            await sandbox.exec(["true"])
        assert (await sandbox.exec(["true"])).exit_code == 0
        assert sandbox.machine_id == "m-2"
        await sandbox.aclose()

        assert client.machines.create.await_count == 2
        assert client.machines.delete.await_args_list == [call("m-1"), call("m-1"), call("m-1"), call("m-2")]

    async def test_close_waits_for_a_machine_being_created(self) -> None:
        client = _fake_client()
        gate = asyncio.Event()
        client.machines.create = AsyncMock(side_effect=_released_by(gate))
        sandbox = BoxdSandbox(client=client, create_options={})

        running = asyncio.create_task(sandbox.exec(["true"]))
        await asyncio.sleep(0.01)
        closing = asyncio.create_task(sandbox.aclose())
        await asyncio.sleep(0.01)
        gate.set()
        await asyncio.gather(running, closing)

        client.machines.delete.assert_awaited_once_with("m-1")
        assert sandbox.machine_id is None


def _released_by(gate: asyncio.Event) -> Any:
    # A machine creation that takes until the test releases it, like a slow network round-trip.
    async def create(**kwargs: Any) -> Any:
        await gate.wait()
        return SimpleNamespace(id="m-1", name="ag2-test")

    return create
