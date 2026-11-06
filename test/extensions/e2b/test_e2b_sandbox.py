# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Offline unit tests for E2BSandbox; the E2B SDK is fully mocked."""

import asyncio
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from e2b import CommandExitException, InvalidArgumentException, NotFoundException, TimeoutException

from ag2.annotations import Variable
from ag2.extensions.e2b.sandbox import E2BSandbox
from ag2.tools.sandbox import CodeAdapter, ExecResult, Sandbox


def _fake_remote(*, stdout: str = "ok\n", stderr: str = "", exit_code: int = 0) -> Any:
    return SimpleNamespace(
        sandbox_id="sbx-1",
        commands=SimpleNamespace(
            run=AsyncMock(return_value=SimpleNamespace(stdout=stdout, stderr=stderr, exit_code=exit_code)),
        ),
        files=SimpleNamespace(
            write=AsyncMock(return_value=None),
            remove=AsyncMock(return_value=None),
            make_dir=AsyncMock(return_value=True),
        ),
        set_timeout=AsyncMock(return_value=None),
        is_running=AsyncMock(return_value=True),
        kill=AsyncMock(return_value=True),
    )


def _patch_create(remote: Any) -> Any:
    return patch("ag2.extensions.e2b.sandbox.AsyncSandbox.create", AsyncMock(return_value=remote))


def _slow_create(remote: Any) -> Any:
    # Yields to the loop like a real network call, so concurrent callers interleave.
    async def create(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0)
        return remote

    return create


class TestConstruction:
    def test_satisfies_sandbox_protocol(self) -> None:
        assert isinstance(E2BSandbox(), Sandbox)

    def test_invalid_timeouts_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            E2BSandbox(timeout=0)
        with pytest.raises(ValueError, match="sandbox_timeout"):
            E2BSandbox(sandbox_timeout=0)

    def test_workdir_defaults_to_e2b_home_and_is_posix(self) -> None:
        assert E2BSandbox().workdir == PurePosixPath("/home/user")
        assert E2BSandbox(workdir="/srv").workdir == PurePosixPath("/srv")
        assert E2BSandbox().host_workdir is None

    def test_variable_rejected_in_constructor(self) -> None:
        with pytest.raises(TypeError, match="E2BEnvironment"):
            E2BSandbox(template=Variable("template"))  # type: ignore[arg-type]


@pytest.mark.asyncio
class TestExec:
    async def test_maps_argv_environment_workdir_timeout_and_output(self) -> None:
        remote = _fake_remote(stdout="out\n", stderr="err\n")
        with _patch_create(remote):
            sandbox = E2BSandbox(timeout=30)
            actual = await sandbox.exec(["python", "-c", "print('a b')"], env={"FOO": "bar"}, timeout=12)

        # `ExecResult.output` is contracted to arrive already trimmed.
        assert actual == ExecResult(output="out\nerr", exit_code=0)
        remote.commands.run.assert_awaited_once_with(
            "python -c 'print('\"'\"'a b'\"'\"')'",
            envs={"FOO": "bar"},
            cwd="/home/user",
            timeout=12,
        )

    async def test_default_timeout_is_used(self) -> None:
        remote = _fake_remote()
        with _patch_create(remote):
            await E2BSandbox(timeout=7).exec(["true"])
        assert remote.commands.run.await_args.kwargs["timeout"] == 7

    async def test_empty_argv_returns_failure(self) -> None:
        assert await E2BSandbox().exec([]) == ExecResult(output="", exit_code=2)

    async def test_nonzero_exit_keeps_status_and_output(self) -> None:
        # The SDK raises on a non-zero exit instead of returning a result.
        remote = _fake_remote()
        remote.commands.run = AsyncMock(
            side_effect=CommandExitException(stderr="boom\n", stdout="partial\n", exit_code=3, error=None)
        )
        with _patch_create(remote):
            assert await E2BSandbox().exec(["sh"]) == ExecResult(output="partial\nboom", exit_code=3)

    async def test_command_timeout_maps_to_124(self) -> None:
        remote = _fake_remote()
        remote.commands.run = AsyncMock(side_effect=TimeoutException("Request timed out"))
        with _patch_create(remote):
            actual = await E2BSandbox().exec(["sleep", "10"], timeout=2)
        assert actual == ExecResult(output="E2B execution timed out after 2s", exit_code=124)

    async def test_expired_sandbox_is_not_reported_as_a_command_timeout(self) -> None:
        # A dead sandbox also surfaces as TimeoutException ("The sandbox was not found").
        remote = _fake_remote()
        remote.commands.run = AsyncMock(side_effect=TimeoutException("The sandbox was not found"))
        remote.is_running = AsyncMock(return_value=False)
        with _patch_create(remote):
            actual = await E2BSandbox().exec(["ls"])
        assert actual.exit_code == 1
        assert "no longer available" in actual.output

    async def test_not_found_is_reported_as_gone(self) -> None:
        remote = _fake_remote()
        remote.set_timeout = AsyncMock(side_effect=NotFoundException("sandbox not found"))
        with _patch_create(remote) as create:
            sandbox = E2BSandbox(sandbox_timeout=300)
            # A command longer than the lifetime left forces a keep-alive call.
            actual = await sandbox.exec(["make"], timeout=600)
            # Never silently recreated: the old files would be gone.
            await sandbox.exec(["make"], timeout=600)
        assert actual.exit_code == 1
        assert "no longer available" in actual.output
        create.assert_awaited_once()

    async def test_other_sdk_errors_become_results(self) -> None:
        remote = _fake_remote()
        remote.commands.run = AsyncMock(side_effect=InvalidArgumentException("bad input"))
        with _patch_create(remote):
            assert await E2BSandbox().exec(["ls"]) == ExecResult(output="E2B error: bad input", exit_code=1)


@pytest.mark.asyncio
class TestKeepAlive:
    async def test_fresh_sandbox_is_not_refreshed(self) -> None:
        remote = _fake_remote()
        with _patch_create(remote):
            sandbox = E2BSandbox(sandbox_timeout=300)
            await sandbox.exec(["true"])
            await sandbox.exec(["true"])
        remote.set_timeout.assert_not_awaited()

    async def test_refreshes_once_half_the_lifetime_has_passed(self) -> None:
        remote = _fake_remote()
        clock = MagicMock(return_value=1000.0)
        with _patch_create(remote), patch("ag2.extensions.e2b.sandbox.time.monotonic", clock):
            sandbox = E2BSandbox(sandbox_timeout=300)
            await sandbox.exec(["true"])
            clock.return_value = 1149.0
            await sandbox.exec(["true"])
            remote.set_timeout.assert_not_awaited()
            clock.return_value = 1151.0
            await sandbox.exec(["true"])
            await sandbox.exec(["true"])
        remote.set_timeout.assert_awaited_once_with(300)

    async def test_long_command_extends_lifetime_beyond_its_timeout(self) -> None:
        remote = _fake_remote()
        with _patch_create(remote):
            await E2BSandbox(sandbox_timeout=300).exec(["make"], timeout=600)
        remote.set_timeout.assert_awaited_once_with(630)


@pytest.mark.asyncio
class TestFileIO:
    async def test_put_and_remove_file_resolve_against_workdir(self) -> None:
        remote = _fake_remote()
        with _patch_create(remote):
            sandbox = E2BSandbox(workdir="/srv")
            await sandbox.put_file(PurePosixPath("hello.txt"), b"world")
            await sandbox.remove_file(PurePosixPath("hello.txt"))
        remote.files.write.assert_awaited_once_with("/srv/hello.txt", b"world")
        remote.files.remove.assert_awaited_once_with("/srv/hello.txt")

    async def test_absolute_paths_rejected(self) -> None:
        sandbox = E2BSandbox()
        with pytest.raises(ValueError, match="Absolute"):
            await sandbox.put_file(PurePosixPath("/etc/passwd"), b"x")
        with pytest.raises(ValueError, match="Absolute"):
            await sandbox.remove_file(PurePosixPath("/etc/passwd"))

    async def test_code_adapter_runs_file_mode_language_and_cleans_up(self) -> None:
        remote = _fake_remote(stdout="js\n")
        with _patch_create(remote):
            result = await CodeAdapter(E2BSandbox(), languages=("javascript",)).run("console.log('js')", "javascript")
        assert result.output == "js"
        path = remote.files.write.await_args.args[0]
        assert path.startswith("/home/user/ag2_") and path.endswith(".js")
        remote.files.remove.assert_awaited_once_with(path)


@pytest.mark.asyncio
class TestLifecycle:
    async def test_aenter_creates_with_forwarded_options_and_aclose_kills(self) -> None:
        remote = _fake_remote()
        with _patch_create(remote) as create:
            sandbox = E2BSandbox(
                template="code-interpreter-v1",
                env_vars={"A": "1"},
                sandbox_timeout=120,
                workdir="/workspace",
                create_options={"api_key": "test", "metadata": {"app": "ag2"}},  # pragma: allowlist secret
            )
            await sandbox.__aenter__()
            assert sandbox.sandbox_id == "sbx-1"
            await sandbox.aclose()
            await sandbox.aclose()

        create.assert_awaited_once_with(
            "code-interpreter-v1",
            timeout=120,
            envs={"A": "1"},
            api_key="test",  # pragma: allowlist secret
            metadata={"app": "ag2"},
        )
        remote.files.make_dir.assert_awaited_once_with("/workspace")
        remote.kill.assert_awaited_once()
        assert sandbox.closed
        assert sandbox.sandbox_id is None

    async def test_use_after_close_raises(self) -> None:
        sandbox = E2BSandbox()
        await sandbox.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await sandbox.exec(["ls"])

    async def test_concurrent_first_use_creates_one_sandbox(self) -> None:
        remote = _fake_remote()
        create = AsyncMock(side_effect=_slow_create(remote))
        with patch("ag2.extensions.e2b.sandbox.AsyncSandbox.create", create):
            sandbox = E2BSandbox()
            await asyncio.gather(*(sandbox.exec(["true"]) for _ in range(5)))
        create.assert_awaited_once()

    async def test_setup_failure_kills_the_new_sandbox_and_disarms_atexit(self) -> None:
        remote = _fake_remote()
        remote.files.make_dir = AsyncMock(side_effect=RuntimeError("read-only"))
        with _patch_create(remote), patch("ag2.extensions.e2b.sandbox.atexit") as exit_hooks:
            sandbox = E2BSandbox(workdir="/workspace")
            with pytest.raises(RuntimeError, match="read-only"):
                await sandbox.__aenter__()
        remote.kill.assert_awaited_once()
        assert sandbox.sandbox_id is None
        exit_hooks.unregister.assert_called_once_with(exit_hooks.register.call_args.args[0])

    async def test_failed_kill_is_not_raised_and_stays_retryable(self) -> None:
        remote = _fake_remote()
        remote.kill = AsyncMock(side_effect=RuntimeError("server unreachable"))
        with _patch_create(remote), patch("ag2.extensions.e2b.sandbox.atexit") as exit_hooks:
            sandbox = E2BSandbox()
            await sandbox.__aenter__()
            await sandbox.aclose()
            assert sandbox.closed
            # The sandbox is still alive, so the atexit fallback stays armed
            # and a second close retries instead of leaking it.
            exit_hooks.unregister.assert_not_called()
            await sandbox.aclose()
        assert remote.kill.await_count == 2

    async def test_successful_close_disarms_atexit(self) -> None:
        with _patch_create(_fake_remote()), patch("ag2.extensions.e2b.sandbox.atexit") as exit_hooks:
            sandbox = E2BSandbox()
            await sandbox.__aenter__()
            await sandbox.aclose()
        exit_hooks.register.assert_called_once()
        exit_hooks.unregister.assert_called_once_with(exit_hooks.register.call_args.args[0])

    async def test_atexit_fallback_kills_by_id_with_connection_options(self) -> None:
        remote = _fake_remote()
        with (
            _patch_create(remote),
            patch("ag2.extensions.e2b.sandbox.atexit") as exit_hooks,
            patch("ag2.extensions.e2b.sandbox.SyncSandbox.kill") as sync_kill,
        ):
            options = {"api_key": "test", "domain": "e2b.example", "secure": True}  # pragma: allowlist secret
            await E2BSandbox(create_options=options).__aenter__()
            [on_exit] = exit_hooks.register.call_args.args
            on_exit()
        # Only connection settings are forwarded; creation options are not.
        sync_kill.assert_called_once_with("sbx-1", api_key="test", domain="e2b.example")  # pragma: allowlist secret


def _first_call_waits(gate: asyncio.Event) -> Any:
    # The first call blocks until the test releases it; later calls succeed at once.
    calls = 0

    async def call(*args: Any, **kwargs: Any) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            await gate.wait()
        return True

    return call


def _gated(result: Any = None) -> tuple[asyncio.Event, Any]:
    # An SDK call that blocks until the test releases it, like a slow network round-trip.
    gate = asyncio.Event()

    async def call(*args: Any, **kwargs: Any) -> Any:
        await gate.wait()
        return result

    return gate, call


@pytest.mark.asyncio
class TestConcurrency:
    async def test_concurrent_refreshes_never_shorten_the_lifetime(self) -> None:
        # set_timeout can shorten a lifetime, so a 430 s refresh landing after a
        # 630 s one would cut the 600 s command short.
        remote = _fake_remote()
        gate, slow_set_timeout = _gated()
        remote.set_timeout = AsyncMock(side_effect=slow_set_timeout)
        with _patch_create(remote):
            sandbox = E2BSandbox(sandbox_timeout=300)
            await sandbox.__aenter__()
            long_run = asyncio.create_task(sandbox.exec(["make"], timeout=600))
            shorter_run = asyncio.create_task(sandbox.exec(["make"], timeout=400))
            await asyncio.sleep(0.01)
            gate.set()
            await asyncio.gather(long_run, shorter_run)
        remote.set_timeout.assert_awaited_once_with(630)

    async def test_concurrent_first_use_waits_for_the_workdir(self) -> None:
        remote = _fake_remote()
        gate, slow_make_dir = _gated(True)
        remote.files.make_dir = AsyncMock(side_effect=slow_make_dir)
        workdir_ready = []

        async def write(*args: Any, **kwargs: Any) -> None:
            workdir_ready.append(gate.is_set())

        remote.files.write = write
        with _patch_create(remote):
            sandbox = E2BSandbox(workdir="/workspace")
            first = asyncio.create_task(sandbox.put_file(PurePosixPath("a.txt"), b"a"))
            await asyncio.sleep(0.01)
            second = asyncio.create_task(sandbox.put_file(PurePosixPath("b.txt"), b"b"))
            await asyncio.sleep(0.01)
            gate.set()
            await asyncio.gather(first, second)
        assert workdir_ready == [True, True]

    async def test_close_during_creation_kills_the_new_sandbox(self) -> None:
        remote = _fake_remote()
        gate, slow_create = _gated(remote)
        with patch("ag2.extensions.e2b.sandbox.AsyncSandbox.create", AsyncMock(side_effect=slow_create)):
            sandbox = E2BSandbox()
            opening = asyncio.create_task(sandbox.__aenter__())
            await asyncio.sleep(0.01)
            closing = asyncio.create_task(sandbox.aclose())
            await asyncio.sleep(0.01)
            gate.set()
            await closing
            # The kill is done by the time `aclose` returns, not left to the creator.
            remote.kill.assert_awaited_once()
            with pytest.raises(RuntimeError, match="closed"):
                await opening
        remote.kill.assert_awaited_once()
        assert sandbox.sandbox_id is None

    async def test_late_cleanup_of_a_failed_setup_keeps_the_new_sandbox(self) -> None:
        # A cancelled caller's shielded kill of its failed sandbox finishes only
        # after a later call already replaced it; the new handle must survive.
        failed, fresh = _fake_remote(), _fake_remote()
        fresh.sandbox_id = "sbx-2"
        failed.files.make_dir = AsyncMock(side_effect=RuntimeError("read-only"))
        gate = asyncio.Event()
        failed.kill = AsyncMock(side_effect=_first_call_waits(gate))
        create = AsyncMock(side_effect=[failed, fresh])
        with patch("ag2.extensions.e2b.sandbox.AsyncSandbox.create", create):
            sandbox = E2BSandbox(workdir="/workspace")
            first = asyncio.create_task(sandbox.exec(["a"]))
            await asyncio.sleep(0.01)
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            await sandbox.exec(["b"])
            gate.set()
            await asyncio.sleep(0.01)
            assert sandbox.sandbox_id == "sbx-2"
            await sandbox.aclose()
        fresh.kill.assert_awaited_once()


@pytest.mark.asyncio
class TestFailureReporting:
    async def test_unverifiable_state_is_not_reported_as_gone(self) -> None:
        remote = _fake_remote()
        remote.commands.run = AsyncMock(side_effect=TimeoutException("Request timed out"))
        remote.is_running = AsyncMock(side_effect=ConnectionError("network blip"))
        with _patch_create(remote):
            actual = await E2BSandbox().exec(["sleep", "99"], timeout=5)
        assert actual == ExecResult(
            output="E2B execution timed out after 5s, and the sandbox state could not be checked: network blip",
            exit_code=1,
        )

    async def test_failed_kill_after_failed_setup_stays_retryable(self) -> None:
        remote = _fake_remote()
        remote.files.make_dir = AsyncMock(side_effect=RuntimeError("read-only"))
        remote.kill = AsyncMock(side_effect=[RuntimeError("server unreachable"), True])
        with _patch_create(remote), patch("ag2.extensions.e2b.sandbox.atexit") as exit_hooks:
            sandbox = E2BSandbox(workdir="/workspace")
            with pytest.raises(RuntimeError, match="read-only"):
                await sandbox.__aenter__()
            assert sandbox.sandbox_id == "sbx-1"
            exit_hooks.unregister.assert_not_called()
            await sandbox.aclose()
        assert remote.kill.await_count == 2
        assert sandbox.sandbox_id is None
        exit_hooks.unregister.assert_called_once()
