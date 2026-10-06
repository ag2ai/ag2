# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import sys
from copy import deepcopy
from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock

import pytest
from sprites import AsyncSprite, AsyncSpritesClient
from sprites.async_filesystem import AsyncSpriteFilesystem, AsyncSpritePath
from sprites.exceptions import APIError, NetworkError
from sprites.exceptions import TimeoutError as SpriteTimeoutError
from sprites.exec import CompletedProcess

from ag2 import Agent, TextInput
from ag2.events import ToolCallEvent, ToolResultsEvent
from ag2.extensions.sprites import SpritesEnvironment, SpritesSandbox
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools import SandboxCodeTool, SandboxShellTool
from ag2.tools.sandbox import CodeAdapter, ExecResult, LanguageRunner, Sandbox, SandboxFactory, WorkdirAware
from ag2.tools.sandbox.adapter import ShellAdapter


@pytest.fixture
def sprite() -> MagicMock:
    remote = MagicMock(spec=AsyncSprite)
    remote.client = MagicMock(spec=AsyncSpritesClient)
    remote.run.return_value = CompletedProcess(args=["pwd"], returncode=0, stdout=b"ok\n", stderr=b"")
    remote.filesystem.return_value = MagicMock(spec=AsyncSpriteFilesystem)
    remote.filesystem.return_value.path.return_value = MagicMock(spec=AsyncSpritePath)
    return remote


def test_protocols_and_caller_owned_lifecycle(sprite: MagicMock) -> None:
    env = SpritesEnvironment(sprite, workdir="/home/sprite/project")
    assert isinstance(env, SandboxFactory)
    assert isinstance(env, WorkdirAware)
    assert isinstance(env.sandbox, Sandbox)
    assert env.workdir == PurePosixPath("/home/sprite/project")
    assert env.sandbox.host_workdir is None
    assert SandboxShellTool(env).workdir == env.workdir
    assert deepcopy(env) is env
    assert deepcopy(env.sandbox) is env.sandbox
    assert isinstance(deepcopy(SandboxCodeTool(env)), SandboxCodeTool)
    sprite.run.assert_not_called()
    sprite.filesystem.assert_not_called()


@pytest.mark.parametrize("timeout", [None, "60", 0, -1, float("inf"), float("nan")])
def test_rejects_invalid_timeout(sprite: MagicMock, timeout: object) -> None:
    with pytest.raises(ValueError, match="timeout"):
        SpritesEnvironment(sprite, timeout=timeout)


@pytest.mark.parametrize("limit", [0, -1, 0.5, None])
def test_rejects_invalid_output_limit(sprite: MagicMock, limit: object) -> None:
    with pytest.raises(ValueError, match="max_output"):
        SpritesEnvironment(sprite, max_output=limit)


def test_rejects_relative_workdir(sprite: MagicMock) -> None:
    with pytest.raises(ValueError, match="workdir"):
        SpritesEnvironment(sprite, workdir="project")


@pytest.mark.asyncio
async def test_output_and_env_and_per_call_timeout(sprite: MagicMock) -> None:
    sprite.run.return_value = CompletedProcess(args=["test"], returncode=7, stdout=b"start\n", stderr=b"error\xff")
    env_vars = {"BASE": "one", "OVERRIDE": "old"}
    sandbox = SpritesSandbox(sprite, env_vars=env_vars, max_output=6)
    env_vars["BASE"] = "changed"
    result = await sandbox.exec(["test"], env={"OVERRIDE": "new"}, timeout=0.5)
    assert result == ExecResult(output="error\ufffd", exit_code=7)
    assert sprite.run.await_args.kwargs["env"] == {"BASE": "one", "OVERRIDE": "new"}
    assert sprite.run.await_args.kwargs["cwd"] == "/home/sprite"
    assert sprite.run.await_args.kwargs["timeout"] > 0.5
    assert sprite.run.await_args.kwargs["check"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        SpriteTimeoutError("private diagnostic"),
        asyncio.TimeoutError(),
        NetworkError("private diagnostic"),
        APIError("private diagnostic", status_code=429),
    ],
)
async def test_transport_failure_never_claims_remote_termination(sprite: MagicMock, error: Exception) -> None:
    sprite.run.side_effect = error
    with pytest.raises(RuntimeError, match="execution status is unknown") as raised:
        await SpritesSandbox(sprite).exec(["sleep", "30"])
    assert "private diagnostic" not in str(raised.value)
    if isinstance(error, APIError):
        assert "429" in str(raised.value)
    sprite.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_context_and_close_leave_sprite_and_client_usable(sprite: MagicMock) -> None:
    env = SpritesEnvironment(sprite)
    async with env:
        async with env.open() as first:
            await first.exec(["pwd"])
        async with env.open() as second:
            assert second is first
    await env.aclose()
    await second.exec(["pwd"])
    sprite.delete.assert_not_called()
    sprite.client.aclose.assert_not_called()
    sprite.client.create_sprite.assert_not_called()
    assert sprite.run.await_count == 2


@pytest.mark.asyncio
async def test_empty_argv_never_calls_sdk(sprite: MagicMock) -> None:
    assert await SpritesSandbox(sprite).exec([]) == ExecResult(output="", exit_code=2)
    sprite.run.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/etc/passwd", "../outside", "nested/../../outside", "."])
async def test_file_paths_reject_escaping_and_directory_targets(sprite: MagicMock, path: str) -> None:
    sandbox = SpritesSandbox(sprite)
    with pytest.raises(ValueError):
        await sandbox.put_file(PurePosixPath(path), b"content")
    with pytest.raises(ValueError):
        await sandbox.remove_file(PurePosixPath(path))
    sprite.filesystem.assert_not_called()


@pytest.mark.asyncio
async def test_file_io_uses_sdk_and_missing_file_cleanup_is_idempotent(sprite: MagicMock) -> None:
    sandbox = SpritesSandbox(sprite, workdir="/home/sprite/project")
    await sandbox.put_file(PurePosixPath("script.js"), b"console.log(42)")
    await sandbox.remove_file(PurePosixPath("script.js"))
    sprite.filesystem.assert_called_with("/home/sprite/project")
    path = sprite.filesystem.return_value.path
    path.assert_called_with("script.js")
    path.return_value.write_bytes.assert_awaited_once_with(b"console.log(42)")
    path.return_value.unlink.assert_awaited_once_with(missing_ok=True)


@pytest.mark.asyncio
async def test_file_mode_adapter_cleans_only_its_script(sprite: MagicMock) -> None:
    adapter = CodeAdapter(SpritesEnvironment(sprite), languages=("javascript",))
    assert (await adapter.run("console.log(42)", "javascript")).exit_code == 0
    remote_path = sprite.filesystem.return_value.path
    names = [call.args[0] for call in remote_path.call_args_list]
    assert len(names) == 2 and names[0] == names[1]
    assert names[0].startswith("ag2_") and names[0].endswith(".js")
    remote_path.return_value.unlink.assert_awaited_once_with(missing_ok=True)


class LocalTransport:
    """Execute SDK argv in a temporary directory to exercise the remote watchdog offline."""

    def __init__(self, path: Path) -> None:
        self.path = path

    async def run(self, *args: str, **kwargs: object) -> CompletedProcess:
        argv = [sys.executable if args[0] == "python" else args[0], *args[1:]]
        env = {**os.environ, **(kwargs.get("env") or {})}
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.path,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
        return CompletedProcess(args=list(args), returncode=process.returncode, stdout=stdout, stderr=stderr)


@pytest.fixture
def local_sprite(sprite: MagicMock, tmp_path: Path) -> MagicMock:
    if sys.platform == "win32":
        pytest.skip("Sprites run POSIX processes; local watchdog tests require POSIX")
    sprite.run.side_effect = LocalTransport(tmp_path).run
    return sprite


@pytest.mark.asyncio
async def test_real_execution_preserves_arguments_and_nonzero_exit(local_sprite: MagicMock) -> None:
    sandbox = SpritesSandbox(local_sprite)
    literal = "$(printf unexpected); 'quoted' value"
    result = await sandbox.exec([sys.executable, "-c", "import sys; print(sys.argv[1]); sys.exit(7)", literal])
    assert result == ExecResult(output=literal + "\n", exit_code=7)
    missing = await sandbox.exec(["ag2-command-that-does-not-exist"])
    assert missing.exit_code == 127


@pytest.mark.asyncio
async def test_remote_deadline_prevents_delayed_side_effect(local_sprite: MagicMock, tmp_path: Path) -> None:
    sandbox = SpritesSandbox(local_sprite, timeout=0.1)
    code = "import time; from pathlib import Path; time.sleep(0.5); Path('too-late').write_text('oops')"
    result = await sandbox.exec([sys.executable, "-c", code])
    assert result.exit_code == 124
    await asyncio.sleep(0.6)
    assert not (tmp_path / "too-late").exists()


@pytest.mark.asyncio
async def test_remote_deadline_kills_children_that_ignore_sigterm(local_sprite: MagicMock, tmp_path: Path) -> None:
    sandbox = SpritesSandbox(local_sprite, timeout=0.2)
    child = "import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(0.6); Path('child-survived').touch()"
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(5)"
    result = await sandbox.exec([sys.executable, "-c", parent])
    assert result.exit_code == 124
    await asyncio.sleep(0.7)
    assert not (tmp_path / "child-survived").exists()


@pytest.mark.asyncio
async def test_two_tools_share_files_through_agent(local_sprite: MagicMock) -> None:
    env = SpritesEnvironment(local_sprite)
    config = TrackingConfig(
        TestConfig(
            ToolCallEvent(
                name="run_code",
                arguments=json.dumps({
                    "code": "from pathlib import Path; Path('shared.txt').write_text('persistent')",
                    "language": "python",
                }),
            ),
            ToolCallEvent(name="run_shell_command", arguments=json.dumps({"command": "cat shared.txt"})),
            "done",
        )
    )
    agent = Agent(
        "developer",
        config=config,
        tools=[
            SandboxCodeTool(env, runners={"python": LanguageRunner(inline_argv=(sys.executable, "-c"))}),
            SandboxShellTool(env),
        ],
    )
    reply = await agent.ask("Write a file with Python, then read it with the shell.")
    assert await reply.content() == "done"
    results: ToolResultsEvent = config.mock.call_args_list[2].args[0]
    assert results.results[0].result.parts == [TextInput("persistent")]
    assert local_sprite.run.await_count == 2
    assert "persistent" in await ShellAdapter(env).run("cat shared.txt")


@pytest.mark.asyncio
async def test_cancellation_propagates_without_retry_or_deleting_sprite(sprite: MagicMock) -> None:
    sprite.run.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await SpritesSandbox(sprite).exec(["sleep", "30"])
    sprite.run.assert_awaited_once()
    sprite.delete.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
async def test_invalid_per_call_timeout_never_starts_a_command(sprite: MagicMock, timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        await SpritesSandbox(sprite).exec(["pwd"], timeout=timeout)
    sprite.run.assert_not_called()


@pytest.mark.asyncio
async def test_file_errors_do_not_expose_sdk_diagnostics(sprite: MagicMock) -> None:
    path = sprite.filesystem.return_value.path.return_value
    path.write_bytes.side_effect = NetworkError("private diagnostic")
    path.unlink.side_effect = NetworkError("private diagnostic")
    sandbox = SpritesSandbox(sprite)
    with pytest.raises(RuntimeError, match="file write status is unknown") as write_error:
        await sandbox.put_file(PurePosixPath("script.js"), b"content")
    with pytest.raises(RuntimeError, match="file removal status is unknown") as remove_error:
        await sandbox.remove_file(PurePosixPath("script.js"))
    assert "private diagnostic" not in str(write_error.value)
    assert "private diagnostic" not in str(remove_error.value)
    path.write_bytes.assert_awaited_once()
    path.unlink.assert_awaited_once()


@pytest.mark.asyncio
async def test_per_call_deadline_overrides_default(local_sprite: MagicMock) -> None:
    sandbox = SpritesSandbox(local_sprite, timeout=0.001)
    result = await sandbox.exec([sys.executable, "-c", "import time; time.sleep(0.05); print('finished')"], timeout=5)
    assert result == ExecResult(output="finished\n", exit_code=0)
