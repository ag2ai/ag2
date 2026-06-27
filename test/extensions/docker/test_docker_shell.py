# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ag2 import Context, Variable
from ag2.extensions.docker import DockerEnvironment
from ag2.tools.sandbox.adapter import ShellAdapter


def _exec_result(output: bytes = b"ok\n", exit_code: int = 0) -> SimpleNamespace:
    return SimpleNamespace(output=output, exit_code=exit_code)


def _fake_container(exec_result: SimpleNamespace | None = None, container_id: str = "deadbeef") -> Any:
    return SimpleNamespace(
        short_id=container_id,
        exec_run=MagicMock(return_value=exec_result or _exec_result()),
        stop=MagicMock(return_value=None),
        remove=MagicMock(return_value=None),
    )


def _patch_docker(container: Any) -> Any:
    client = SimpleNamespace(
        containers=SimpleNamespace(run=MagicMock(return_value=container)),
        close=MagicMock(return_value=None),
    )
    return patch("ag2.extensions.docker.sandbox.docker.from_env", return_value=client)


@pytest.mark.asyncio
class TestShellAdapterOverDocker:
    async def test_run_executes_shell_command(self) -> None:
        container = _fake_container(_exec_result(b"hello\n"))
        with _patch_docker(container):
            shell = ShellAdapter(DockerEnvironment())
            result = await shell.run("echo hello")

        assert result == "hello"
        assert container.exec_run.call_args.args[0] == ["sh", "-c", "echo hello"]

    async def test_allowed_blocks_non_matching_command(self) -> None:
        container = _fake_container()
        with _patch_docker(container):
            shell = ShellAdapter(DockerEnvironment(), allowed=["echo"])
            result = await shell.run("touch file.txt")

        assert result == "Command not allowed: 'touch file.txt'"
        container.exec_run.assert_not_called()

    async def test_blocked_rejects_matching_command(self) -> None:
        container = _fake_container()
        with _patch_docker(container):
            shell = ShellAdapter(DockerEnvironment(), blocked=["rm -rf"])
            result = await shell.run("rm -rf /workspace")

        assert result == "Command not allowed: 'rm -rf /workspace'"
        container.exec_run.assert_not_called()

    async def test_nonzero_exit_code_is_included(self) -> None:
        container = _fake_container(_exec_result(b"nope\n", exit_code=7))
        with _patch_docker(container):
            shell = ShellAdapter(DockerEnvironment())
            result = await shell.run("exit 7")

        assert "nope" in result
        assert "exit code: 7" in result


@pytest.mark.asyncio
class TestVariableResolution:
    async def test_image_resolved_from_context(self) -> None:
        container = _fake_container(_exec_result(b"ok\n"))
        client = SimpleNamespace(
            containers=SimpleNamespace(run=MagicMock(return_value=container)),
            close=MagicMock(return_value=None),
        )
        ctx = Context(stream=MagicMock(), variables={"tenant_image": "python:3.11-slim"})
        with patch("ag2.extensions.docker.sandbox.docker.from_env", return_value=client):
            shell = ShellAdapter(DockerEnvironment(image=Variable("tenant_image")))
            await shell.run("echo hi", context=ctx)

        assert client.containers.run.call_args.args[0] == "python:3.11-slim"

    async def test_missing_variable_raises_key_error(self) -> None:
        ctx = Context(stream=MagicMock(), variables={})
        shell = ShellAdapter(DockerEnvironment(image=Variable("tenant_image")))

        with pytest.raises(KeyError, match="tenant_image"):
            await shell.run("echo hi", context=ctx)
