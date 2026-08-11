# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import io
import tarfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ag2.annotations import Variable
from ag2.extensions.docker.sandbox import DockerSandbox
from ag2.tools.sandbox import ExecResult


def _exec_result(output: bytes = b"ok\n", exit_code: int = 0) -> SimpleNamespace:
    return SimpleNamespace(output=output, exit_code=exit_code)


def _fake_container(exec_result: SimpleNamespace | None = None, container_id: str = "deadbeef") -> Any:
    return SimpleNamespace(
        short_id=container_id,
        exec_run=MagicMock(return_value=exec_result or _exec_result()),
        stop=MagicMock(return_value=None),
        remove=MagicMock(return_value=None),
        put_archive=MagicMock(return_value=True),
        get_archive=MagicMock(return_value=(iter([b""]), {})),
    )


def _fake_client(container: Any) -> Any:
    return SimpleNamespace(
        containers=SimpleNamespace(run=MagicMock(return_value=container)),
        close=MagicMock(return_value=None),
    )


def _patch_docker(container: Any) -> Any:
    return patch(
        "ag2.extensions.docker.sandbox.docker.from_env",
        return_value=_fake_client(container),
    )


class TestConstruction:
    def test_invalid_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            DockerSandbox(timeout=0)

    def test_workdir_is_posix_path_inside_container(self) -> None:
        sandbox = DockerSandbox(workdir="/sandbox")
        assert sandbox.workdir == PurePosixPath("/sandbox")

    def test_host_workdir_none_without_host_path(self) -> None:
        sandbox = DockerSandbox()
        assert sandbox.host_workdir is None

    def test_host_workdir_is_set_when_host_path_given(self, tmp_path: Path) -> None:
        sandbox = DockerSandbox(host_path=tmp_path)
        assert sandbox.host_workdir == tmp_path.resolve()


@pytest.mark.asyncio
class TestExec:
    async def test_argv_form_runs_command(self) -> None:
        container = _fake_container(_exec_result(b"42\n"))
        with _patch_docker(container):
            sandbox = DockerSandbox()
            result = await sandbox.exec(["python", "-c", "print(40+2)"])

        assert result == ExecResult(output="42", exit_code=0)
        assert container.exec_run.call_args.args[0] == ["python", "-c", "print(40+2)"]

    async def test_explicit_shell_argv_is_passed_through(self) -> None:
        container = _fake_container(_exec_result(b"hello\n"))
        with _patch_docker(container):
            sandbox = DockerSandbox()
            result = await sandbox.exec(["sh", "-c", "echo hello"])

        assert result == ExecResult(output="hello", exit_code=0)
        assert container.exec_run.call_args.args[0] == ["sh", "-c", "echo hello"]

    async def test_env_is_forwarded_to_exec_run(self) -> None:
        container = _fake_container(_exec_result(b"bar\n"))
        with _patch_docker(container):
            sandbox = DockerSandbox()
            await sandbox.exec(["env"], env={"FOO": "bar"})

        assert container.exec_run.call_args.kwargs["environment"] == {"FOO": "bar"}

    async def test_container_created_only_once(self) -> None:
        container = _fake_container()
        client = _fake_client(container)
        with patch("ag2.extensions.docker.sandbox.docker.from_env", return_value=client):
            sandbox = DockerSandbox()
            await sandbox.exec(["echo", "1"])
            await sandbox.exec(["echo", "2"])

        assert client.containers.run.call_count == 1

    async def test_host_path_is_bind_mounted(self, tmp_path: Path) -> None:
        container = _fake_container()
        client = _fake_client(container)
        with patch("ag2.extensions.docker.sandbox.docker.from_env", return_value=client):
            sandbox = DockerSandbox(host_path=tmp_path)
            await sandbox.exec(["pwd"])

        assert client.containers.run.call_args.kwargs["volumes"] == {
            str(tmp_path.resolve()): {"bind": "/workspace", "mode": "rw"}
        }


@pytest.mark.asyncio
class TestLifecycle:
    async def test_aenter_starts_container(self) -> None:
        container = _fake_container()
        client = _fake_client(container)
        with patch("ag2.extensions.docker.sandbox.docker.from_env", return_value=client):
            async with DockerSandbox() as _sandbox:
                pass

        client.containers.run.assert_called_once()
        container.stop.assert_called_once()

    async def test_aclose_stops_container_and_closes_client(self) -> None:
        container = _fake_container()
        client = _fake_client(container)
        with patch("ag2.extensions.docker.sandbox.docker.from_env", return_value=client):
            sandbox = DockerSandbox()
            await sandbox.exec(["echo", "1"])
            await sandbox.aclose()

        container.stop.assert_called_once()
        client.close.assert_called_once()

    async def test_aclose_without_exec_is_safe(self) -> None:
        sandbox = DockerSandbox()
        await sandbox.aclose()


@pytest.mark.asyncio
class TestFileIO:
    async def test_put_file_calls_put_archive(self) -> None:
        container = _fake_container()
        with _patch_docker(container):
            sandbox = DockerSandbox(workdir="/sandbox")
            await sandbox.put_file(PurePosixPath("hello.txt"), b"world")

        call = container.put_archive.call_args
        assert call.args[0] == "/sandbox"
        archive = call.args[1]
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r") as tf:
            assert tf.getnames() == ["hello.txt"]
            f = tf.extractfile("hello.txt")
            assert f is not None and f.read() == b"world"

    async def test_absolute_path_rejected(self) -> None:
        sandbox = DockerSandbox()
        with pytest.raises(ValueError, match="Absolute"):
            await sandbox.put_file(PurePosixPath("/etc/passwd"), b"x")


def test_variable_image_rejected_by_constructor() -> None:
    with pytest.raises(TypeError):
        DockerSandbox(image=Variable("tenant_image"))  # type: ignore[arg-type]
