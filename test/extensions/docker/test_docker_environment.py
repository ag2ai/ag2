# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ag2 import Context, Variable
from ag2.extensions.docker import DockerEnvironment
from ag2.extensions.docker.sandbox import DockerSandbox
from ag2.tools import SandboxShellTool
from ag2.tools.sandbox import SandboxFactory


def _exec_result(output: bytes = b"ok\n", exit_code: int = 0) -> SimpleNamespace:
    return SimpleNamespace(output=output, exit_code=exit_code)


def _fake_container(exec_result: SimpleNamespace | None = None) -> Any:
    return SimpleNamespace(
        short_id="deadbeef",
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


def test_satisfies_factory_protocol() -> None:
    factory: SandboxFactory = DockerEnvironment()
    assert isinstance(factory, SandboxFactory)


@pytest.mark.asyncio
class TestOpen:
    async def test_open_yields_docker_sandbox(self) -> None:
        container = _fake_container()
        client = _fake_client(container)
        with patch("ag2.extensions.docker.sandbox.docker.from_env", return_value=client):
            factory = DockerEnvironment(image="python:3.12-slim")
            async with factory.open() as sandbox:
                assert isinstance(sandbox, DockerSandbox)

    async def test_open_resolves_image_variable_from_context(self) -> None:
        container = _fake_container()
        client = _fake_client(container)
        ctx = Context(stream=MagicMock(), variables={"tenant_image": "python:3.11-slim"})
        with patch("ag2.extensions.docker.sandbox.docker.from_env", return_value=client):
            factory = DockerEnvironment(image=Variable("tenant_image"))
            async with factory.open(ctx) as sandbox:
                await sandbox.exec(["python", "--version"])

        assert client.containers.run.call_args.args[0] == "python:3.11-slim"

    async def test_open_concrete_values_work_without_context(self) -> None:
        container = _fake_container()
        client = _fake_client(container)
        with patch("ag2.extensions.docker.sandbox.docker.from_env", return_value=client):
            factory = DockerEnvironment(image="python:3.12-slim")
            async with factory.open() as sandbox:
                await sandbox.exec(["python", "--version"])

        assert client.containers.run.call_args.args[0] == "python:3.12-slim"

    async def test_open_missing_variable_raises_key_error(self) -> None:
        ctx = Context(stream=MagicMock(), variables={})
        factory = DockerEnvironment(image=Variable("tenant_image"))

        with pytest.raises(KeyError, match="tenant_image"):
            async with factory.open(ctx):
                pass

    async def test_open_variable_without_context_raises(self) -> None:
        factory = DockerEnvironment(image=Variable("tenant_image"))

        with pytest.raises(RuntimeError, match="Variable but no Context"):
            async with factory.open():
                pass

    async def test_open_keeps_container_alive_for_reuse(self) -> None:
        # Caching: the container survives scope exit so the next open() reuses
        # it (state persists). Only aclose() tears it down.
        container = _fake_container()
        client = _fake_client(container)
        with patch("ag2.extensions.docker.sandbox.docker.from_env", return_value=client):
            factory = DockerEnvironment()
            async with factory.open() as sandbox:
                await sandbox.exec(["echo", "hi"])
            container.stop.assert_not_called()

            async with factory.open() as sandbox2:
                assert sandbox2 is sandbox
            # one container created across both opens
            assert client.containers.run.call_count == 1

            await factory.aclose()
        container.stop.assert_called_once()

    async def test_aclose_tears_down_cached_container(self) -> None:
        container = _fake_container()
        client = _fake_client(container)
        with patch("ag2.extensions.docker.sandbox.docker.from_env", return_value=client):
            factory = DockerEnvironment()
            async with factory.open() as sandbox:
                await sandbox.exec(["echo", "hi"])
            await factory.aclose()
            await factory.aclose()  # idempotent

        container.stop.assert_called_once()


class TestDeepcopy:
    """Regression for finding #4: the threading.Lock held by the environment /
    sandbox makes them un-deepcopy-able, which broke Agent.add_tool (it
    deepcopies every tool). __deepcopy__ returns self (shared handle)."""

    def test_environment_deepcopy_returns_same_instance(self) -> None:
        env = DockerEnvironment(image="python:3.12-slim")
        assert deepcopy(env) is env

    def test_sandbox_deepcopy_returns_same_instance(self) -> None:
        sandbox = DockerSandbox(image="python:3.12-slim")
        assert deepcopy(sandbox) is sandbox

    def test_tool_backed_by_environment_is_deepcopyable(self) -> None:
        # Exactly what Agent.add_tool -> FunctionTool.ensure_tool does.
        tool = SandboxShellTool(DockerEnvironment(image="python:3.12-slim"))
        assert isinstance(deepcopy(tool), SandboxShellTool)
