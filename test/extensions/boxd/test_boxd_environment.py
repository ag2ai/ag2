# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Offline unit tests for BoxdEnvironment; the boxd SDK is fully mocked."""

from copy import deepcopy
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from boxd import AuthenticationError
from boxd import ExecResult as BoxdExecResult

from ag2 import Context, Variable
from ag2.extensions.boxd import BoxdEnvironment, BoxdResources
from ag2.extensions.boxd.sandbox import BoxdSandbox
from ag2.tools import SandboxCodeTool, SandboxShellTool
from ag2.tools.sandbox import SandboxFactory, WorkdirAware


def _fake_client() -> Any:
    return SimpleNamespace(
        base_url="https://boxd.sh:9443",
        region=None,
        machines=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(id="m-1", name="ag2-test")),
            wait_until_ready=AsyncMock(return_value=None),
            exec=AsyncMock(return_value=BoxdExecResult(stdout="ok\n", stderr="", exit_code=0)),
            delete=AsyncMock(return_value=None),
            files=SimpleNamespace(upload=AsyncMock(return_value=0)),
        ),
        close=AsyncMock(return_value=None),
    )


def _patch_async_boxd(client: Any) -> Any:
    return patch("ag2.extensions.boxd.environment.AsyncBoxd", return_value=client)


def test_satisfies_factory_protocol() -> None:
    factory: SandboxFactory = BoxdEnvironment(api_key="test")
    assert isinstance(factory, SandboxFactory)


def test_invalid_durations_rejected() -> None:
    with pytest.raises(ValueError, match="timeout"):
        BoxdEnvironment(timeout=0)
    with pytest.raises(ValueError, match="delete_after"):
        BoxdEnvironment(delete_after=0)


def test_declares_its_workdir_to_the_shell_tool() -> None:
    # A boxd machine runs as user `boxd` out of /home/boxd, not /workspace.
    env = BoxdEnvironment(api_key="test", workdir="/home/boxd/project")

    assert isinstance(env, WorkdirAware)
    assert env.workdir == PurePosixPath("/home/boxd/project")
    assert SandboxShellTool(env).workdir == PurePosixPath("/home/boxd/project")


def test_default_workdir_is_the_boxd_home() -> None:
    assert SandboxShellTool(BoxdEnvironment(api_key="test")).workdir == PurePosixPath("/home/boxd")


@pytest.mark.asyncio
class TestOpen:
    async def test_open_yields_boxd_sandbox_with_bounded_lifecycle(self) -> None:
        client = _fake_client()
        with _patch_async_boxd(client) as boxd_cls:
            factory = BoxdEnvironment(
                api_key="test",
                region="fra",
                name="my-sandbox",
                labels={"team": "data"},
                resources=BoxdResources(vcpu=4, memory="16G", disk="50G"),
                delete_after=600,
            )
            async with factory.open() as sandbox:
                assert isinstance(sandbox, BoxdSandbox)
            await factory.aclose()

        boxd_cls.assert_called_once_with(api_key="test", base_url=None, region="fra")
        client.machines.create.assert_awaited_once_with(
            name="my-sandbox",
            image=None,
            labels={"created-by": "ag2", "team": "data"},
            vcpu=4,
            memory="16G",
            disk="50G",
            isolated=True,
            delete_after=600,
        )
        client.machines.delete.assert_awaited_once_with("m-1")

    async def test_defaults_let_boxd_pick_name_and_size(self) -> None:
        client = _fake_client()
        with _patch_async_boxd(client):
            factory = BoxdEnvironment()
            async with factory.open():
                pass
            await factory.aclose()

        client.machines.create.assert_awaited_once_with(
            name=None,
            image=None,
            labels={"created-by": "ag2"},
            vcpu=None,
            memory=None,
            disk=None,
            isolated=True,
            delete_after=3600,
        )

    async def test_env_vars_reach_every_command_under_per_call_env(self) -> None:
        # `env_vars` are sent with every command, under the per-call `env`.
        client = _fake_client()
        with _patch_async_boxd(client):
            factory = BoxdEnvironment(env_vars={"APP_ENV": "sandbox", "MODE": "a"})
            async with factory.open() as sandbox:
                await sandbox.exec(["env"], env={"MODE": "b"})
            await factory.aclose()

        assert "env" not in client.machines.create.await_args.kwargs
        assert client.machines.exec.await_args.kwargs["env"] == {"APP_ENV": "sandbox", "MODE": "b"}

    async def test_open_resolves_variables_from_context(self) -> None:
        client = _fake_client()
        context = Context(
            stream=MagicMock(),
            variables={"boxd_key": "test", "tenant": {"tenant": "acme"}},
        )
        with _patch_async_boxd(client) as boxd_cls:
            factory = BoxdEnvironment(api_key=Variable("boxd_key"), labels=Variable("tenant"))
            async with factory.open(context):
                pass
            await factory.aclose()

        boxd_cls.assert_called_once_with(api_key="test", base_url=None, region=None)
        assert client.machines.create.await_args.kwargs["labels"] == {"created-by": "ag2", "tenant": "acme"}

    async def test_open_missing_variable_raises_key_error(self) -> None:
        context = Context(stream=MagicMock(), variables={})
        factory = BoxdEnvironment(api_key="test", image=Variable("tenant_image"))
        with pytest.raises(KeyError, match="tenant_image"):
            async with factory.open(context):
                pass

    async def test_missing_context_for_variable_raises(self) -> None:
        factory = BoxdEnvironment(api_key=Variable("boxd_key"))
        with pytest.raises(RuntimeError, match="Variable but no Context"):
            async with factory.open():
                pass

    async def test_create_error_propagates_and_releases_the_client(self) -> None:
        client = _fake_client()
        client.machines.create = AsyncMock(
            side_effect=AuthenticationError("API key not recognized", code="unauthenticated", reason="invalid_token")
        )
        with _patch_async_boxd(client):
            factory = BoxdEnvironment(api_key="wrong")
            with pytest.raises(AuthenticationError, match="not recognized"):
                async with factory.open():
                    pass

        client.close.assert_awaited_once()
        client.machines.delete.assert_not_awaited()

    async def test_open_reuses_sandbox_until_environment_close(self) -> None:
        client = _fake_client()
        with _patch_async_boxd(client):
            factory = BoxdEnvironment(api_key="test")
            async with factory.open() as first:
                pass
            async with factory.open() as second:
                assert second is first
            client.machines.delete.assert_not_awaited()
            await factory.aclose()
            await factory.aclose()

        client.machines.create.assert_awaited_once()
        client.machines.delete.assert_awaited_once()

    async def test_code_tool_runs_python(self) -> None:
        client = _fake_client()
        with _patch_async_boxd(client):
            factory = BoxdEnvironment(api_key="test")
            tool = SandboxCodeTool(factory)
            result = await tool.environment.run("print('ok')", "python")
            await factory.aclose()

        assert result.output == "ok"
        assert client.machines.exec.await_args.args == ("m-1", ["python", "-c", "print('ok')"])


class TestDeepcopy:
    def test_environment_deepcopy_returns_same_instance(self) -> None:
        env = BoxdEnvironment(api_key="test")
        assert deepcopy(env) is env

    def test_sandbox_deepcopy_returns_same_instance(self) -> None:
        sandbox = BoxdSandbox(client=_fake_client(), create_options={})
        assert deepcopy(sandbox) is sandbox

    def test_tool_backed_by_environment_is_deepcopyable(self) -> None:
        tool = SandboxShellTool(BoxdEnvironment(api_key="test"))
        assert isinstance(deepcopy(tool), SandboxShellTool)
