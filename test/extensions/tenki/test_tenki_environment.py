# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenki_sandbox import CommandResult

from ag2 import Context, Variable
from ag2.extensions.tenki import TenkiEnvironment, TenkiResources
from ag2.extensions.tenki.sandbox import TenkiSandbox
from ag2.tools import SandboxCodeTool, SandboxShellTool
from ag2.tools.sandbox import SandboxFactory


def _fake_remote() -> Any:
    return SimpleNamespace(
        id="sb-1",
        state="RUNNING",
        exec=AsyncMock(return_value=CommandResult(argv=["python3"], exit_code=0, stdout=b"ok\n")),
        fs=SimpleNamespace(
            write_bytes=AsyncMock(return_value=None),
            remove=AsyncMock(return_value=None),
        ),
        close_if_open=AsyncMock(return_value=None),
    )


def _fake_client(remote: Any, project_ids: tuple[str, ...] = ("project-1",)) -> Any:
    projects = tuple(SimpleNamespace(id=project_id) for project_id in project_ids)
    identity = SimpleNamespace(workspaces=(SimpleNamespace(projects=projects),))
    return SimpleNamespace(
        auth_token="test",  # pragma: allowlist secret
        base_url="https://api.tenki.cloud",
        create=AsyncMock(return_value=remote),
        who_am_i=AsyncMock(return_value=identity),
        close=AsyncMock(return_value=None),
    )


def _patch_async_client(client: Any) -> Any:
    return patch("ag2.extensions.tenki.environment.AsyncClient", return_value=client)


def test_satisfies_factory_protocol() -> None:
    factory: SandboxFactory = TenkiEnvironment(api_key="test")
    assert isinstance(factory, SandboxFactory)


def test_invalid_durations_rejected() -> None:
    with pytest.raises(ValueError, match="timeout"):
        TenkiEnvironment(timeout=0)
    with pytest.raises(ValueError, match="max_duration"):
        TenkiEnvironment(max_duration=0)


def test_declares_real_tenki_workdir_to_shell_tool() -> None:
    env = TenkiEnvironment(workdir="/home/tenki/project")
    assert env.workdir == PurePosixPath("/home/tenki/project")
    assert SandboxShellTool(env).workdir == PurePosixPath("/home/tenki/project")


@pytest.mark.asyncio
class TestOpen:
    async def test_open_yields_tenki_sandbox_with_bounded_lifecycle(self) -> None:
        remote = _fake_remote()
        client = _fake_client(remote)
        with _patch_async_client(client):
            factory = TenkiEnvironment(
                api_key="test",
                project_id="project-1",
                resources=TenkiResources(cpu_cores=2, memory_mb=4096, disk_size_gb=5),
                max_duration=600,
            )
            async with factory.open() as sandbox:
                assert isinstance(sandbox, TenkiSandbox)
            await factory.aclose()

        client.create.assert_awaited_once_with(
            wait=False,
            project_id="project-1",
            name="ag2",
            image=None,
            env={},
            cpu_cores=2,
            memory_mb=4096,
            disk_size_gb=5,
            allow_inbound=False,
            allow_outbound=True,
            max_duration=600,
            tags=["ag2"],
        )
        remote.close_if_open.assert_awaited_once()

    async def test_open_resolves_variables_from_context(self) -> None:
        remote = _fake_remote()
        client = _fake_client(remote)
        context = Context(
            stream=MagicMock(),
            variables={"tenki_key": "test", "tenki_project": "project-2"},
        )
        with _patch_async_client(client):
            factory = TenkiEnvironment(
                api_key=Variable("tenki_key"),
                project_id=Variable("tenki_project"),
            )
            async with factory.open(context):
                pass
            await factory.aclose()

        client.create.assert_awaited_once()
        assert client.create.await_args.kwargs["project_id"] == "project-2"

    async def test_missing_context_for_variable_raises(self) -> None:
        factory = TenkiEnvironment(api_key=Variable("tenki_key"))
        with pytest.raises(RuntimeError, match="Variable but no Context"):
            async with factory.open():
                pass

    async def test_multiple_projects_requires_explicit_project(self) -> None:
        client = _fake_client(_fake_remote(), project_ids=("project-1", "project-2"))
        with _patch_async_client(client):
            factory = TenkiEnvironment(api_key="test")
            with pytest.raises(RuntimeError, match="multiple projects"):
                async with factory.open():
                    pass

        client.close.assert_awaited_once()

    async def test_open_reuses_sandbox_until_environment_close(self) -> None:
        remote = _fake_remote()
        client = _fake_client(remote)
        with _patch_async_client(client):
            factory = TenkiEnvironment(api_key="test", project_id="project-1")
            async with factory.open() as first:
                pass
            async with factory.open() as second:
                assert second is first
            remote.close_if_open.assert_not_awaited()
            await factory.aclose()
            await factory.aclose()

        client.create.assert_awaited_once()
        remote.close_if_open.assert_awaited_once()

    async def test_code_tool_uses_python3_from_backend_runner(self) -> None:
        remote = _fake_remote()
        client = _fake_client(remote)
        with _patch_async_client(client):
            factory = TenkiEnvironment(api_key="test", project_id="project-1")
            tool = SandboxCodeTool(factory)
            result = await tool.environment.run("print('ok')", "python")
            await factory.aclose()

        assert result.output == "ok\n"
        assert remote.exec.await_args.args[:2] == ("python3", "-c")


class TestDeepcopy:
    def test_environment_deepcopy_returns_same_instance(self) -> None:
        env = TenkiEnvironment(api_key="test")
        assert deepcopy(env) is env

    def test_sandbox_deepcopy_returns_same_instance(self) -> None:
        sandbox = TenkiSandbox(client=_fake_client(_fake_remote()), create_options={})
        assert deepcopy(sandbox) is sandbox

    def test_tool_backed_by_environment_is_deepcopyable(self) -> None:
        tool = SandboxShellTool(TenkiEnvironment(api_key="test"))
        assert isinstance(deepcopy(tool), SandboxShellTool)
