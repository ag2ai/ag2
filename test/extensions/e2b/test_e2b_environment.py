# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Offline unit tests for E2BEnvironment; the E2B SDK is fully mocked."""

import asyncio
from copy import deepcopy
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ag2 import Context, Variable
from ag2.extensions.e2b import E2BEnvironment
from ag2.extensions.e2b.sandbox import E2BSandbox
from ag2.tools import SandboxCodeTool, SandboxShellTool
from ag2.tools.sandbox import SandboxFactory, WorkdirAware


def _fake_remote(sandbox_id: str = "sbx-1") -> Any:
    return SimpleNamespace(
        sandbox_id=sandbox_id,
        commands=SimpleNamespace(
            run=AsyncMock(return_value=SimpleNamespace(stdout="ok\n", stderr="", exit_code=0)),
        ),
        files=SimpleNamespace(
            write=AsyncMock(return_value=None),
            remove=AsyncMock(return_value=None),
            make_dir=AsyncMock(return_value=False),
        ),
        set_timeout=AsyncMock(return_value=None),
        is_running=AsyncMock(return_value=True),
        kill=AsyncMock(return_value=True),
    )


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


def _patch_create(*remotes: Any) -> Any:
    return patch("ag2.extensions.e2b.sandbox.AsyncSandbox.create", AsyncMock(side_effect=list(remotes)))


def test_satisfies_factory_protocol() -> None:
    factory: SandboxFactory = E2BEnvironment()
    assert isinstance(factory, SandboxFactory)


def test_invalid_arguments_rejected() -> None:
    with pytest.raises(ValueError, match="timeout"):
        E2BEnvironment(timeout=0)
    with pytest.raises(ValueError, match="sandbox_timeout"):
        E2BEnvironment(sandbox_timeout=0)
    with pytest.raises(ValueError, match="api_key"):
        E2BEnvironment(create_options={"api_key": "test"})  # pragma: allowlist secret


def test_declares_its_workdir_to_the_shell_tool() -> None:
    # E2B runs commands as `user` out of /home/user, not the /workspace the
    # shell tool otherwise assumes.
    env = E2BEnvironment()

    assert isinstance(env, WorkdirAware)
    assert env.workdir == PurePosixPath("/home/user")
    assert SandboxShellTool(env).workdir == PurePosixPath("/home/user")


@pytest.mark.asyncio
class TestOpen:
    async def test_open_yields_e2b_sandbox_with_bounded_lifecycle(self) -> None:
        remote = _fake_remote()
        with _patch_create(remote) as create:
            factory = E2BEnvironment(
                api_key="test",  # pragma: allowlist secret
                template="base",
                env_vars={"A": "1"},
                sandbox_timeout=600,
                create_options={"allow_internet_access": False},
            )
            async with factory.open() as sandbox:
                assert isinstance(sandbox, E2BSandbox)
            remote.kill.assert_not_awaited()
            await factory.aclose()

        create.assert_awaited_once_with(
            "base",
            timeout=600,
            envs={"A": "1"},
            allow_internet_access=False,
            api_key="test",  # pragma: allowlist secret
        )
        remote.kill.assert_awaited_once()

    async def test_open_resolves_variables_from_context(self) -> None:
        context = Context(
            stream=MagicMock(),
            variables={"e2b_key": "tenant-key", "tenant_template": "tenant-tpl"},  # pragma: allowlist secret
        )
        with _patch_create(_fake_remote()) as create:
            factory = E2BEnvironment(api_key=Variable("e2b_key"), template=Variable("tenant_template"))
            async with factory.open(context):
                pass
            await factory.aclose()

        assert create.await_args.args == ("tenant-tpl",)
        assert create.await_args.kwargs["api_key"] == "tenant-key"  # pragma: allowlist secret

    async def test_open_missing_variable_raises_key_error(self) -> None:
        context = Context(stream=MagicMock(), variables={})
        factory = E2BEnvironment(template=Variable("tenant_template"))
        with pytest.raises(KeyError, match="tenant_template"):
            async with factory.open(context):
                pass

    async def test_missing_context_for_variable_raises(self) -> None:
        factory = E2BEnvironment(api_key=Variable("e2b_key"))
        with pytest.raises(RuntimeError, match="Variable but no Context"):
            async with factory.open():
                pass

    async def test_open_reuses_sandbox_until_environment_close(self) -> None:
        remote = _fake_remote()
        with _patch_create(remote) as create:
            factory = E2BEnvironment()
            async with factory.open() as first:
                pass
            async with factory.open() as second:
                assert second is first
            await factory.aclose()
            await factory.aclose()

        create.assert_awaited_once()
        remote.kill.assert_awaited_once()

    async def test_distinct_resolved_values_get_distinct_sandboxes(self) -> None:
        first_remote, second_remote = _fake_remote("sbx-a"), _fake_remote("sbx-b")
        with _patch_create(first_remote, second_remote):
            factory = E2BEnvironment(api_key=Variable("key"))
            async with factory.open(Context(stream=MagicMock(), variables={"key": "a"})) as a:
                pass
            async with factory.open(Context(stream=MagicMock(), variables={"key": "b"})) as b:
                pass
            assert (a.sandbox_id, b.sandbox_id) == ("sbx-a", "sbx-b")
            await factory.aclose()

        first_remote.kill.assert_awaited_once()
        second_remote.kill.assert_awaited_once()

    async def test_failed_creation_is_evicted_and_retried(self) -> None:
        remote = _fake_remote()
        create = AsyncMock(side_effect=[RuntimeError("quota"), remote])
        with patch("ag2.extensions.e2b.sandbox.AsyncSandbox.create", create):
            factory = E2BEnvironment()
            with pytest.raises(RuntimeError, match="quota"):
                async with factory.open():
                    pass
            async with factory.open() as sandbox:
                assert sandbox.sandbox_id == "sbx-1"
            await factory.aclose()

        assert create.await_count == 2

    async def test_close_retries_a_failed_kill(self) -> None:
        remote = _fake_remote()
        remote.kill = AsyncMock(side_effect=[RuntimeError("network"), True])
        with _patch_create(remote):
            factory = E2BEnvironment()
            async with factory.open():
                pass
            await factory.aclose()
            await factory.aclose()
            await factory.aclose()
        assert remote.kill.await_count == 2

    async def test_close_retries_a_sandbox_whose_setup_and_kill_failed(self) -> None:
        failed, fresh = _fake_remote("sbx-failed"), _fake_remote("sbx-fresh")
        failed.files.make_dir = AsyncMock(side_effect=RuntimeError("read-only"))
        failed.kill = AsyncMock(side_effect=[RuntimeError("network"), RuntimeError("network"), True])
        with _patch_create(failed, fresh):
            factory = E2BEnvironment(workdir="/workspace")
            with pytest.raises(RuntimeError, match="read-only"):
                async with factory.open():
                    pass
            # A retried open replaces the cache entry; the failed sandbox must not be forgotten.
            async with factory.open() as sandbox:
                assert sandbox.sandbox_id == "sbx-fresh"
            await factory.aclose()
        assert failed.kill.await_count == 3
        fresh.kill.assert_awaited_once()

    async def test_cancelled_close_leaves_the_rest_for_the_next_close(self) -> None:
        first, second = _fake_remote("sbx-a"), _fake_remote("sbx-b")
        hang = asyncio.Event()
        first.kill = AsyncMock(side_effect=_first_call_waits(hang))
        with _patch_create(first, second):
            factory = E2BEnvironment(api_key=Variable("key"))
            for key in ("a", "b"):
                async with factory.open(Context(stream=MagicMock(), variables={"key": key})):
                    pass
            closing = asyncio.create_task(factory.aclose())
            await asyncio.sleep(0.01)
            closing.cancel()
            await asyncio.gather(closing, return_exceptions=True)
            await factory.aclose()
        assert first.kill.await_count == 2
        second.kill.assert_awaited_once()

    async def test_tools_share_one_sandbox(self) -> None:
        remote = _fake_remote()
        with _patch_create(remote) as create:
            factory = E2BEnvironment()
            code = SandboxCodeTool(factory)
            result = await code.environment.run("print('ok')", "python")
            async with factory.open() as sandbox:
                assert (await sandbox.exec(["ls"])).exit_code == 0
            await factory.aclose()

        assert result.output == "ok"
        assert remote.commands.run.await_args_list[0].args == ("python -c 'print('\"'\"'ok'\"'\"')'",)
        create.assert_awaited_once()


class TestDeepcopy:
    def test_environment_deepcopy_returns_same_instance(self) -> None:
        env = E2BEnvironment()
        assert deepcopy(env) is env

    def test_sandbox_deepcopy_returns_same_instance(self) -> None:
        sandbox = E2BSandbox()
        assert deepcopy(sandbox) is sandbox

    def test_tool_backed_by_environment_is_deepcopyable(self) -> None:
        tool = SandboxShellTool(E2BEnvironment())
        assert isinstance(deepcopy(tool), SandboxShellTool)
