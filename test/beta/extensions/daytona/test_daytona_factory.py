# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autogen.beta import Context, Variable
from autogen.beta.extensions.daytona import DaytonaEnvironment, DaytonaSandbox
from autogen.beta.tools.sandbox import SandboxFactory


def _fake_sandbox() -> Any:
    response = SimpleNamespace(result="ok", exit_code=0)
    return SimpleNamespace(
        id="sb-1",
        process=SimpleNamespace(exec=AsyncMock(return_value=response)),
        fs=SimpleNamespace(
            upload_file=AsyncMock(return_value=None),
            download_file=AsyncMock(return_value=b""),
            delete_file=AsyncMock(return_value=None),
        ),
        delete=AsyncMock(return_value=None),
    )


def _fake_client(sandbox: Any) -> Any:
    return SimpleNamespace(
        create=AsyncMock(return_value=sandbox),
        close=AsyncMock(return_value=None),
    )


def _patch_async_daytona(sandbox: Any) -> Any:
    return patch(
        "autogen.beta.extensions.daytona.factory.AsyncDaytona",
        return_value=_fake_client(sandbox),
    )


def test_satisfies_factory_protocol() -> None:
    factory: SandboxFactory = DaytonaEnvironment(api_key="test")
    assert isinstance(factory, SandboxFactory)


def test_snapshot_and_image_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="snapshot.*image"):
        DaytonaEnvironment(snapshot="snap", image="python:3.12")


def test_invalid_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="timeout"):
        DaytonaEnvironment(api_key="test", timeout=0)


@pytest.mark.asyncio
class TestOpen:
    async def test_open_yields_daytona_sandbox(self) -> None:
        sandbox = _fake_sandbox()
        with _patch_async_daytona(sandbox):
            factory = DaytonaEnvironment(api_key="test", image="python:3.12")
            async with factory.open() as sb:
                assert isinstance(sb, DaytonaSandbox)

    async def test_open_resolves_image_variable_from_context(self) -> None:
        sandbox = _fake_sandbox()
        ctx = Context(stream=MagicMock(), variables={"tenant_image": "python:3.11"})
        with _patch_async_daytona(sandbox) as patched:
            factory = DaytonaEnvironment(api_key="test", image=Variable("tenant_image"))
            async with factory.open(ctx) as _sb:
                pass

        params = patched.return_value.create.await_args.args[0]
        assert params.image == "python:3.11"

    async def test_open_missing_variable_raises_key_error(self) -> None:
        ctx = Context(stream=MagicMock(), variables={})
        factory = DaytonaEnvironment(api_key="test", image=Variable("tenant_image"))
        with pytest.raises(KeyError, match="tenant_image"):
            async with factory.open(ctx):
                pass

    async def test_open_variable_credentials_without_context_raises(self) -> None:
        factory = DaytonaEnvironment(api_key=Variable("k"))
        with pytest.raises(RuntimeError, match="Variable but no Context"):
            async with factory.open():
                pass

    async def test_open_keeps_sandbox_alive_for_reuse(self) -> None:
        # Caching: the sandbox survives scope exit so the next open() reuses
        # it. Only aclose() deletes it.
        sandbox = _fake_sandbox()
        with _patch_async_daytona(sandbox):
            factory = DaytonaEnvironment(api_key="test", image="python:3.12")
            async with factory.open() as sb:
                await sb.exec(["echo", "hi"])
            sandbox.delete.assert_not_awaited()

            async with factory.open() as sb2:
                assert sb2 is sb

            await factory.aclose()
        sandbox.delete.assert_awaited_once()

    async def test_aclose_tears_down_cached_sandbox(self) -> None:
        sandbox = _fake_sandbox()
        with _patch_async_daytona(sandbox):
            factory = DaytonaEnvironment(api_key="test", image="python:3.12")
            async with factory.open() as sb:
                await sb.exec(["echo", "hi"])
            await factory.aclose()
            await factory.aclose()  # idempotent

        sandbox.delete.assert_awaited_once()
