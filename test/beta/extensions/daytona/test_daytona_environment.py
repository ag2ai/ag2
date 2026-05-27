# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import daytona
import pytest

from autogen.beta import Context, Variable
from autogen.beta.extensions.daytona import DaytonaCodeEnvironment, DaytonaResources


def _fake_sandbox(result: str = "ok", exit_code: int = 0, sandbox_id: str = "sb-1") -> Any:
    """Build a SimpleNamespace that quacks like a Daytona sandbox."""
    response = SimpleNamespace(result=result, exit_code=exit_code)
    return SimpleNamespace(
        id=sandbox_id,
        process=SimpleNamespace(
            code_run=AsyncMock(return_value=response),
            exec=AsyncMock(return_value=response),
        ),
        fs=SimpleNamespace(
            upload_file=AsyncMock(return_value=None),
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
    """Patch AsyncDaytona to return a client whose .create() yields *sandbox*."""
    return patch(
        "autogen.beta.extensions.daytona.environment.AsyncDaytona",
        return_value=_fake_client(sandbox),
    )


class TestConstruction:
    def test_snapshot_and_image_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="snapshot.*image"):
            DaytonaCodeEnvironment(snapshot="snap", image="python:3.12")

    def test_invalid_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            DaytonaCodeEnvironment(timeout=0)

    def test_supported_languages_default(self) -> None:
        env = DaytonaCodeEnvironment()
        assert env.supported_languages == ("python", "bash", "javascript", "typescript")

    def test_supported_languages_custom(self) -> None:
        env = DaytonaCodeEnvironment(languages=("python",))
        assert env.supported_languages == ("python",)

    def test_construction_creates_no_sandbox(self) -> None:
        # CLAUDE.md: no side effects in __init__ — sandbox must be lazy.
        with patch("autogen.beta.extensions.daytona.environment.AsyncDaytona") as mock_client:
            DaytonaCodeEnvironment(api_key="test")
            mock_client.assert_not_called()


@pytest.mark.asyncio
class TestRun:
    async def test_python_uses_code_run(self) -> None:
        sandbox = _fake_sandbox(result="42\n")
        with _patch_async_daytona(sandbox):
            env = DaytonaCodeEnvironment(api_key="test")
            result = await env.run("print(40+2)", "python")

        assert result.exit_code == 0
        assert "42" in result.output
        sandbox.process.code_run.assert_awaited_once_with("print(40+2)", timeout=60)

    async def test_bash_uploads_and_execs(self) -> None:
        sandbox = _fake_sandbox(result="hello\n")
        with _patch_async_daytona(sandbox):
            env = DaytonaCodeEnvironment(api_key="test")
            result = await env.run("echo hello", "bash")

        assert result.exit_code == 0
        assert "hello" in result.output
        sandbox.fs.upload_file.assert_awaited_once()
        sandbox.process.exec.assert_awaited_once()
        # Best-effort cleanup of the temp script
        sandbox.fs.delete_file.assert_awaited_once()

    async def test_javascript_uses_node(self) -> None:
        sandbox = _fake_sandbox()
        with _patch_async_daytona(sandbox):
            env = DaytonaCodeEnvironment(api_key="test")
            await env.run("console.log(1)", "javascript")

        # Verify the exec command starts with `node`
        cmd = sandbox.process.exec.await_args.args[0]
        assert cmd.startswith("node ")

    async def test_disabled_language_returns_error_without_creating_sandbox(self) -> None:
        env = DaytonaCodeEnvironment(api_key="test", languages=("python",))
        # No patch — sandbox creation must not happen for a rejected language
        result = await env.run("echo nope", "bash")

        assert result.exit_code != 0
        assert "not enabled" in result.output

    async def test_sandbox_created_only_once(self) -> None:
        sandbox = _fake_sandbox()
        client = _fake_client(sandbox)
        with patch(
            "autogen.beta.extensions.daytona.environment.AsyncDaytona",
            return_value=client,
        ):
            env = DaytonaCodeEnvironment(api_key="test")
            await env.run("print(1)", "python")
            await env.run("print(2)", "python")
            await env.run("print(3)", "python")

        # Three runs, one sandbox creation
        assert client.create.await_count == 1

    async def test_timeout_returns_124(self) -> None:
        sandbox = _fake_sandbox()
        sandbox.process.code_run = AsyncMock(side_effect=daytona.DaytonaTimeoutError("slow"))
        with _patch_async_daytona(sandbox):
            env = DaytonaCodeEnvironment(api_key="test")
            result = await env.run("while True: pass", "python")

        assert result.exit_code == 124
        assert "timed out" in result.output.lower()

    async def test_rate_limit_surfaces_as_error_string(self) -> None:
        sandbox = _fake_sandbox()
        sandbox.process.code_run = AsyncMock(side_effect=daytona.DaytonaRateLimitError("slow down"))
        with _patch_async_daytona(sandbox):
            env = DaytonaCodeEnvironment(api_key="test")
            result = await env.run("print(1)", "python")

        assert result.exit_code != 0
        assert "rate limit" in result.output.lower()


@pytest.mark.asyncio
class TestLifecycle:
    async def test_aclose_deletes_sandbox_and_closes_client(self) -> None:
        sandbox = _fake_sandbox()
        client = _fake_client(sandbox)
        with patch(
            "autogen.beta.extensions.daytona.environment.AsyncDaytona",
            return_value=client,
        ):
            env = DaytonaCodeEnvironment(api_key="test")
            await env.run("print(1)", "python")
            await env.aclose()

        sandbox.delete.assert_awaited_once()
        client.close.assert_awaited_once()

    async def test_aclose_idempotent(self) -> None:
        sandbox = _fake_sandbox()
        with _patch_async_daytona(sandbox):
            env = DaytonaCodeEnvironment(api_key="test")
            await env.run("print(1)", "python")
            await env.aclose()
            await env.aclose()

        sandbox.delete.assert_awaited_once()

    async def test_aclose_without_run_is_safe(self) -> None:
        env = DaytonaCodeEnvironment(api_key="test")
        # Never created a sandbox — aclose must not raise
        await env.aclose()

    async def test_async_context_manager_cleans_up(self) -> None:
        sandbox = _fake_sandbox()
        with _patch_async_daytona(sandbox):
            async with DaytonaCodeEnvironment(api_key="test") as env:
                await env.run("print(1)", "python")

        sandbox.delete.assert_awaited_once()


class TestResources:
    def test_resources_dataclass(self) -> None:
        r = DaytonaResources(cpu=2, memory=2048, disk=10)
        assert r.cpu == 2
        assert r.memory == 2048
        assert r.disk == 10

    def test_resources_all_optional(self) -> None:
        r = DaytonaResources()
        assert r.cpu is None
        assert r.memory is None
        assert r.disk is None


@pytest.mark.asyncio
class TestVariableResolution:
    """Per CLAUDE.md: 2 tests for Variable params — resolve and missing-key.

    Uses ``image`` as the representative Variable param; the resolution
    path is shared across all Variable-accepting fields.
    """

    async def test_image_resolved_from_context(self) -> None:
        sandbox = _fake_sandbox()
        ctx = Context(stream=MagicMock(), variables={"tenant_image": "python:3.11"})
        with _patch_async_daytona(sandbox) as mock_async_daytona:
            env = DaytonaCodeEnvironment(api_key="test", image=Variable("tenant_image"))
            await env.run("print(1)", "python", context=ctx)

        # Verify the resolved image was used in the create call
        create_call = mock_async_daytona.return_value.create.await_args
        params = create_call.args[0]
        assert params.image == "python:3.11"

    async def test_missing_variable_raises_key_error(self) -> None:
        ctx = Context(stream=MagicMock(), variables={})
        # No need to patch AsyncDaytona — resolution fails before sandbox creation
        env = DaytonaCodeEnvironment(api_key="test", image=Variable("tenant_image"))

        with pytest.raises(KeyError, match="tenant_image"):
            await env.run("print(1)", "python", context=ctx)
