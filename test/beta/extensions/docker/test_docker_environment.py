# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from autogen.beta import Context, Variable
from autogen.beta.extensions.docker import DockerCodeEnvironment


def _exec_result(output: bytes = b"ok\n", exit_code: int = 0) -> SimpleNamespace:
    return SimpleNamespace(output=output, exit_code=exit_code)


def _fake_container(exec_result: SimpleNamespace | None = None, container_id: str = "deadbeef") -> Any:
    return SimpleNamespace(
        short_id=container_id,
        exec_run=MagicMock(return_value=exec_result or _exec_result()),
        stop=MagicMock(return_value=None),
        remove=MagicMock(return_value=None),
    )


def _fake_client(container: Any) -> Any:
    return SimpleNamespace(
        containers=SimpleNamespace(run=MagicMock(return_value=container)),
        close=MagicMock(return_value=None),
    )


def _patch_docker(container: Any) -> Any:
    return patch(
        "autogen.beta.extensions.docker.environment.docker.from_env",
        return_value=_fake_client(container),
    )


class TestConstruction:
    def test_invalid_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            DockerCodeEnvironment(timeout=0)

    def test_supported_languages_default(self) -> None:
        env = DockerCodeEnvironment()
        assert env.supported_languages == ("python", "bash")

    def test_supported_languages_custom(self) -> None:
        env = DockerCodeEnvironment(languages=("python", "javascript"))
        assert env.supported_languages == ("python", "javascript")

    def test_construction_creates_no_container(self) -> None:
        # CLAUDE.md: no side effects in __init__ — container must be lazy.
        with patch("autogen.beta.extensions.docker.environment.docker.from_env") as mock_from_env:
            DockerCodeEnvironment()
            mock_from_env.assert_not_called()

    def test_safety_defaults(self) -> None:
        env = DockerCodeEnvironment()
        assert env._network_mode == "none"
        assert env._mem_limit == "512m"
        assert env._auto_remove is True


@pytest.mark.asyncio
class TestRun:
    async def test_python_uses_python_dash_c(self) -> None:
        container = _fake_container(_exec_result(b"42\n"))
        with _patch_docker(container):
            env = DockerCodeEnvironment()
            result = await env.run("print(40+2)", "python")

        assert result.exit_code == 0
        assert "42" in result.output
        cmd_arg = container.exec_run.call_args.args[0]
        assert cmd_arg[:2] == ["python", "-c"]
        assert cmd_arg[2] == "print(40+2)"

    async def test_bash_uses_bash_dash_c(self) -> None:
        container = _fake_container(_exec_result(b"hello\n"))
        with _patch_docker(container):
            env = DockerCodeEnvironment()
            result = await env.run("echo hello", "bash")

        assert result.exit_code == 0
        cmd_arg = container.exec_run.call_args.args[0]
        assert cmd_arg[:2] == ["bash", "-c"]

    async def test_javascript_uses_node_via_tempfile(self) -> None:
        container = _fake_container(_exec_result(b"1\n"))
        with _patch_docker(container):
            env = DockerCodeEnvironment(languages=("python", "bash", "javascript"))
            await env.run("console.log(1)", "javascript")

        # File-based path: cmd is ["sh", "-c", "<base64 decode then `node <file>`>"]
        cmd_arg = container.exec_run.call_args.args[0]
        assert cmd_arg[:2] == ["sh", "-c"]
        assert "node " in cmd_arg[2]

    async def test_disabled_language_returns_error_without_creating_container(self) -> None:
        env = DockerCodeEnvironment(languages=("python",))
        # No patch — container creation must not happen for a rejected language
        result = await env.run("echo nope", "bash")

        assert result.exit_code != 0
        assert "not enabled" in result.output

    async def test_container_created_only_once(self) -> None:
        container = _fake_container()
        client = _fake_client(container)
        with patch(
            "autogen.beta.extensions.docker.environment.docker.from_env",
            return_value=client,
        ):
            env = DockerCodeEnvironment()
            await env.run("print(1)", "python")
            await env.run("print(2)", "python")
            await env.run("print(3)", "python")

        # Three runs, one container creation
        assert client.containers.run.call_count == 1

    async def test_run_passes_safety_kwargs(self) -> None:
        container = _fake_container()
        client = _fake_client(container)
        with patch(
            "autogen.beta.extensions.docker.environment.docker.from_env",
            return_value=client,
        ):
            env = DockerCodeEnvironment(image="python:3.12-slim", mem_limit="256m", network_mode="none")
            await env.run("print(1)", "python")

        call = client.containers.run.call_args
        assert call.args[0] == "python:3.12-slim"
        assert call.kwargs["network_mode"] == "none"
        assert call.kwargs["mem_limit"] == "256m"
        assert call.kwargs["detach"] is True


@pytest.mark.asyncio
class TestLifecycle:
    async def test_aclose_stops_container_and_closes_client(self) -> None:
        container = _fake_container()
        client = _fake_client(container)
        with patch(
            "autogen.beta.extensions.docker.environment.docker.from_env",
            return_value=client,
        ):
            env = DockerCodeEnvironment()
            await env.run("print(1)", "python")
            await env.aclose()

        container.stop.assert_called_once()
        client.close.assert_called_once()

    async def test_aclose_idempotent(self) -> None:
        container = _fake_container()
        with _patch_docker(container):
            env = DockerCodeEnvironment()
            await env.run("print(1)", "python")
            await env.aclose()
            await env.aclose()

        container.stop.assert_called_once()

    async def test_aclose_without_run_is_safe(self) -> None:
        env = DockerCodeEnvironment()
        await env.aclose()  # no container created — must not raise

    async def test_async_context_manager_cleans_up(self) -> None:
        container = _fake_container()
        with _patch_docker(container):
            async with DockerCodeEnvironment() as env:
                await env.run("print(1)", "python")

        container.stop.assert_called_once()


@pytest.mark.asyncio
class TestVariableResolution:
    """Per CLAUDE.md: 2 tests for Variable params — resolve and missing-key."""

    async def test_image_resolved_from_context(self) -> None:
        container = _fake_container()
        client = _fake_client(container)
        ctx = Context(stream=MagicMock(), variables={"tenant_image": "python:3.11-slim"})
        with patch(
            "autogen.beta.extensions.docker.environment.docker.from_env",
            return_value=client,
        ):
            env = DockerCodeEnvironment(image=Variable("tenant_image"))
            await env.run("print(1)", "python", context=ctx)

        # The resolved image must be what was passed to containers.run
        call = client.containers.run.call_args
        assert call.args[0] == "python:3.11-slim"

    async def test_missing_variable_raises_key_error(self) -> None:
        ctx = Context(stream=MagicMock(), variables={})
        env = DockerCodeEnvironment(image=Variable("tenant_image"))

        with pytest.raises(KeyError, match="tenant_image"):
            await env.run("print(1)", "python", context=ctx)
