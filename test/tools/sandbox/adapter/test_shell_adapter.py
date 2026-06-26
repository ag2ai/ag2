# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock

import pytest

from ag2 import Context
from ag2.tools.sandbox import ExecResult, Sandbox
from ag2.tools.sandbox.adapter import ShellAdapter
from ag2.tools.sandbox.local import LocalSandbox


class _RecordingFactory:
    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox
        self.contexts: list[object] = []

    @asynccontextmanager
    async def open(self, context: object = None) -> AsyncIterator[Sandbox]:
        self.contexts.append(context)
        yield self._sandbox


class _FakeRemoteSandbox:
    """A remote/container-style sandbox: a sandbox-side ``PurePosixPath``
    workdir and no host filesystem (``host_workdir is None``)."""

    def __init__(self) -> None:
        self.execs: list[Sequence[str]] = []

    @property
    def workdir(self) -> PurePosixPath:
        return PurePosixPath("/workspace")

    @property
    def host_workdir(self) -> None:
        return None

    async def exec(self, argv: Sequence[str], *, env: object = None, timeout: object = None) -> ExecResult:
        self.execs.append(argv)
        return ExecResult(output="ok", exit_code=0)


@pytest.mark.asyncio
class TestShellAdapterFiltering:
    async def test_allowed_blocks_non_matching_command(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        adapter = ShellAdapter(sandbox, allowed=["echo"])
        result = await adapter.run("touch file.txt")
        assert "Command not allowed" in result

    async def test_blocked_rejects_matching_command(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        adapter = ShellAdapter(sandbox, blocked=["rm -rf"])
        result = await adapter.run("rm -rf /workspace")
        assert "Command not allowed" in result

    async def test_ignore_denies_access_to_matching_path(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("SECRET=1")
        sandbox = LocalSandbox(tmp_path)
        adapter = ShellAdapter(sandbox, ignore=["**/.env"])
        result = await adapter.run("cat .env")
        assert "Access denied" in result

    async def test_ignore_applies_on_remote_backend(self) -> None:
        # A remote backend has no host workdir; ignore must still apply by
        # matching literal argv paths against the sandbox-side workdir.
        sandbox = _FakeRemoteSandbox()
        adapter = ShellAdapter(sandbox, ignore=["**/.env"])
        result = await adapter.run("cat .env")
        assert "Access denied" in result
        assert sandbox.execs == []  # blocked before reaching the backend

    async def test_ignore_allows_non_matching_on_remote_backend(self) -> None:
        sandbox = _FakeRemoteSandbox()
        adapter = ShellAdapter(sandbox, ignore=["**/.env"])
        result = await adapter.run("cat README.md")
        assert "ok" in result
        assert len(sandbox.execs) == 1

    async def test_readonly_blocks_writes_by_default(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        adapter = ShellAdapter(sandbox, readonly=True)
        result = await adapter.run("touch new.txt")
        assert "Command not allowed" in result

    async def test_readonly_allows_read_commands(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        adapter = ShellAdapter(sandbox, readonly=True)
        result = await adapter.run("echo hello")
        assert "hello" in result


@pytest.mark.asyncio
class TestShellAdapterAsync:
    async def test_run_executes_command(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        adapter = ShellAdapter(sandbox)
        result = await adapter.run("echo hi")
        assert "hi" in result

    async def test_run_includes_exit_code_on_failure(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        adapter = ShellAdapter(sandbox)
        result = await adapter.run("exit 7")
        assert "exit code: 7" in result


@pytest.mark.asyncio
class TestShellAdapterWithFactory:
    async def test_factory_opens_per_call(self, tmp_path: Path) -> None:
        factory = _RecordingFactory(LocalSandbox(tmp_path))
        adapter = ShellAdapter(factory)

        await adapter.run("echo a")
        await adapter.run("echo b")

        assert len(factory.contexts) == 2

    async def test_context_variables_forwarded_to_factory(self, tmp_path: Path) -> None:
        factory = _RecordingFactory(LocalSandbox(tmp_path))
        adapter = ShellAdapter(factory)
        ctx = Context(stream=MagicMock(), variables={"x": "value"})

        await adapter.run("echo a", context=ctx)

        assert factory.contexts == [ctx]
