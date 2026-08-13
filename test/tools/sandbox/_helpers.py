# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Reusable sandbox doubles for tool tests.

A remote backend reaches AG2 as a :class:`~ag2.tools.sandbox.SandboxFactory`
that opens a :class:`~ag2.tools.sandbox.Sandbox`. These doubles stand in for
that pair without a container or a cloud account.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

from ag2.tools.sandbox import ExecResult, Sandbox, SandboxBase


class RecordingSandbox(SandboxBase):
    """A remote/container-style sandbox: a sandbox-side workdir and no host
    filesystem (``host_workdir is None``). Records every argv it is asked to
    run instead of executing anything.
    """

    def __init__(self, *, workdir: str = "/workspace", output: str = "ok", exit_code: int = 0) -> None:
        self.execs: list[list[str]] = []
        self._workdir = PurePosixPath(workdir)
        self._output = output
        self._exit_code = exit_code

    @property
    def workdir(self) -> PurePosixPath:
        return self._workdir

    @property
    def host_workdir(self) -> Path | None:
        return None

    async def exec(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        self.execs.append(list(argv))
        return ExecResult(output=self._output, exit_code=self._exit_code)


class RecordingFactory:
    """A remote factory that declares none of the optional backend hooks.

    Records the context it was opened with, so tests can assert that a tool
    forwards the :class:`~ag2.Context` it received.
    """

    def __init__(self, sandbox: Sandbox | None = None) -> None:
        self.sandbox = sandbox if sandbox is not None else RecordingSandbox()
        self.contexts: list[object] = []

    @asynccontextmanager
    async def open(self, context: object = None) -> AsyncGenerator[Sandbox]:
        self.contexts.append(context)
        yield self.sandbox


class WorkdirDeclaringFactory(RecordingFactory):
    """A remote factory that satisfies :class:`~ag2.tools.sandbox.WorkdirAware`."""

    def __init__(self, workdir: PurePosixPath, sandbox: Sandbox | None = None) -> None:
        super().__init__(sandbox)
        self._declared = workdir

    @property
    def workdir(self) -> PurePosixPath:
        return self._declared
