# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from autogen.beta.tools.sandbox.base import ExecResult, Sandbox
from autogen.beta.tools.sandbox.factory import SandboxFactory, SingletonFactory
from autogen.beta.tools.sandbox.filter import READONLY_COMMANDS, check_ignore, contains_shell_operator, matches
from autogen.beta.tools.sandbox.local import LocalSandbox

if TYPE_CHECKING:
    from autogen.beta.context import ConversationContext


class ShellAdapter:
    """One :class:`ShellEnvironment` implementation that works on every
    :class:`Sandbox`.

    Filtering (``allowed`` / ``blocked`` / ``ignore`` / ``readonly``)
    lives here once. Execution delegates to the wrapped
    :class:`Sandbox` or :class:`SandboxFactory`; the adapter never
    duplicates backend logic.

    Args:
        sandbox: Either a long-lived :class:`Sandbox` (used as-is) or a
                 :class:`SandboxFactory` (opened per :meth:`run` so
                 :class:`~autogen.beta.annotations.Variable` parameters
                 get resolved against the active Context).
        allowed / blocked / ignore / readonly: filter set, identical
                 to the v1 :class:`LocalShellEnvironment` semantics.
        env: Extra environment variables passed into each command.
        timeout: Per-command timeout in seconds. ``None`` lets the
                 backend pick its default.
    """

    def __init__(
        self,
        sandbox: "Sandbox | SandboxFactory",
        *,
        allowed: list[str] | None = None,
        blocked: list[str] | None = None,
        ignore: list[str] | None = None,
        readonly: bool = False,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._allowed: list[str] | None = list(READONLY_COMMANDS) if readonly and allowed is None else allowed
        self._blocked = blocked
        self._ignore = ignore
        self._env = env
        self._timeout = timeout

    @property
    def workdir(self) -> PurePosixPath:
        """Working directory exposed to callers as a sandbox-side POSIX path."""
        if isinstance(self._sandbox, SandboxFactory):
            return PurePosixPath("/workspace")
        return self._sandbox.workdir  # type: ignore[union-attr]

    def _filter(self, command: str) -> str | None:
        if self._allowed is not None:
            if not any(matches(p, command) for p in self._allowed):
                return f"Command not allowed: {command!r}"
            # In restricted mode, shell operators (redirection, pipes,
            # chaining, command substitution) would let an allowed head
            # command spawn or redirect to disallowed ones — block them.
            if contains_shell_operator(command):
                return f"Command not allowed (shell operators are not permitted in restricted mode): {command!r}"
        if self._blocked is not None and any(matches(p, command) for p in self._blocked):
            return f"Command not allowed: {command!r}"
        if self._ignore is not None:
            host = _host_workdir_for(self._sandbox)
            if host is not None:
                denied = check_ignore(command, host, self._ignore)
                if denied is not None:
                    return denied
        return None

    async def run(
        self,
        command: str,
        *,
        context: "ConversationContext | None" = None,
    ) -> str:
        denied = self._filter(command)
        if denied is not None:
            return denied

        result = await self._exec_async(command, context)
        return _format(result)

    def run_sync(
        self,
        command: str,
        *,
        context: "ConversationContext | None" = None,
    ) -> str:
        denied = self._filter(command)
        if denied is not None:
            return denied

        sandbox = self._sandbox
        if isinstance(sandbox, LocalSandbox):
            result = sandbox.exec_sync(
                ["sh", "-c", command],
                env=self._env,
                timeout=self._timeout,
            )
            return _format(result)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                f"{type(self).__name__}.run_sync() cannot be invoked from inside an "
                "active event loop. Drive the underlying SandboxFactory directly "
                "from async callers."
            )

        return asyncio.run(self.run(command, context=context))

    async def _exec_async(
        self,
        command: str,
        context: "ConversationContext | None",
    ) -> ExecResult:
        argv = ["sh", "-c", command]
        if isinstance(self._sandbox, SandboxFactory):
            async with self._sandbox.open(context) as sandbox:
                return await sandbox.exec(argv, env=self._env, timeout=self._timeout)
        return await self._sandbox.exec(argv, env=self._env, timeout=self._timeout)  # type: ignore[union-attr]


def _format(result: ExecResult) -> str:
    if result.exit_code != 0:
        suffix = f"[exit code: {result.exit_code}]"
        return f"{result.output}\n{suffix}" if result.output else suffix
    return result.output


def _host_workdir_for(sandbox: "Sandbox | SandboxFactory"):  # type: ignore[no-untyped-def]
    if isinstance(sandbox, SingletonFactory):
        sandbox = sandbox._sandbox  # noqa: SLF001
    if isinstance(sandbox, SandboxFactory):
        return None
    return getattr(sandbox, "host_workdir", None)
