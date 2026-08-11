# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from ag2.tools.sandbox.base import ExecResult, Sandbox
from ag2.tools.sandbox.factory import SandboxFactory, SingletonFactory
from ag2.tools.sandbox.filter import READONLY_COMMANDS, check_ignore, contains_shell_operator, matches

if TYPE_CHECKING:
    from ag2.context import ConversationContext


class ShellAdapter:
    """Shell surface (``run``) over any :class:`Sandbox`.

    Implements the command policy once and works on every backend — local
    subprocess, Docker container, Daytona sandbox, or any custom one.

    Filtering (``allowed`` / ``blocked`` / ``ignore`` / ``readonly``)
    lives here once. Execution delegates to the wrapped
    :class:`Sandbox` or :class:`SandboxFactory`; the adapter never
    duplicates backend logic.

    Args:
        sandbox: Either a long-lived :class:`Sandbox` (used as-is) or a
                 :class:`SandboxFactory` (opened per :meth:`run` so
                 :class:`~ag2.annotations.Variable` parameters
                 get resolved against the active Context).
        allowed / blocked / ignore / readonly: command filter set.
                 ``blocked`` is best-effort: it only matches the head command's
                 prefix, so chaining (``;`` / ``|`` / ``&&`` / ``$(...)``) can
                 bypass it. It is **not** a security boundary — use ``allowed`` /
                 ``readonly`` or an isolated container for that.
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
        self._factory: SandboxFactory = sandbox if isinstance(sandbox, SandboxFactory) else SingletonFactory(sandbox)
        self._allowed: list[str] | None = list(READONLY_COMMANDS) if readonly and allowed is None else allowed
        self._blocked = blocked
        self._ignore = ignore
        self._env = env
        self._timeout = timeout

    @property
    def workdir(self) -> "Path | PurePosixPath":
        """Working directory exposed to callers.

        For a host-backed sandbox (local subprocess, incl. a
        :class:`~ag2.tools.sandbox.LocalEnvironment` /
        :class:`SingletonFactory` wrapping one) this is the real host
        :class:`~pathlib.Path` (so ``.exists()`` etc. work); for a remote /
        container backend it is the sandbox-side :class:`PurePosixPath`. A
        not-yet-opened remote :class:`SandboxFactory` reports the
        conventional ``/workspace`` since no sandbox is bound yet.
        """
        factory = self._factory
        if not isinstance(factory, SingletonFactory):
            return PurePosixPath("/workspace")
        sandbox = factory.sandbox
        host = sandbox.host_workdir
        if host is not None:
            return host
        return sandbox.workdir

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
            # self.workdir gives a host Path for local backends and a
            # PurePosixPath for remote/container ones — check_ignore handles
            # both, so ignore applies on every backend (not just local).
            denied = check_ignore(command, self.workdir, self._ignore)
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

    async def _exec_async(
        self,
        command: str,
        context: "ConversationContext | None",
    ) -> ExecResult:
        argv = ["sh", "-c", command]
        async with self._factory.open(context) as sandbox:
            return await sandbox.exec(argv, env=self._env, timeout=self._timeout)


def _format(result: ExecResult) -> str:
    if result.exit_code != 0:
        suffix = f"[exit code: {result.exit_code}]"
        return f"{result.output}\n{suffix}" if result.output else suffix
    return result.output
