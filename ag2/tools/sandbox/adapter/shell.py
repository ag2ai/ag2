# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from ag2.tools.sandbox.base import ExecResult, Sandbox
from ag2.tools.sandbox.factory import SandboxFactory, SingletonFactory, WorkdirAware
from ag2.tools.sandbox.filter import (
    READONLY_COMMANDS,
    check_ignore,
    contains_shell_operator,
    matches_argv,
    split_command,
)

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
        allowed / blocked / ignore / readonly: command filter set. Any of them
                 switches on restricted mode, where the command runs as its
                 checked argv without a shell. With ``ignore`` set, wildcard
                 arguments are also rejected: some Windows runtimes expand
                 them even without a shell. ``blocked`` matches the argv word by word
                 (the program by its base name), so it cannot stop a program
                 that runs another one (``sh -c``, ``env``, ``xargs``): use
                 ``allowed`` / ``readonly`` or an isolated container for that.
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
        not-yet-opened remote :class:`SandboxFactory` has no sandbox bound yet,
        so it reports what it declares as a
        :class:`~ag2.tools.sandbox.WorkdirAware` factory, and the conventional
        ``/workspace`` when it declares nothing.
        """
        factory = self._factory
        if not isinstance(factory, SingletonFactory):
            if isinstance(factory, WorkdirAware):
                return factory.workdir
            return PurePosixPath("/workspace")
        sandbox = factory.sandbox
        host = sandbox.host_workdir
        if host is not None:
            return host
        return sandbox.workdir

    @property
    def restricted(self) -> bool:
        """Whether commands run as a checked argv, without a shell."""
        return self._allowed is not None or self._blocked is not None or self._ignore is not None

    def _argv(self, command: str) -> list[str] | None:
        # Restricted mode runs exactly the argv it checks: through ``sh -c`` the
        # shell would expand braces, variables, globs and ``$(...)`` after the check.
        if not self.restricted:
            return ["sh", "-c", command]
        return split_command(command)

    def _filter(self, command: str, argv: list[str]) -> str | None:
        if self._allowed is not None and not any(matches_argv(p, argv) for p in self._allowed):
            return f"Command not allowed: {command!r}"
        if self.restricted and contains_shell_operator(command):
            return f"Command not allowed (shell syntax is not available in restricted mode): {command!r}"
        if self._blocked is not None and argv:
            # Preserve path-based prefixes while also matching the program by
            # its base name, so ``/bin/rm`` is matched by ``rm``.
            head = [PurePosixPath(argv[0]).name, *argv[1:]]
            if any(matches_argv(p, argv) or matches_argv(p, head) for p in self._blocked):
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
        """Run ``command`` and return its output, or the reason it was refused."""
        argv = self._argv(command)
        if argv is None:
            return f"Command not allowed (unbalanced quotes): {command!r}"
        denied = self._filter(command, argv)
        if denied is not None:
            return denied

        async with self._factory.open(context) as sandbox:
            result = await sandbox.exec(argv, env=self._env, timeout=self._timeout)
        return _format(result)


def _format(result: ExecResult) -> str:
    if result.exit_code != 0:
        suffix = f"[exit code: {result.exit_code}]"
        return f"{result.output}\n{suffix}" if result.output else suffix
    return result.output
