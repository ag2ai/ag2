# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import shutil
import tempfile
from pathlib import Path

from autogen.beta.tools.sandbox.local import _run_subprocess

from .base import READONLY_COMMANDS, ShellEnvironment, check_ignore, matches


class LocalShellEnvironment(ShellEnvironment):
    """Executes shell commands locally via :mod:`subprocess`.

    Satisfies the :class:`~autogen.beta.tools.shell.ShellEnvironment` protocol.
    All security constraints (allowed/blocked commands, path filtering,
    timeouts) are encapsulated here — callers interact only through
    :meth:`run` and :attr:`workdir`.

    Quick start::

        # No restrictions — agent can run any command
        env = LocalShellEnvironment(path="/tmp/my_project")

        # Only allow git and python commands
        env = LocalShellEnvironment(
            path="/tmp/my_project",
            allowed=["git", "python", "pip"],
        )

        # Block dangerous commands
        env = LocalShellEnvironment(
            path="/tmp/my_project",
            blocked=["rm -rf", "curl", "wget"],
        )

        # Hide sensitive files from the agent
        env = LocalShellEnvironment(
            path="/tmp/my_project",
            ignore=["**/.env", "*.key", "secrets/**"],
        )

        # Read-only mode — agent can inspect but not modify
        env = LocalShellEnvironment(
            path="/tmp/my_project",
            readonly=True,
        )

    Args:
        path: Working directory for all shell commands. If ``None``, a temporary
              directory is created automatically with prefix ``"ag2_shell_"``.
        cleanup: Delete the directory on process exit. Defaults to ``True`` when
                 ``path=None`` (auto temp dir) and ``False`` when ``path`` is set.
        allowed: Whitelist of command prefixes. Only commands *starting with* one
                 of these strings are executed.
        blocked: Blacklist of command prefixes. Commands *starting with* any of
                 these strings are rejected.
        ignore: Gitignore-style path patterns. Literal file paths parsed from the
                command string are resolved and checked against these patterns.
        readonly: If ``True`` and ``allowed`` is not set, restrict commands to a
                  built-in read-only list (``cat``, ``ls``, ``grep``, etc.).
        env: Extra environment variables merged into each command's environment.
        timeout: Per-command timeout in seconds. Default: 60.
        max_output: Maximum number of characters returned from a command. Default: 100 000.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        cleanup: bool | None = None,
        allowed: list[str] | None = None,
        blocked: list[str] | None = None,
        ignore: list[str] | None = None,
        readonly: bool = False,
        env: dict[str, str] | None = None,
        timeout: float = 60,
        max_output: int = 100_000,
    ) -> None:
        cleanup_flag = cleanup if cleanup is not None else (path is None)

        if path is None:
            tmpdir = tempfile.mkdtemp(prefix="ag2_shell_")
            self._workdir = Path(tmpdir)
        else:
            self._workdir = Path(path).resolve()
            self._workdir.mkdir(parents=True, exist_ok=True)

        if cleanup_flag:
            workdir_str = str(self._workdir)
            atexit.register(lambda: shutil.rmtree(workdir_str, ignore_errors=True))

        # readonly=True with no explicit allowed → use built-in read-only list.
        # explicit allowed always takes precedence over readonly.
        self._allowed: list[str] | None = list(READONLY_COMMANDS) if readonly and allowed is None else allowed
        self._blocked = blocked
        self._ignore = ignore
        self._env = env
        self._timeout = timeout
        self._max_output = max_output

    @classmethod
    def ensure_env(cls, env: ShellEnvironment | str | os.PathLike[str]) -> ShellEnvironment:
        if isinstance(env, ShellEnvironment):
            return env
        return cls(env)

    @property
    def workdir(self) -> Path:
        """The working directory used for command execution."""
        return self._workdir

    def run(self, command: str) -> str:
        """Execute *command* and return its output as a string.

        Applies allowed/blocked filtering, ignore-pattern checks, then runs
        the command via the shared :func:`_run_subprocess` helper used by
        :class:`~autogen.beta.tools.sandbox.LocalSandbox`.
        """
        if self._allowed is not None and not any(matches(p, command) for p in self._allowed):
            return f"Command not allowed: {command!r}"

        if self._blocked is not None and any(matches(p, command) for p in self._blocked):
            return f"Command not allowed: {command!r}"

        if self._ignore is not None:
            denied = check_ignore(command, self._workdir, self._ignore)
            if denied is not None:
                return denied

        result = _run_subprocess(
            [command],
            cwd=self._workdir,
            env=self._env,
            timeout=self._timeout,
            max_output=self._max_output,
            shell=True,
        )

        if result.exit_code != 0:
            suffix = f"[exit code: {result.exit_code}]"
            return f"{result.output}\n{suffix}" if result.output else suffix
        return result.output
