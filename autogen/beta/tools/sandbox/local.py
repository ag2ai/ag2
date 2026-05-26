# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import atexit
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from autogen.beta.tools.code.environment.base import CodeLanguage

from .base import ExecResult, Sandbox

if TYPE_CHECKING:
    from autogen.beta.context import ConversationContext


def _run_subprocess(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    timeout: float,
    max_output: int,
    shell: bool,
) -> ExecResult:
    """Synchronous subprocess execution shared by :class:`LocalSandbox` and
    :class:`~autogen.beta.tools.shell.LocalShellEnvironment`.

    Centralises the timeout / truncation / exit-code conventions so both
    callers stay in sync.
    """
    merged_env = {**os.environ, **env} if env is not None else None

    if shell:
        cmd: str | list[str] = argv[0] if len(argv) == 1 else " ".join(argv)
    else:
        cmd = argv

    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=merged_env,
        )
    except FileNotFoundError as e:
        return ExecResult(output=f"Command not found: {e}", exit_code=127)
    except subprocess.TimeoutExpired:
        return ExecResult(
            output=f"Command timed out after {timeout}s",
            exit_code=124,
        )

    output = (result.stdout + result.stderr).strip()
    if (total := len(output)) > max_output:
        output = output[:max_output]
        output += f"\n[truncated: showing first {max_output} of {total} chars]"
    return ExecResult(output=output, exit_code=result.returncode or 0)


class LocalSandbox(Sandbox):
    """Sandbox backed by a local subprocess.

    A thin async wrapper around :func:`subprocess.run` (executed in a
    worker thread via :func:`asyncio.to_thread`).  Shares its execution
    helper with :class:`~autogen.beta.tools.shell.LocalShellEnvironment`
    so both paths apply the same timeout, truncation and exit-code
    conventions.

    Args:
        path: Working directory. ``None`` (default) creates a temporary
              directory with prefix ``"ag2_sandbox_"``.
        cleanup: Delete ``path`` on process exit. Defaults to ``True``
                 when ``path=None`` (auto temp dir) and ``False`` otherwise.
        timeout: Default per-call timeout in seconds. Overridable per
                 :meth:`exec` call.
        max_output: Maximum number of characters in :attr:`ExecResult.output`.
                    Excess is truncated with a trailing notice.
        languages: Code interpreters this sandbox declares as available.
                   Defaults to ``("python", "bash")`` — the minimum that
                   ships on most Unix hosts.  Set explicitly if you also
                   have ``node`` / ``ts-node`` installed.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        cleanup: bool | None = None,
        timeout: float = 60,
        max_output: int = 100_000,
        languages: tuple[CodeLanguage, ...] = ("python", "bash"),
    ) -> None:
        cleanup_flag = cleanup if cleanup is not None else (path is None)

        if path is None:
            tmpdir = tempfile.mkdtemp(prefix="ag2_sandbox_")
            self._workdir = Path(tmpdir)
        else:
            self._workdir = Path(path).resolve()
            self._workdir.mkdir(parents=True, exist_ok=True)

        if cleanup_flag:
            workdir_str = str(self._workdir)
            atexit.register(shutil.rmtree, workdir_str, ignore_errors=True)

        self._default_timeout = timeout
        self._max_output = max_output
        self._languages = tuple(languages)
        self._closed = False

    @property
    def workdir(self) -> Path:
        return self._workdir

    @property
    def supported_languages(self) -> tuple[CodeLanguage, ...]:
        return self._languages

    async def exec(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        shell: bool = False,
        context: "ConversationContext | None" = None,
    ) -> ExecResult:
        del context  # LocalSandbox has no Variable-capable params.
        if self._closed:
            raise RuntimeError("LocalSandbox has been closed.")
        if not argv:
            return ExecResult(output="", exit_code=2)

        return await asyncio.to_thread(
            _run_subprocess,
            argv,
            cwd=self._workdir,
            env=env,
            timeout=timeout if timeout is not None else self._default_timeout,
            max_output=self._max_output,
            shell=shell,
        )

    async def aclose(self) -> None:
        self._closed = True
