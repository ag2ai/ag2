# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import atexit
import os
import shutil
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path, PurePosixPath

from ag2.tools.code.environment.base import CodeLanguage

from .base import ExecResult, SandboxBase


def _resolve_workdir_path(path: PurePosixPath, host_workdir: Path) -> Path:
    """Resolve ``path`` (relative to the sandbox workdir) on the host.

    Absolute paths are rejected so writes stay confined to ``host_workdir``.
    Symlink escapes are guarded by resolving and checking containment.
    """
    if path.is_absolute():
        raise ValueError(f"Absolute paths are not allowed in put_file/get_file: {path}")
    resolved = (host_workdir / Path(*path.parts)).resolve()
    host_resolved = host_workdir.resolve()
    if not resolved.is_relative_to(host_resolved):
        raise ValueError(f"Path escapes sandbox workdir: {path}")
    return resolved


def _run_subprocess(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    timeout: float,
    max_output: int,
) -> ExecResult:
    """Run *argv* under :func:`subprocess.run` with the project's conventions."""
    merged_env = {**os.environ, **env} if env is not None else None

    try:
        result = subprocess.run(
            argv,
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


class LocalSandbox(SandboxBase):
    """Sandbox backed by a local subprocess.

    A thin wrapper around :func:`subprocess.run` with a fixed working
    directory. ``__aenter__`` / ``__aexit__`` are no-ops — local sandboxes
    are usable immediately after construction.

    Args:
        path: Working directory. ``None`` (default) creates a temporary
              directory with prefix ``"ag2_sandbox_"``.
        cleanup: Delete ``path`` on :meth:`aclose` (and as an atexit
                 backstop). Defaults to ``True`` when ``path=None``
                 (auto temp dir) and ``False`` otherwise.
        timeout: Default per-call timeout in seconds. Must be ``> 0``.
                 Overridable per :meth:`exec` call.
        max_output: Maximum number of characters in :attr:`ExecResult.output`.
                    Excess is truncated with a trailing notice.
        env_vars: Default environment variables merged into every command
                  (on top of the inherited parent environment). Per-call
                  ``env`` values passed to :meth:`exec` take precedence.
        languages: Languages this sandbox advertises via
                   :attr:`supported_languages` — informational only;
                   the canonical language matrix lives on :class:`CodeAdapter`.
                   Defaults to ``("python", "bash")``.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        cleanup: bool | None = None,
        timeout: float = 60,
        max_output: int = 100_000,
        env_vars: dict[str, str] | None = None,
        languages: tuple[CodeLanguage, ...] = ("python", "bash"),
    ) -> None:
        if timeout <= 0:
            raise ValueError("`timeout` must be > 0 seconds.")

        self._path = os.fspath(path) if path is not None else None
        self._cleanup_workdir = cleanup if cleanup is not None else (path is None)

        # Workdir is materialized lazily on first runtime use (see
        # _ensure_workdir): no filesystem side effects at construction time.
        self._host_workdir: Path | None = None
        self._workdir: PurePosixPath | None = None
        self._atexit_registered = False

        self._default_timeout = timeout
        self._max_output = max_output
        self._env_vars = dict(env_vars) if env_vars else {}
        self._languages = tuple(languages)
        self._closed = False

    def _ensure_workdir(self) -> Path:
        """Materialize the working directory on first use.

        Runs synchronously without awaiting, so concurrent first calls in
        the same event loop cannot interleave inside it. Idempotent.
        """
        if self._host_workdir is not None:
            return self._host_workdir

        if self._path is None:
            host_workdir = Path(tempfile.mkdtemp(prefix="ag2_sandbox_"))
        else:
            host_workdir = Path(self._path).resolve()
            host_workdir.mkdir(parents=True, exist_ok=True)

        self._host_workdir = host_workdir
        self._workdir = PurePosixPath(host_workdir.as_posix())
        if self._cleanup_workdir and not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True
        return host_workdir

    def _merge_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        """Merge the sandbox's default ``env_vars`` with a per-call ``env``.

        Per-call values win. Returns ``None`` when there is nothing to add,
        so the subprocess inherits the parent environment unchanged.
        """
        if not self._env_vars and env is None:
            return None
        return {**self._env_vars, **(env or {})}

    @property
    def workdir(self) -> PurePosixPath:
        self._ensure_workdir()
        assert self._workdir is not None
        return self._workdir

    @property
    def host_workdir(self) -> Path:
        return self._ensure_workdir()

    @property
    def supported_languages(self) -> tuple[CodeLanguage, ...]:
        return self._languages

    async def exec(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        if self._closed:
            raise RuntimeError("LocalSandbox has been closed.")
        if not argv:
            return ExecResult(output="", exit_code=2)

        host_workdir = self._ensure_workdir()
        return await asyncio.to_thread(
            _run_subprocess,
            argv,
            cwd=host_workdir,
            env=self._merge_env(env),
            timeout=timeout if timeout is not None else self._default_timeout,
            max_output=self._max_output,
        )

    async def put_file(self, path: PurePosixPath, content: bytes) -> None:
        if self._closed:
            raise RuntimeError("LocalSandbox has been closed.")
        target = _resolve_workdir_path(path, self._ensure_workdir())
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, content)

    async def remove_file(self, path: PurePosixPath) -> None:
        if self._closed:
            raise RuntimeError("LocalSandbox has been closed.")
        target = _resolve_workdir_path(path, self._ensure_workdir())
        await asyncio.to_thread(target.unlink, True)  # missing_ok=True

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cleanup_workdir_now()

    def _cleanup_workdir_now(self) -> None:
        if self._atexit_registered:
            with suppress(ValueError):
                atexit.unregister(self._atexit_cleanup)
            self._atexit_registered = False
        if self._cleanup_workdir and self._host_workdir is not None:
            shutil.rmtree(self._host_workdir, ignore_errors=True)
            self._cleanup_workdir = False

    def _atexit_cleanup(self) -> None:
        if self._cleanup_workdir and self._host_workdir is not None:
            shutil.rmtree(self._host_workdir, ignore_errors=True)
