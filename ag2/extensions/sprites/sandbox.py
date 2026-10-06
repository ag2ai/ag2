# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any

from sprites import AsyncSprite
from sprites.exceptions import APIError, SpriteError

from ag2.tools.sandbox import ExecResult, SandboxBase

# This supervisor runs inside the Sprite, not on the agent's host. A client-side
# wait timeout alone cannot establish that the remote command was terminated.
_COMMAND_RUNNER = """
import functools
import os
import signal
import subprocess
import sys


def stop(pgid, signum, frame):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    os._exit(128 + signum)


def main():
    try:
        process = subprocess.Popen(sys.argv[2:], start_new_session=True)
    except FileNotFoundError:
        print("Command not found", file=sys.stderr)
        return 127
    except OSError:
        print("Command could not be started", file=sys.stderr)
        return 126
    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, functools.partial(stop, process.pid))
    try:
        code = process.wait(timeout=float(sys.argv[1]))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        print("Execution timed out", file=sys.stderr)
        return 124
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main())
"""


class SpritesSandbox(SandboxBase):
    """Execute commands in a caller-owned Sprite using the native async SDK.

    Construction, context exit, and :meth:`aclose` never create or delete a
    Sprite or close the supplied client's connections. Reuse the same Sprite
    for persistent files; commands start independent processes.

    The Sprite must have Python 3 available as ``python``. A supervisor inside
    the Sprite enforces command deadlines by killing the command's process
    group and forwards termination signals to that group. Workspace pyenv
    settings (including ``.python-version`` and ``PYENV_VERSION``) must select
    an installed Python even for shell commands. A transport failure instead raises an error with unknown execution
    status; it is never converted into a confirmed timeout or retried.

    Args:
        sprite: An existing async Sprite handle, owned by the caller.
        workdir: Existing absolute POSIX working directory inside the Sprite.
            This is not a filesystem security boundary.
        timeout: Default per-command deadline in seconds, positive and finite.
        max_output: Maximum returned characters of combined stdout and stderr.
            Keeps the end and adds a truncation marker outside this limit;
            does not bound SDK capture buffers.
        env_vars: Environment variables applied to every command. Per-call
            values override these. Host environment variables are not forwarded.
    """

    def __init__(
        self,
        sprite: AsyncSprite,
        *,
        workdir: str | os.PathLike[str] = "/home/sprite",
        timeout: float = 60,
        max_output: int = 100_000,
        env_vars: dict[str, str] | None = None,
    ) -> None:
        _validate_timeout(timeout)
        if not isinstance(max_output, int) or max_output <= 0:
            raise ValueError("max_output must be a positive integer")
        path = PurePosixPath(workdir)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("workdir must be an absolute POSIX path without '..'")
        self._sprite = sprite
        self._workdir = path
        self._timeout = timeout
        self._max_output = max_output
        self._env_vars = dict(env_vars or {})

    @property
    def workdir(self) -> PurePosixPath:
        """Working directory inside the Sprite."""
        return self._workdir

    @property
    def host_workdir(self) -> Path | None:
        """No host filesystem is mounted by this backend."""
        return None

    def __deepcopy__(self, memo: dict[int, Any]) -> "SpritesSandbox":
        # Tools are copied when attached to agents; the caller's resource is shared.
        return self

    async def exec(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Run literal arguments with a remote deadline; transport failures raise."""
        if not argv:
            return ExecResult(output="", exit_code=2)
        deadline = self._timeout if timeout is None else timeout
        _validate_timeout(deadline)
        try:
            result = await self._sprite.run(
                "python",
                "-c",
                _COMMAND_RUNNER,
                str(deadline),
                *argv,
                capture_output=True,
                check=False,
                cwd=str(self._workdir),
                env={**self._env_vars, **(env or {})},
                # Leave time for the remote supervisor to kill and reap the
                # command and deliver its exit status before abandoning transport.
                timeout=deadline + 10,
            )
        except (SpriteError, TimeoutError, asyncio.TimeoutError) as error:
            raise _request_error("execution", error) from None
        output = ((result.stdout or b"") + (result.stderr or b"")).decode("utf-8", errors="replace").strip()
        if len(output) > self._max_output:
            total = len(output)
            output = output[-self._max_output :] + f"\n[truncated: showing last {self._max_output} of {total} chars]"
        return ExecResult(output=output, exit_code=result.returncode)

    async def put_file(self, path: PurePosixPath, content: bytes) -> None:
        """Write bytes to a relative path using the Sprite filesystem API."""
        _validate_relative_file(path)
        try:
            await self._sprite.filesystem(str(self._workdir)).path(str(path)).write_bytes(content)
        except SpriteError as error:
            raise _request_error("file write", error) from None

    async def remove_file(self, path: PurePosixPath) -> None:
        """Remove a relative file, accepting an already missing file."""
        _validate_relative_file(path)
        try:
            await self._sprite.filesystem(str(self._workdir)).path(str(path)).unlink(missing_ok=True)
        except SpriteError as error:
            raise _request_error("file removal", error) from None


def _validate_timeout(timeout: float) -> None:
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")


def _validate_relative_file(path: PurePosixPath) -> None:
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError("File paths must be relative to workdir and cannot contain '..'")


def _request_error(operation: str, error: Exception) -> RuntimeError:
    logging.getLogger(__name__).warning("Sprite %s failed", operation, exc_info=True)
    status = f", HTTP {error.status_code}" if isinstance(error, APIError) and error.status_code is not None else ""
    return RuntimeError(
        f"Sprite {operation} status is unknown ({type(error).__name__}{status}); no retry was attempted."
    )
