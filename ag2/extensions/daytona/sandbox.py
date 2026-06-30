# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import atexit
import logging
import shlex
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any

from daytona import (
    AsyncDaytona,
    DaytonaError,
    DaytonaNotFoundError,
    DaytonaRateLimitError,
    DaytonaTimeoutError,
)

from ag2.annotations import Variable
from ag2.tools.sandbox import ExecResult, SandboxBase

logger = logging.getLogger(__name__)


class DaytonaSandbox(SandboxBase):
    """Sandbox backed by a Daytona managed cloud sandbox.

    All shaping parameters are concrete values — :class:`Variable`
    resolution lives on :class:`DaytonaEnvironment`.

    Lifecycle: the sandbox is created on ``__aenter__`` and torn down
    on ``__aexit__`` / :meth:`aclose`. Lazy creation on first call is
    retained as a safety net.
    """

    def __init__(
        self,
        *,
        client: AsyncDaytona,
        params: Any,
        timeout: int = 60,
        workdir: str = "/workspace",
    ) -> None:
        for name, value in (("client", client), ("params", params)):
            if isinstance(value, Variable):
                raise TypeError(
                    f"DaytonaSandbox.{name} must be a concrete value; got Variable. "
                    "Wrap with DaytonaEnvironment to resolve Variables from a Context."
                )

        if timeout < 1:
            raise ValueError("`timeout` must be >= 1 second.")

        self._client = client
        self._params = params
        self._default_timeout = timeout
        self._workdir = PurePosixPath(workdir)
        self._sandbox: Any = None
        # See DockerSandbox for the cross-loop rationale. Daytona's create is
        # async (can't be pushed to a worker thread cleanly), so we keep a
        # per-loop asyncio.Lock that only guards the one-time creation; once
        # the sandbox exists, callers short-circuit before ever touching it.
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._atexit_registered = False

    @property
    def workdir(self) -> PurePosixPath:
        return self._workdir

    @property
    def host_workdir(self) -> Path | None:
        return None

    @property
    def closed(self) -> bool:
        return self._closed

    def _creation_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def __aenter__(self) -> "DaytonaSandbox":
        await self._ensure_sandbox()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __deepcopy__(self, memo: dict) -> "DaytonaSandbox":  # type: ignore[type-arg]
        # A live cloud-sandbox handle holding loop-bound async state — not
        # deepcopy-able and not meaningfully duplicable. Sharing on copy is the
        # only sane semantics (lets a bare DaytonaSandbox be attached to a tool).
        return self

    async def exec(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        del env  # Daytona's process.exec does not accept per-call env.
        if not argv:
            return ExecResult(output="", exit_code=2)

        sandbox = await self._ensure_sandbox()
        cmd = shlex.join(argv)
        exec_timeout = int(timeout) if timeout is not None else self._default_timeout
        try:
            response = await sandbox.process.exec(cmd, timeout=exec_timeout)
        except DaytonaTimeoutError as e:
            return ExecResult(output=f"Execution timed out: {e}", exit_code=124)
        except DaytonaRateLimitError as e:
            return ExecResult(output=f"Daytona rate limit exceeded: {e}", exit_code=1)
        except DaytonaError as e:
            return ExecResult(output=f"Daytona error: {e}", exit_code=1)

        return ExecResult(output=response.result or "", exit_code=response.exit_code or 0)

    async def put_file(self, path: PurePosixPath, content: bytes) -> None:
        if path.is_absolute():
            raise ValueError(f"Absolute paths are not allowed in put_file/get_file: {path}")
        sandbox = await self._ensure_sandbox()
        target = self._workdir / path
        await sandbox.fs.upload_file(content, str(target))

    async def remove_file(self, path: PurePosixPath) -> None:
        if path.is_absolute():
            raise ValueError(f"Absolute paths are not allowed in remove_file: {path}")
        sandbox = await self._ensure_sandbox()
        target = self._workdir / path
        with suppress(DaytonaNotFoundError):
            await sandbox.fs.delete_file(str(target))

    async def _ensure_sandbox(self) -> Any:
        if self._closed:
            raise RuntimeError("DaytonaSandbox has been closed.")
        if self._sandbox is not None:
            return self._sandbox
        async with self._creation_lock():
            if self._closed:
                raise RuntimeError("DaytonaSandbox has been closed.")
            if self._sandbox is not None:
                return self._sandbox
            self._sandbox = await self._client.create(self._params)
            if not self._atexit_registered:
                atexit.register(self._atexit_close)
                self._atexit_registered = True
            logger.info("Daytona sandbox created (id=%s)", getattr(self._sandbox, "id", "?"))
            return self._sandbox

    async def aclose(self) -> None:
        if self._atexit_registered:
            atexit.unregister(self._atexit_close)
            self._atexit_registered = False
        self._closed = True
        if self._sandbox is not None:
            try:
                await self._sandbox.delete()
            except DaytonaNotFoundError:
                pass
            except Exception as e:
                logger.debug("Suppressed exception during sandbox deletion: %s", e)
            self._sandbox = None
        if self._client is not None:
            try:
                await self._client.close()  # type: ignore[no-untyped-call]
            except Exception as e:
                logger.debug("Suppressed exception during client close: %s", e)
            self._client = None  # type: ignore[assignment]

    def _atexit_close(self) -> None:
        if self._sandbox is None:
            return
        try:
            asyncio.run(self.aclose())
        except Exception as e:
            logger.debug("Suppressed exception during atexit sandbox cleanup: %s", e)
