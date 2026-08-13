# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Adapt the Tenki Python SDK to AG2's sandbox protocol."""

import asyncio
import atexit
import errno
import logging
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any

from tenki import (
    AsyncClient,
    AsyncSandbox,
    Client,
    CommandResult,
    CommandTimeoutError,
    SandboxError,
)
from tenki import (
    FileNotFoundError as TenkiFileNotFoundError,
)

from ag2.annotations import Variable
from ag2.tools.sandbox import ExecResult, SandboxBase

logger = logging.getLogger(__name__)


def _posix_exit_code(result: CommandResult) -> int:
    """Map a Tenki result onto the POSIX codes ``ExecResult`` promises.

    Tenki reports ``exit_code == -1`` whenever the command produced no real
    wait status — it timed out, was signalled, or could never be exec'd. A
    negative code is not a POSIX status and means nothing to a model reading
    tool output, so translate it the way a shell would.
    """
    if result.exit_code >= 0:
        return result.exit_code
    # A timeout is also reported with signal="terminated", so it has to be read
    # before the signal case, or every timeout would surface as a plain kill.
    if result.reason == "timeout":
        return 124
    if result.signal:
        # 128 is the base a shell adds a signal number to. Tenki reports the
        # signal by name ("killed", "terminated"), so the number isn't available
        # to add — the name stays in `output` instead.
        return 128
    # Anything left never started. Tenki describes those in prose and fills
    # `errno` only sometimes — EACCES arrives with it set, ENOENT with it
    # unset — so both have to be consulted.
    reason = result.reason or ""
    if result.errno == errno.EACCES or "permission denied" in reason:
        return 126  # found, but not executable
    if result.errno == errno.ENOENT or "executable file not found" in reason:
        return 127
    return 1


class TenkiSandbox(SandboxBase):
    """Sandbox backed by a Tenki managed cloud sandbox.

    Creation is lazy and failure-atomic after the SDK returns a session ID:
    readiness errors and cancellation terminate the new session. A finite
    ``max_duration`` in ``create_options`` remains the server-side backstop.
    """

    def __init__(
        self,
        *,
        client: AsyncClient,
        create_options: dict[str, Any],
        timeout: float = 60,
        workdir: str = "/home/tenki",
    ) -> None:
        for name, value in (("client", client), ("create_options", create_options)):
            if isinstance(value, Variable):
                raise TypeError(
                    f"TenkiSandbox.{name} must be a concrete value; got Variable. "
                    "Wrap with TenkiEnvironment to resolve Variables from a Context."
                )
        if timeout <= 0:
            raise ValueError("`timeout` must be greater than 0 seconds.")

        self._client: AsyncClient | None = client
        self._create_options = create_options
        self._default_timeout = timeout
        self._workdir = PurePosixPath(workdir)
        self._sandbox: AsyncSandbox | None = None
        self._ready = False
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._atexit_registered = False
        # Captured up front: the atexit fallback runs after `aclose` has already
        # dropped the async client, and has to build a fresh sync one.
        self._sync_auth_token = client.auth_token
        self._sync_base_url = client.base_url

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

    async def __aenter__(self) -> "TenkiSandbox":
        await self._ensure_sandbox()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __deepcopy__(self, memo: dict[int, Any]) -> "TenkiSandbox":
        return self

    async def exec(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        if not argv:
            return ExecResult(output="", exit_code=2)

        sandbox = await self._ensure_sandbox()
        exec_timeout = timeout if timeout is not None else self._default_timeout
        try:
            result = await sandbox.exec(
                *argv,
                cwd=str(self._workdir),
                env=env,
                timeout=exec_timeout,
            )
        except (CommandTimeoutError, TimeoutError) as e:
            return ExecResult(output=f"Tenki execution timed out: {e}", exit_code=124)
        except SandboxError as e:
            return ExecResult(output=f"Tenki error: {e}", exit_code=1)

        output = (result.stdout_text + result.stderr_text).strip()
        if result.reason == "timeout":
            note = f"Tenki execution timed out after {exec_timeout}s"
            output = f"{output}\n{note}" if output else note
        else:
            if result.reason and not output and not result.ok:
                output = f"Tenki execution ended: {result.reason}"
            if result.signal:
                output += f"\nTenki execution signal: {result.signal}"
        return ExecResult(output=output, exit_code=_posix_exit_code(result))

    async def put_file(self, path: PurePosixPath, content: bytes) -> None:
        if path.is_absolute():
            raise ValueError(f"Absolute paths are not allowed in put_file: {path}")
        sandbox = await self._ensure_sandbox()
        await sandbox.fs.write_bytes(str(self._workdir / path), content)

    async def remove_file(self, path: PurePosixPath) -> None:
        if path.is_absolute():
            raise ValueError(f"Absolute paths are not allowed in remove_file: {path}")
        sandbox = await self._ensure_sandbox()
        with suppress(TenkiFileNotFoundError):
            await sandbox.fs.remove(str(self._workdir / path), recursive=False)

    async def _ensure_sandbox(self) -> AsyncSandbox:
        if self._closed:
            raise RuntimeError("TenkiSandbox has been closed.")
        if self._sandbox is not None and self._ready:
            return self._sandbox

        async with self._creation_lock():
            if self._closed:
                raise RuntimeError("TenkiSandbox has been closed.")
            if self._sandbox is not None and self._ready:
                return self._sandbox
            if self._client is None:
                raise RuntimeError("TenkiSandbox client has been closed.")

            options = dict(self._create_options)
            if not options.get("workspace_id"):
                options["workspace_id"] = await self._resolve_workspace_id()

            sandbox = await self._client.create(wait=False, **options)
            self._sandbox = sandbox
            self._register_atexit()
            try:
                if sandbox.state != "RUNNING":
                    await sandbox.wait_ready(self._default_timeout)
            except BaseException:
                try:
                    await asyncio.shield(sandbox.close_if_open())
                except BaseException as cleanup_error:
                    logger.debug("Failed to terminate Tenki sandbox after create error: %s", cleanup_error)
                self._sandbox = None
                self._unregister_atexit()
                raise

            self._ready = True
            logger.info("Tenki sandbox created (id=%s)", sandbox.id)
            return sandbox

    async def _resolve_workspace_id(self) -> str:
        if self._client is None:
            raise RuntimeError("TenkiSandbox client has been closed.")
        identity = await self._client.who_am_i()
        workspaces = list(identity.workspaces)
        if len(workspaces) == 1:
            return workspaces[0].id
        if not workspaces:
            raise RuntimeError(
                "The Tenki API key has no visible workspace. Create a workspace before opening a sandbox."
            )
        raise RuntimeError("The Tenki API key can access multiple workspaces. Pass `workspace_id` to TenkiEnvironment.")

    async def aclose(self) -> None:
        """Terminate the session and release the client.

        Safe to call repeatedly. A close that fails leaves both the session
        handle and the client intact so a later call can retry; the atexit
        fallback stays armed for the case where no retry comes.
        """
        self._closed = True
        terminated = True
        if self._sandbox is not None:
            try:
                await self._sandbox.close_if_open()
            except Exception as e:
                terminated = False
                logger.debug("Suppressed exception during Tenki sandbox close: %s", e)
            else:
                self._sandbox = None
                self._ready = False
        # Only disarm the atexit fallback once the session is really gone: a failed
        # close leaves a live sandbox that still deserves a last attempt at exit.
        if not terminated:
            # Keep the client open too. The sandbox handle routes through it, so
            # closing it here would leave the retained handle unusable and turn
            # every later `aclose` into a silent no-op.
            return
        self._unregister_atexit()
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as e:
                logger.debug("Suppressed exception during Tenki client close: %s", e)
            self._client = None

    def _register_atexit(self) -> None:
        if not self._atexit_registered:
            atexit.register(self._atexit_close)
            self._atexit_registered = True

    def _unregister_atexit(self) -> None:
        if self._atexit_registered:
            atexit.unregister(self._atexit_close)
            self._atexit_registered = False

    def _atexit_close(self) -> None:
        if self._sandbox is None:
            return
        try:
            with Client(auth_token=self._sync_auth_token, base_url=self._sync_base_url) as client:
                client.get(self._sandbox.id).close_if_open()
        except Exception as e:
            logger.debug("Suppressed exception during atexit Tenki sandbox cleanup: %s", e)
