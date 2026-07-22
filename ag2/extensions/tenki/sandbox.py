# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Adapt the Tenki Python SDK to AG2's sandbox protocol."""

import asyncio
import atexit
import logging
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any, cast

from tenki_sandbox import (
    AsyncClient,
    Client,
    CommandTimeoutError,
    SandboxError,
)
from tenki_sandbox import (
    FileNotFoundError as TenkiFileNotFoundError,
)

from ag2.annotations import Variable
from ag2.tools.sandbox import ExecResult, SandboxBase

logger = logging.getLogger(__name__)


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
        self._sandbox: Any = None
        self._ready = False
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._atexit_registered = False
        self._sync_auth_token = cast(str | None, getattr(client, "auth_token", None))
        self._sync_base_url = cast(str | None, getattr(client, "base_url", None))

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

    def __deepcopy__(self, memo: dict) -> "TenkiSandbox":  # type: ignore[type-arg]
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

        output = result.stdout_text + result.stderr_text
        if result.reason and not output:
            output = f"Tenki execution ended: {result.reason}"
        if result.signal:
            output += f"\nTenki execution signal: {result.signal}"
        exit_code = result.exit_code if not result.signal else result.exit_code or 128
        return ExecResult(output=output, exit_code=exit_code)

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

    async def _ensure_sandbox(self) -> Any:
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
            if not options.get("project_id"):
                options["project_id"] = await self._resolve_project_id()

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

    async def _resolve_project_id(self) -> str:
        if self._client is None:
            raise RuntimeError("TenkiSandbox client has been closed.")
        identity = await self._client.who_am_i()
        projects = [project for workspace in identity.workspaces for project in workspace.projects]
        if len(projects) == 1:
            return projects[0].id
        if not projects:
            raise RuntimeError("The Tenki API key has no visible project. Create a project before opening a sandbox.")
        raise RuntimeError("The Tenki API key can access multiple projects. Pass `project_id` to TenkiEnvironment.")

    async def aclose(self) -> None:
        self._unregister_atexit()
        self._closed = True
        error: BaseException | None = None
        if self._sandbox is not None:
            try:
                await self._sandbox.close_if_open()
            except BaseException as e:
                error = e
            else:
                self._sandbox = None
                self._ready = False
        if self._client is not None:
            try:
                await self._client.close()
            except BaseException as e:
                error = error or e
            self._client = None
        if error is not None:
            raise error

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
