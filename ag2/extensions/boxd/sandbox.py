# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Adapt the boxd Python SDK to AG2's sandbox protocol."""

import asyncio
import atexit
import logging
import re
from pathlib import Path, PurePosixPath
from typing import Any

from boxd import AsyncBoxd, Boxd, BoxdError, NotFoundError

from ag2.annotations import Variable
from ag2.tools.sandbox import ExecResult, SandboxBase

logger = logging.getLogger(__name__)

# Environment variable names must be plain shell identifiers.
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _is_timeout(error: BoxdError) -> bool:
    """Whether the SDK gave up on an exec because its ``timeout`` ran out.

    Given a ``timeout=``, the SDK turns a spent deadline into a plain
    ``BoxdError`` with no status code ("exec timed out after 3s"); the server
    has already killed the remote process group by then.
    """
    return error.code is None and str(error).startswith("exec timed out")


def check_env_names(env: dict[str, str] | None) -> None:
    """Raise ``ValueError`` unless every name in ``env`` is a shell identifier."""
    for key in env or {}:
        if not _ENV_NAME.match(key):
            raise ValueError(f"Invalid environment variable name for a boxd sandbox: {key!r}")


def _text(value: str | bytes) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


class BoxdSandbox(SandboxBase):
    """Sandbox backed by a boxd machine (a KVM microVM).

    Creation is lazy and failure-atomic after the SDK returns a machine ID:
    readiness errors and cancellation delete the new machine. A delete that
    fails keeps the machine handle, so :meth:`aclose`, the next creation
    attempt and the atexit fallback can retry it. A finite ``delete_after`` in
    ``create_options`` remains the server-side backstop.

    Use one instance from one event loop at a time; creation and close are
    serialized per loop.

    ``env`` is sent with every command, under the per-call ``env`` of
    :meth:`exec`.
    """

    def __init__(
        self,
        *,
        client: AsyncBoxd,
        create_options: dict[str, Any],
        timeout: float = 60,
        workdir: str = "/home/boxd",
        env: dict[str, str] | None = None,
        api_key: str | None = None,  # pragma: allowlist secret
    ) -> None:
        for name, value in (("client", client), ("create_options", create_options)):
            if isinstance(value, Variable):
                raise TypeError(
                    f"BoxdSandbox.{name} must be a concrete value; got Variable. "
                    "Wrap with BoxdEnvironment to resolve Variables from a Context."
                )
        if timeout <= 0:
            raise ValueError("`timeout` must be greater than 0 seconds.")
        check_env_names(env)

        self._client: AsyncBoxd | None = client
        self._create_options = create_options
        self._default_timeout = timeout
        self._workdir = PurePosixPath(workdir)
        self._env = dict(env or {})
        self._machine_id: str | None = None
        self._ready = False
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._atexit_registered = False
        # Captured up front: the atexit fallback runs after `aclose` has already
        # dropped the async client, and has to build a fresh sync one. A `None`
        # key makes that client read `BOXD_API_KEY`, as the async one did.
        self._sync_api_key = api_key
        self._sync_base_url = client.base_url
        self._sync_region = client.region

    @property
    def workdir(self) -> PurePosixPath:
        return self._workdir

    @property
    def host_workdir(self) -> Path | None:
        return None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def machine_id(self) -> str | None:
        """ID of the boxd machine, once it has been created."""
        return self._machine_id

    def _creation_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def __aenter__(self) -> "BoxdSandbox":
        await self._ensure_sandbox()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __deepcopy__(self, memo: dict[int, Any]) -> "BoxdSandbox":
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
        check_env_names(env)
        merged_env = {**self._env, **(env or {})}

        client, machine_id = await self._ensure_sandbox()
        exec_timeout = timeout if timeout is not None else self._default_timeout
        try:
            result = await client.machines.exec(
                machine_id,
                argv,
                env=merged_env or None,
                cwd=str(self._workdir),
                timeout=exec_timeout,
            )
        except BoxdError as e:
            if _is_timeout(e):
                return ExecResult(output=f"boxd execution timed out after {exec_timeout}s", exit_code=124)
            return ExecResult(output=f"boxd error: {e}", exit_code=1)

        output = (_text(result.stdout) + _text(result.stderr)).strip()
        # Commands run through the machine's shell, so the exit code already
        # follows POSIX: 127 not found, 126 not executable, 128+N for signal N.
        # A negative code means the stream ended without reporting one.
        if result.exit_code < 0:
            note = "boxd execution ended without an exit status"
            return ExecResult(output=f"{output}\n{note}" if output else note, exit_code=1)
        return ExecResult(output=output, exit_code=result.exit_code)

    async def put_file(self, path: PurePosixPath, content: bytes) -> None:
        if path.is_absolute():
            raise ValueError(f"Absolute paths are not allowed in put_file: {path}")
        client, machine_id = await self._ensure_sandbox()
        # `fs.write` creates missing parent directories.
        await client.machines.fs.write(machine_id, str(self._workdir / path), content)

    async def _ensure_sandbox(self) -> tuple[AsyncBoxd, str]:
        if self._closed:
            raise RuntimeError("BoxdSandbox has been closed.")
        if self._client is not None and self._machine_id is not None and self._ready:
            return self._client, self._machine_id

        async with self._creation_lock():
            if self._closed:
                raise RuntimeError("BoxdSandbox has been closed.")
            if self._client is None:
                raise RuntimeError("BoxdSandbox client has been closed.")
            client = self._client
            if self._machine_id is not None and self._ready:
                return client, self._machine_id
            # A machine left behind by a failed setup must be gone before a new one exists.
            if self._machine_id is not None and not await self._delete():
                raise RuntimeError("A boxd machine from a failed setup could not be deleted; retry later.")

            machine = await client.machines.create(**self._create_options)
            self._machine_id = machine.id
            self._register_atexit()
            try:
                await client.machines.wait_until_ready(machine.id, timeout=self._default_timeout)
                # Commands `cd` into the workdir first, and fail if it is missing.
                prepared = await client.machines.exec(
                    machine.id,
                    ["mkdir", "-p", str(self._workdir)],
                    timeout=self._default_timeout,
                )
                if prepared.exit_code != 0:
                    detail = (_text(prepared.stdout) + _text(prepared.stderr)).strip()
                    raise RuntimeError(f"Could not create workdir {self._workdir} in boxd machine: {detail}")
            except BaseException:
                # On failure the handle stays for `aclose`, the next attempt
                # and the atexit fallback to retry.
                await asyncio.shield(self._delete())
                raise

            self._ready = True
            logger.info("boxd machine created (id=%s, name=%s)", machine.id, machine.name)
            return client, machine.id

    async def aclose(self) -> None:
        """Delete the machine and release the client.

        Safe to call repeatedly. Waits for a creation in progress on the same
        event loop, so a machine that is still being created is deleted too. A
        delete that fails keeps both the machine handle and the client so a
        later call can retry; the atexit fallback stays armed for the case
        where no retry comes.
        """
        self._closed = True
        async with self._creation_lock():
            if self._machine_id is not None and not await self._delete():
                return
            if self._client is not None:
                try:
                    await self._client.close()
                except Exception as e:
                    logger.debug("Suppressed exception during boxd client close: %s", e)
                self._client = None

    async def _delete(self) -> bool:
        """Delete the machine this sandbox owns; whether it is gone afterwards."""
        machine_id = self._machine_id
        assert machine_id is not None and self._client is not None
        try:
            await self._client.machines.delete(machine_id)
        except NotFoundError:
            pass  # already gone, e.g. removed by its `delete_after` timer
        except Exception as e:
            logger.debug("Suppressed exception during boxd machine delete: %s", e)
            return False
        # A shielded delete can finish after a later call already replaced the handle.
        if self._machine_id == machine_id:
            self._machine_id = None
            self._ready = False
            self._unregister_atexit()
        return True

    def _register_atexit(self) -> None:
        if not self._atexit_registered:
            atexit.register(self._atexit_close)
            self._atexit_registered = True

    def _unregister_atexit(self) -> None:
        if self._atexit_registered:
            atexit.unregister(self._atexit_close)
            self._atexit_registered = False

    def _atexit_close(self) -> None:
        if self._machine_id is None:
            return
        try:
            with Boxd(api_key=self._sync_api_key, base_url=self._sync_base_url, region=self._sync_region) as client:
                client.machines.delete(self._machine_id)
        except Exception as e:
            logger.debug("Suppressed exception during atexit boxd machine cleanup: %s", e)
