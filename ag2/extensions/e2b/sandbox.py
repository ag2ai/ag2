# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Adapt the E2B Python SDK to AG2's sandbox protocol."""

import asyncio
import atexit
import logging
import math
import shlex
import time
from pathlib import Path, PurePosixPath
from typing import Any

from e2b import (
    AsyncSandbox,
    CommandExitException,
    NotFoundException,
    SandboxException,
    TimeoutException,
)
from e2b import Sandbox as SyncSandbox
from e2b.connection_config import ApiParams

from ag2.annotations import Variable
from ag2.tools.sandbox import ExecResult, SandboxBase

logger = logging.getLogger(__name__)

# Headroom kept on the server-side lifetime beyond a running command's own
# timeout, so a command never outlives the sandbox it runs in.
_LIFETIME_GRACE = 30

_API_PARAMS = frozenset(ApiParams.__annotations__)
_RESERVED_CREATE_OPTIONS = frozenset({"template", "timeout", "envs"})


class E2BSandbox(SandboxBase):
    """Sandbox backed by an E2B cloud sandbox.

    All parameters are concrete values; :class:`Variable` resolution lives on
    :class:`~ag2.extensions.e2b.E2BEnvironment`.

    The sandbox is created on ``__aenter__`` (or lazily on first use) and
    killed on ``__aexit__`` / :meth:`aclose`. E2B reclaims a sandbox once its
    server-side lifetime runs out, which bounds a leak when client cleanup
    never runs. While the sandbox is in use, every command extends that
    lifetime once less than half of it is left, and always far enough to
    cover the command's own timeout; the lifetime is never shortened. An idle
    sandbox therefore lives at least half of ``sandbox_timeout`` after its
    last command. A sandbox that is gone is reported as an error and is never
    silently recreated, since a new one would not have its files.

    Args:
        template: E2B template name or ID. ``None`` uses the SDK default.
        env_vars: Environment variables set for every command.
        timeout: Default per-command timeout in seconds.
        sandbox_timeout: Server-side lifetime in seconds that commands keep
            extending. ``None`` uses the SDK default.
        workdir: Sandbox-side directory commands and relative paths resolve
            against. Created on startup if it does not exist.
        create_options: Extra keyword arguments forwarded to
            ``AsyncSandbox.create`` (for example ``api_key``, ``domain``,
            ``metadata``, ``secure``, ``allow_internet_access``, ``network``
            or ``lifecycle``).
    """

    def __init__(
        self,
        *,
        template: str | None = None,
        env_vars: dict[str, str] | None = None,
        timeout: float = 60,
        sandbox_timeout: int | None = None,
        workdir: str = "/home/user",
        create_options: dict[str, Any] | None = None,
    ) -> None:
        for name, value in (("template", template), ("env_vars", env_vars), ("create_options", create_options)):
            if isinstance(value, Variable):
                raise TypeError(
                    f"E2BSandbox.{name} must be a concrete value; got Variable. "
                    "Wrap with E2BEnvironment to resolve Variables from a Context."
                )
        if timeout <= 0:
            raise ValueError("`timeout` must be greater than 0 seconds.")
        if sandbox_timeout is not None and sandbox_timeout <= 0:
            raise ValueError("`sandbox_timeout` must be greater than 0 seconds.")
        reserved = _RESERVED_CREATE_OPTIONS.intersection(create_options or {})
        if reserved:
            raise ValueError(
                f"Pass {sorted(reserved)} as E2BSandbox arguments (template, env_vars, sandbox_timeout), "
                "not through `create_options`."
            )

        self._template = template
        self._env_vars = env_vars
        self._default_timeout = timeout
        self._lifetime = sandbox_timeout if sandbox_timeout is not None else AsyncSandbox.default_sandbox_timeout
        self._workdir = PurePosixPath(workdir)
        self._create_options = dict(create_options or {})
        # Connection settings for the sync atexit fallback, which cannot reuse
        # the async handle once the event loop that owns it is gone.
        self._api_params = {k: v for k, v in self._create_options.items() if k in _API_PARAMS}

        # Set as soon as E2B returns a sandbox, so cleanup can reach it; handed
        # to callers only once `_ready`, after setup succeeded.
        self._sandbox: AsyncSandbox | None = None
        self._ready = False
        self._expires_at = 0.0
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

    @property
    def sandbox_id(self) -> str | None:
        """ID of the E2B sandbox this object still owns, or ``None`` once it is killed."""
        return self._sandbox.sandbox_id if self._sandbox is not None else None

    def _loop_lock(self) -> asyncio.Lock:
        # Serializes creation, lifetime refreshes and close on one event loop.
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def __aenter__(self) -> "E2BSandbox":
        await self._ensure_sandbox()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __deepcopy__(self, memo: dict[int, Any]) -> "E2BSandbox":
        # A live cloud-sandbox handle: sharing on copy is the only sane semantics.
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
            await self._keep_alive(sandbox, exec_timeout)
            result = await sandbox.commands.run(
                shlex.join(argv),
                envs=env,
                cwd=str(self._workdir),
                timeout=exec_timeout,
            )
        except CommandExitException as e:
            return ExecResult(output=(e.stdout + e.stderr).strip(), exit_code=e.exit_code)
        except TimeoutException as e:
            # E2B also reports a sandbox that is gone as a timeout, so ask
            # before telling the model its command merely ran too long.
            try:
                running = await sandbox.is_running()
            except Exception as check_error:
                return ExecResult(
                    output=f"E2B execution timed out after {exec_timeout}s, and the sandbox state "
                    f"could not be checked: {check_error}",
                    exit_code=1,
                )
            if running:
                return ExecResult(output=f"E2B execution timed out after {exec_timeout}s", exit_code=124)
            return _gone(e)
        except NotFoundException as e:
            return _gone(e)
        except SandboxException as e:
            return ExecResult(output=f"E2B error: {e}", exit_code=1)

        return ExecResult(output=(result.stdout + result.stderr).strip(), exit_code=result.exit_code)

    async def put_file(self, path: PurePosixPath, content: bytes) -> None:
        if path.is_absolute():
            raise ValueError(f"Absolute paths are not allowed in put_file: {path}")
        sandbox = await self._ensure_sandbox()
        await sandbox.files.write(str(self._workdir / path), content)

    async def remove_file(self, path: PurePosixPath) -> None:
        if path.is_absolute():
            raise ValueError(f"Absolute paths are not allowed in remove_file: {path}")
        sandbox = await self._ensure_sandbox()
        # E2B's remove is already a no-op for a missing path.
        await sandbox.files.remove(str(self._workdir / path))

    async def _keep_alive(self, sandbox: AsyncSandbox, exec_timeout: float) -> None:
        """Extend the server-side lifetime when it would lapse soon.

        Refreshes at most once per half lifetime, and always when the command
        about to run could outlast what is left. E2B's ``set_timeout`` can also
        shorten a lifetime, so refreshes are serialized and only ever move the
        deadline later, which keeps it covering every command already running.
        """
        needed = math.ceil(exec_timeout) + _LIFETIME_GRACE
        if self._expires_at - time.monotonic() > max(needed, self._lifetime / 2):
            return
        async with self._loop_lock():
            now = time.monotonic()
            if self._expires_at - now > max(needed, self._lifetime / 2):
                return
            lifetime = max(self._lifetime, needed)
            await sandbox.set_timeout(lifetime)
            self._expires_at = now + lifetime

    async def _ensure_sandbox(self) -> AsyncSandbox:
        if self._closed:
            raise RuntimeError("E2BSandbox has been closed.")
        if self._ready and self._sandbox is not None:
            return self._sandbox

        async with self._loop_lock():
            if self._closed:
                raise RuntimeError("E2BSandbox has been closed.")
            if self._ready and self._sandbox is not None:
                return self._sandbox
            # A sandbox left behind by a failed setup must be gone before a new one exists.
            if self._sandbox is not None and not await self._kill():
                raise RuntimeError("An E2B sandbox from a failed setup could not be killed; retry later.")

            started = time.monotonic()
            sandbox = await AsyncSandbox.create(
                self._template,
                timeout=self._lifetime,
                envs=self._env_vars,
                **self._create_options,
            )
            self._sandbox = sandbox
            self._register_atexit()
            try:
                await sandbox.files.make_dir(str(self._workdir))
            except BaseException:
                # On failure the handle stays for `aclose` and the atexit fallback to retry.
                await asyncio.shield(self._kill())
                raise

            if self._closed:
                # `aclose` ran while this creation was in flight.
                await self._kill()
                raise RuntimeError("E2BSandbox has been closed.")
            self._expires_at = started + self._lifetime
            self._ready = True
            logger.info("E2B sandbox created (id=%s)", sandbox.sandbox_id)
            return sandbox

    async def aclose(self) -> None:
        """Kill the sandbox. Safe to call repeatedly.

        Waits for a creation in progress on the same event loop, so a sandbox
        that is still being created is killed too. A kill that fails keeps the
        handle so a later call can retry, and leaves the atexit fallback armed
        for the case where no retry comes.
        """
        self._closed = True
        async with self._loop_lock():
            if self._sandbox is not None:
                await self._kill()

    async def _kill(self) -> bool:
        assert self._sandbox is not None
        try:
            await self._sandbox.kill()
        except Exception as e:
            logger.debug("Suppressed exception during E2B sandbox kill: %s", e)
            return False
        self._sandbox = None
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
        if self._sandbox is None:
            return
        try:
            SyncSandbox.kill(self._sandbox.sandbox_id, **self._api_params)
        except Exception as e:
            logger.debug("Suppressed exception during atexit E2B sandbox cleanup: %s", e)


def _gone(error: Exception) -> ExecResult:
    return ExecResult(
        output=f"E2B sandbox is no longer available (it expired, was paused, or was killed): {error}",
        exit_code=1,
    )
