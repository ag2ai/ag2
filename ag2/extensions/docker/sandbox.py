# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import atexit
import io
import logging
import os
import tarfile
import threading
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any

import docker
from docker.errors import APIError, NotFound

from ag2.annotations import Variable
from ag2.tools.sandbox import ExecResult, SandboxBase

logger = logging.getLogger(__name__)


class DockerSandbox(SandboxBase):
    """Sandbox backed by a long-lived Docker container.

    All shaping parameters (``image``, ``env_vars``, ``network_mode``, …)
    are concrete values — no :class:`~ag2.annotations.Variable`
    is accepted here. Variable resolution lives on
    :class:`DockerEnvironment`, which constructs a :class:`DockerSandbox`
    after binding Variables to a :class:`ConversationContext`.

    Lifecycle is explicit: the container is created on ``__aenter__``
    (or lazily on the first call as a safety net) and released on
    ``__aexit__`` / :meth:`aclose`.
    """

    def __init__(
        self,
        *,
        image: str = "python:3.12-slim",
        env_vars: dict[str, str] | None = None,
        timeout: float = 60,
        network_mode: str = "none",
        mem_limit: str | None = "512m",
        cpu_quota: int | None = None,
        user: str | None = None,
        auto_remove: bool = True,
        host_path: str | os.PathLike[str] | None = None,
        workdir: str = "/workspace",
    ) -> None:
        if timeout < 1:
            raise ValueError("`timeout` must be >= 1 second.")

        for name, value in (("image", image), ("env_vars", env_vars), ("network_mode", network_mode)):
            if isinstance(value, Variable):
                raise TypeError(
                    f"DockerSandbox.{name} must be a concrete value; got Variable. "
                    "Wrap with DockerEnvironment to resolve Variables from a Context."
                )

        self._image = image
        self._env_vars = env_vars or {}
        self._default_timeout = timeout
        self._network_mode = network_mode
        self._mem_limit = mem_limit
        self._cpu_quota = cpu_quota
        self._user = user
        self._auto_remove = auto_remove
        self._host_path = Path(host_path).resolve() if host_path is not None else None
        self._workdir = PurePosixPath(workdir)

        self._client: Any = None
        self._container: Any = None
        # A threading.Lock (not asyncio.Lock) guards container creation so a
        # cached sandbox stays usable across the throw-away event loops that
        # the sync shell path spins up via asyncio.run. asyncio.Lock binds to
        # the first loop that awaits it and would raise on the next one.
        self._create_lock = threading.Lock()
        self._closed = False
        self._atexit_registered = False

    @property
    def workdir(self) -> PurePosixPath:
        return self._workdir

    @property
    def host_workdir(self) -> Path | None:
        return self._host_path

    @property
    def closed(self) -> bool:
        return self._closed

    async def __aenter__(self) -> "DockerSandbox":
        await self._ensure_container()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __deepcopy__(self, memo: dict) -> "DockerSandbox":  # type: ignore[type-arg]
        # A live container handle holding a threading.Lock — not deepcopy-able
        # and not meaningfully duplicable. Sharing on copy is the only sane
        # semantics (lets a bare DockerSandbox be attached to a tool).
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

        container = await self._ensure_container()
        exec_kwargs: dict[str, Any] = {"stderr": True, "stdout": True, "workdir": str(self._workdir)}
        if env is not None:
            exec_kwargs["environment"] = env

        exec_timeout = timeout if timeout is not None else self._default_timeout
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(container.exec_run, argv, **exec_kwargs),
                timeout=exec_timeout,
            )
        except asyncio.TimeoutError:
            await self._restart_container()
            return ExecResult(output=f"Execution timed out after {exec_timeout}s", exit_code=124)
        except APIError as e:
            return ExecResult(output=f"Docker API error: {e}", exit_code=1)

        output_bytes = result.output if isinstance(result.output, bytes) else b"".join(result.output or [])
        return ExecResult(
            output=output_bytes.decode(errors="replace").strip(),
            exit_code=result.exit_code or 0,
        )

    async def put_file(self, path: PurePosixPath, content: bytes) -> None:
        target = self._resolve_inside(path)
        archive = _make_single_file_tar(target.name, content)
        container = await self._ensure_container()
        await asyncio.to_thread(container.put_archive, str(target.parent), archive)

    def _resolve_inside(self, path: PurePosixPath) -> PurePosixPath:
        if path.is_absolute():
            raise ValueError(f"Absolute paths are not allowed in put_file/get_file: {path}")
        return self._workdir / path

    async def _ensure_container(self) -> Any:
        if self._closed:
            raise RuntimeError("DockerSandbox has been closed.")
        if self._container is not None:
            return self._container
        # The blocking docker calls and the creation guard both live in a
        # single worker thread, so the threading.Lock serialises concurrent
        # creators regardless of which event loop (or thread) drives them.
        return await asyncio.to_thread(self._create_container_sync)

    def _create_container_sync(self) -> Any:
        with self._create_lock:
            if self._closed:
                raise RuntimeError("DockerSandbox has been closed.")
            if self._container is not None:
                return self._container

            kwargs: dict[str, Any] = {
                "command": ["sh", "-c", f"mkdir -p {self._workdir} && sleep infinity"],
                "detach": True,
                "network_mode": self._network_mode,
                "mem_limit": self._mem_limit,
                "cpu_quota": self._cpu_quota,
                "user": self._user,
                "environment": self._env_vars,
                "auto_remove": self._auto_remove,
                "working_dir": str(self._workdir),
            }
            if self._host_path is not None:
                kwargs["volumes"] = {str(self._host_path): {"bind": str(self._workdir), "mode": "rw"}}

            client = docker.from_env()
            container = client.containers.run(self._image, **kwargs)
            self._client = client
            self._container = container
            if not self._atexit_registered:
                atexit.register(self._atexit_close)
                self._atexit_registered = True
            logger.info("Docker sandbox container started (id=%s, image=%s)", container.short_id, self._image)
            return container

    async def _restart_container(self) -> None:
        await asyncio.to_thread(self._restart_container_sync)

    def _restart_container_sync(self) -> None:
        with self._create_lock:
            if self._container is None:
                return
            old = self._container
            self._container = None
        try:
            old.stop(timeout=1)
        except Exception as e:
            logger.debug("Suppressed exception during container stop on restart: %s", e)
        if not self._auto_remove:
            try:
                old.remove(force=True)
            except Exception as e:
                logger.debug("Suppressed exception during container remove on restart: %s", e)

    async def aclose(self) -> None:
        if self._atexit_registered:
            with suppress(ValueError):
                atexit.unregister(self._atexit_close)
            self._atexit_registered = False
        self._closed = True
        if self._container is not None:
            try:
                await asyncio.to_thread(self._container.stop, timeout=1)
            except NotFound:
                pass
            except Exception as e:
                logger.debug("Suppressed exception during container stop: %s", e)
            if not self._auto_remove:
                try:
                    await asyncio.to_thread(self._container.remove, force=True)
                except NotFound:
                    pass
                except Exception as e:
                    logger.debug("Suppressed exception during container remove: %s", e)
            self._container = None
        if self._client is not None:
            try:
                await asyncio.to_thread(self._client.close)
            except Exception as e:
                logger.debug("Suppressed exception during client close: %s", e)
            self._client = None

    def _atexit_close(self) -> None:
        if self._container is None:
            return
        try:
            asyncio.run(self.aclose())
        except Exception as e:
            logger.debug("Suppressed exception during atexit container cleanup: %s", e)


def _make_single_file_tar(name: str, content: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name=name)
        info.size = len(content)
        tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()
