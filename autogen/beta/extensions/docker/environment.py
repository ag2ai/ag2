# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import atexit
import base64
import logging
import uuid
from typing import TYPE_CHECKING, Any

import docker
from docker.errors import APIError, NotFound

from autogen.beta.annotations import Variable
from autogen.beta.tools.builtin._resolve import resolve_variable
from autogen.beta.tools.code import CodeEnvironment, CodeLanguage, CodeRunResult

if TYPE_CHECKING:
    from autogen.beta.context import ConversationContext

logger = logging.getLogger(__name__)

_LANG_FILE_EXT: dict[CodeLanguage, str] = {
    "python": "py",
    "bash": "sh",
    "javascript": "js",
    "typescript": "ts",
}

_LANG_RUN_CMD: dict[CodeLanguage, str] = {
    "python": ["python", "-c"],
    "bash": ["bash", "-c"],
}

_FILE_RUN_CMD: dict[CodeLanguage, str] = {
    "javascript": "node",
    "typescript": "ts-node",
}


class DockerCodeEnvironment(CodeEnvironment):
    """:class:`~autogen.beta.tools.code.CodeEnvironment` backed by a local
    Docker container.

    A long-lived container is created lazily on the first :meth:`run`
    call (running ``sleep infinity`` so it stays up between calls) and
    reused for the lifetime of the environment. Each ``run_code``
    invocation issues a ``docker exec`` against that container.

    Safety defaults:

    - ``network_mode="none"`` — no network access. Set to ``"bridge"``
      to opt in.
    - ``mem_limit="512m"`` — bound runaway processes.
    - ``auto_remove=True`` and explicit ``aclose()`` ensure the
      container is removed when no longer needed.

    All sandbox-shaping parameters (``image``, ``env_vars``,
    ``network_mode``) accept a
    :class:`~autogen.beta.annotations.Variable` for deferred resolution
    from ``context.variables``.

    Args:
        image: Docker image. Default ``"python:3.12-slim"`` (has
               ``python`` and ``bash``).
        env_vars: Environment variables passed into the container.
        timeout: Per-execution timeout in seconds. Default: 60.
        network_mode: Container network mode. Default ``"none"``
                      (no network); ``"bridge"`` for default network,
                      ``"host"`` to share host networking.
        mem_limit: Memory limit (Docker syntax, e.g. ``"512m"``,
                   ``"1g"``). Default: ``"512m"``. ``None`` disables.
        cpu_quota: CPU quota in microseconds per 100ms period. ``None``
                   disables.
        user: User to run as inside the container. ``None`` (default)
              uses the image's default user. Recommended override:
              ``"nobody"`` for images that ship that user.
        auto_remove: Whether Docker should auto-remove the container on
                     stop. Default: ``True``.
        languages: Languages this environment will accept. Default
                   ``("python", "bash")`` — these come for free with
                   ``python:3.12-slim``. Add ``"javascript"`` /
                   ``"typescript"`` only if your image has ``node`` /
                   ``ts-node``.
    """

    def __init__(
        self,
        *,
        image: "str | Variable" = "python:3.12-slim",
        env_vars: "dict[str, str] | Variable | None" = None,
        timeout: int = 60,
        network_mode: "str | Variable" = "none",
        mem_limit: str | None = "512m",
        cpu_quota: int | None = None,
        user: str | None = None,
        auto_remove: bool = True,
        languages: tuple[CodeLanguage, ...] = ("python", "bash"),
    ) -> None:
        if timeout < 1:
            raise ValueError("`timeout` must be >= 1 second.")

        self._image = image
        self._env_vars = env_vars
        self._timeout = timeout
        self._network_mode = network_mode
        self._mem_limit = mem_limit
        self._cpu_quota = cpu_quota
        self._user = user
        self._auto_remove = auto_remove
        self._languages: tuple[CodeLanguage, ...] = tuple(languages)

        self._client: docker.DockerClient | None = None
        self._container: Any = None
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def supported_languages(self) -> tuple[CodeLanguage, ...]:
        return self._languages

    async def run(
        self,
        code: str,
        language: CodeLanguage,
        *,
        context: "ConversationContext | None" = None,
    ) -> CodeRunResult:
        if language not in self._languages:
            return CodeRunResult(
                output=f"Language {language!r} is not enabled. Available: {list(self._languages)}",
                exit_code=2,
            )

        container = await self._ensure_container(context)

        if language in _LANG_RUN_CMD:
            cmd: list[str] = [*_LANG_RUN_CMD[language], code]
        else:
            # File-based execution for js/ts: base64-decode the snippet
            # into a temp file inside /workspace, exec the runner, clean up.
            ext = _LANG_FILE_EXT[language]
            runner = _FILE_RUN_CMD[language]
            script_path = f"/workspace/ag2_{uuid.uuid4().hex}.{ext}"
            encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
            cmd = [
                "sh",
                "-c",
                f"echo {encoded} | base64 -d > {script_path} && {runner} {script_path}; "
                f"rc=$?; rm -f {script_path}; exit $rc",
            ]

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(container.exec_run, cmd, stderr=True, stdout=True, workdir="/workspace"),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            # docker exec has no clean cancel API — recycle the container so a runaway
            # process doesn't keep consuming resources.
            await self._restart_container()
            return CodeRunResult(
                output=f"Execution timed out after {self._timeout}s",
                exit_code=124,
            )
        except APIError as e:
            return CodeRunResult(output=f"Docker API error: {e}", exit_code=1)

        output_bytes = result.output if isinstance(result.output, bytes) else b"".join(result.output or [])
        return CodeRunResult(
            output=output_bytes.decode(errors="replace"),
            exit_code=result.exit_code or 0,
        )

    async def _ensure_container(self, context: "ConversationContext | None" = None) -> Any:
        async with self._lock:
            if self._closed:
                raise RuntimeError("DockerCodeEnvironment has been closed.")
            if self._container is not None:
                return self._container

            image = resolve_variable(self._image, context, param_name="image") if context else self._image
            env_vars = (
                resolve_variable(self._env_vars, context, param_name="env_vars") if context else self._env_vars
            ) or {}
            network_mode = (
                resolve_variable(self._network_mode, context, param_name="network_mode")
                if context
                else self._network_mode
            )

            for value, name in ((image, "image"), (network_mode, "network_mode")):
                if isinstance(value, Variable):
                    raise RuntimeError(
                        f"Docker `{name}` given as Variable but no Context available to resolve it. "
                        "Variables are only resolvable when SandboxCodeTool is invoked through an Agent."
                    )

            self._client = await asyncio.to_thread(docker.from_env)
            self._container = await asyncio.to_thread(
                self._client.containers.run,
                image,
                command=["sh", "-c", "mkdir -p /workspace && sleep infinity"],
                detach=True,
                network_mode=network_mode,
                mem_limit=self._mem_limit,
                cpu_quota=self._cpu_quota,
                user=self._user,
                environment=env_vars,
                auto_remove=self._auto_remove,
                working_dir="/workspace",
            )
            atexit.register(self._atexit_close)
            logger.info("Docker sandbox container started (id=%s, image=%s)", self._container.short_id, image)
            return self._container

    async def _restart_container(self) -> None:
        """Stop and recreate the container — used after a timeout where we
        can't cancel the in-flight exec.
        """
        async with self._lock:
            if self._container is None:
                return
            old = self._container
            self._container = None
            try:
                await asyncio.to_thread(old.stop, timeout=1)
            except Exception as e:
                logger.debug("Suppressed exception during container stop on restart: %s", e)
            if not self._auto_remove:
                try:
                    await asyncio.to_thread(old.remove, force=True)
                except Exception as e:
                    logger.debug("Suppressed exception during container remove on restart: %s", e)

    async def aclose(self) -> None:
        """Stop and remove the container. Safe to call multiple times."""
        atexit.unregister(self._atexit_close)
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

    async def __aenter__(self) -> "DockerCodeEnvironment":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
