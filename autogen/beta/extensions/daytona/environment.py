# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import atexit
import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from autogen.beta.annotations import Variable
from autogen.beta.tools.builtin._resolve import resolve_variable
from autogen.beta.tools.code import CodeEnvironment, CodeLanguage, CodeRunResult

if TYPE_CHECKING:
    from daytona import Image

    from autogen.beta.context import ConversationContext

from daytona import (
    AsyncDaytona,
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    DaytonaConfig,
    DaytonaError,
    DaytonaNotFoundError,
    DaytonaRateLimitError,
    DaytonaTimeoutError,
    Resources,
)

logger = logging.getLogger(__name__)

_LANG_FILE_EXT: dict[CodeLanguage, str] = {
    "python": "py",
    "bash": "sh",
    "javascript": "js",
    "typescript": "ts",
}

_LANG_RUN_CMD: dict[CodeLanguage, str] = {
    "bash": "bash",
    "javascript": "node",
    "typescript": "ts-node",
}


@dataclass(slots=True)
class DaytonaResources:
    """Resource limits for a Daytona sandbox.

    Only applied when ``image`` is set on the environment; ignored when a
    ``snapshot`` is used (snapshots carry their own resource profile).
    """

    cpu: int | None = None
    memory: int | None = None
    disk: int | None = None


class DaytonaCodeEnvironment(CodeEnvironment):
    """:class:`~autogen.beta.tools.code.CodeEnvironment` backed by a Daytona sandbox.

    The sandbox is created lazily on the first :meth:`run` call and reused
    for the lifetime of the environment. Cleanup is registered via
    :func:`atexit` so the sandbox is released even if the user forgets to
    call :meth:`aclose` — for tighter scoping use
    ``async with DaytonaCodeEnvironment(...) as env``.

    All sandbox-shaping parameters (``api_key``, ``api_url``, ``target``,
    ``snapshot``, ``image``, ``env_vars``) accept a
    :class:`~autogen.beta.annotations.Variable` for deferred resolution
    from ``context.variables`` — useful for per-tenant credentials or
    A/B-tested images. Variables are resolved on the first :meth:`run`
    call, when the sandbox is created.

    Args:
        api_key: Daytona API key. Falls back to ``DAYTONA_API_KEY``.
        api_url: Daytona API URL. Falls back to ``DAYTONA_API_URL``.
        target: Daytona target region (e.g. ``"us"``, ``"eu"``). Falls back
                to ``DAYTONA_TARGET``.
        snapshot: Snapshot name. Mutually exclusive with ``image``.
        image: Custom Docker image. Mutually exclusive with ``snapshot``.
        env_vars: Environment variables passed into the sandbox.
        resources: Resource limits. Only applied with ``image``.
        timeout: Per-execution timeout in seconds. Default: 60.
        languages: Languages this environment will accept. Defaults to all
                   four supported by Daytona.
    """

    def __init__(
        self,
        *,
        api_key: str | Variable | None = None,
        api_url: str | Variable | None = None,
        target: str | Variable | None = None,
        snapshot: str | Variable | None = None,
        image: "str | Image | Variable | None" = None,
        env_vars: dict[str, str] | Variable | None = None,
        resources: DaytonaResources | None = None,
        timeout: int = 60,
        languages: tuple[CodeLanguage, ...] = ("python", "bash", "javascript", "typescript"),
    ) -> None:
        if (
            snapshot is not None
            and image is not None
            and not isinstance(snapshot, Variable)
            and not isinstance(image, Variable)
        ):
            raise ValueError("Specify either `snapshot` or `image`, not both.")
        if timeout < 1:
            raise ValueError("`timeout` must be >= 1 second.")

        self._api_key = api_key
        self._api_url = api_url
        self._target = target
        self._snapshot = snapshot
        self._image = image
        self._env_vars = env_vars
        self._resources = resources
        self._timeout = timeout
        self._languages: tuple[CodeLanguage, ...] = tuple(languages)

        self._client: AsyncDaytona | None = None
        self._sandbox: Any = None
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

        sandbox = await self._ensure_sandbox(context)

        try:
            if language == "python":
                response = await sandbox.process.code_run(code, timeout=self._timeout)
            else:
                ext = _LANG_FILE_EXT[language]
                script_path = f"/tmp/ag2_{uuid.uuid4().hex}.{ext}"
                await sandbox.fs.upload_file(code.encode("utf-8"), script_path)
                cmd = f"{_LANG_RUN_CMD[language]} {script_path}"
                response = await sandbox.process.exec(cmd, timeout=self._timeout)
                try:
                    await sandbox.fs.delete_file(script_path)
                except DaytonaNotFoundError:
                    pass
                except Exception as e:
                    logger.debug("Failed to delete temp script %s: %s", script_path, e)
        except DaytonaTimeoutError as e:
            return CodeRunResult(output=f"Execution timed out: {e}", exit_code=124)
        except DaytonaRateLimitError as e:
            return CodeRunResult(output=f"Daytona rate limit exceeded: {e}", exit_code=1)
        except DaytonaError as e:
            return CodeRunResult(output=f"Daytona error: {e}", exit_code=1)

        return CodeRunResult(output=response.result or "", exit_code=response.exit_code or 0)

    async def _ensure_sandbox(self, context: "ConversationContext | None" = None) -> Any:
        async with self._lock:
            if self._closed:
                raise RuntimeError("DaytonaCodeEnvironment has been closed.")
            if self._sandbox is not None:
                return self._sandbox

            api_key = resolve_variable(self._api_key, context, param_name="api_key") if context else self._api_key
            api_url = resolve_variable(self._api_url, context, param_name="api_url") if context else self._api_url
            target = resolve_variable(self._target, context, param_name="target") if context else self._target
            snapshot = resolve_variable(self._snapshot, context, param_name="snapshot") if context else self._snapshot
            image = resolve_variable(self._image, context, param_name="image") if context else self._image
            env_vars = (
                resolve_variable(self._env_vars, context, param_name="env_vars") if context else self._env_vars
            ) or {}

            if isinstance(api_key, Variable) or isinstance(api_url, Variable) or isinstance(target, Variable):
                raise RuntimeError(
                    "Daytona credentials given as Variable but no Context available to resolve them. "
                    "Variables are only resolvable when SandboxCodeTool is invoked through an Agent."
                )
            if snapshot is not None and image is not None:
                raise ValueError("Specify either `snapshot` or `image`, not both.")

            config_kwargs: dict[str, str] = {}
            if api_key is not None:
                config_kwargs["api_key"] = api_key
            if api_url is not None:
                config_kwargs["api_url"] = api_url
            if target is not None:
                config_kwargs["target"] = target

            self._client = AsyncDaytona(DaytonaConfig(**config_kwargs))

            params: CreateSandboxFromSnapshotParams | CreateSandboxFromImageParams
            if snapshot is not None:
                params = CreateSandboxFromSnapshotParams(
                    snapshot=snapshot,
                    env_vars=env_vars,
                    auto_stop_interval=0,
                )
            elif image is not None:
                sdk_resources = None
                r = self._resources
                if r is not None and any(v is not None for v in (r.cpu, r.memory, r.disk)):
                    sdk_resources = Resources(cpu=r.cpu, memory=r.memory, disk=r.disk)
                params = CreateSandboxFromImageParams(
                    image=image,
                    env_vars=env_vars,
                    resources=sdk_resources,
                    auto_stop_interval=0,
                )
            else:
                params = CreateSandboxFromSnapshotParams(
                    env_vars=env_vars,
                    auto_stop_interval=0,
                )

            self._sandbox = await self._client.create(params)
            atexit.register(self._atexit_close)
            logger.info("Daytona sandbox created (id=%s)", self._sandbox.id)
            return self._sandbox

    async def aclose(self) -> None:
        """Delete the sandbox and close the client. Safe to call multiple times."""
        atexit.unregister(self._atexit_close)
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
                await self._client.close()
            except Exception as e:
                logger.debug("Suppressed exception during client close: %s", e)
            self._client = None

    def _atexit_close(self) -> None:
        if self._sandbox is None:
            return
        try:
            asyncio.run(self.aclose())
        except Exception as e:
            logger.debug("Suppressed exception during atexit sandbox cleanup: %s", e)

    async def __aenter__(self) -> "DaytonaCodeEnvironment":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
