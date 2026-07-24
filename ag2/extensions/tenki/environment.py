# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Configure reusable Tenki cloud sandboxes for AG2 tools."""

import threading
from collections.abc import AsyncIterator, Hashable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from tenki_sandbox import AsyncClient

from ag2.annotations import Variable
from ag2.tools.builtin._resolve import resolve_variable
from ag2.tools.code import CodeLanguage
from ag2.tools.sandbox import LanguageRunner

from .sandbox import TenkiSandbox

if TYPE_CHECKING:
    from ag2.context import ConversationContext


@dataclass(slots=True)
class TenkiResources:
    """Resource limits for a Tenki sandbox."""

    cpu_cores: int | None = None
    memory_mb: int | None = None
    disk_size_gb: int | None = None


class TenkiEnvironment:
    """:class:`SandboxFactory` for :class:`TenkiSandbox`.

    Tenki credentials and sandbox parameters may be deferred with
    :class:`~ag2.annotations.Variable`. Sandboxes are cached by their resolved
    parameters and remain available until :meth:`aclose`, preserving files
    across tool calls.

    ``TENKI_API_KEY`` and ``TENKI_API_URL`` are read by the SDK when their
    constructor arguments are omitted. If ``project_id`` is omitted, the sole
    project visible to the API key is selected automatically.
    """

    def __init__(
        self,
        *,
        api_key: "str | Variable | None" = None,  # pragma: allowlist secret
        api_url: "str | Variable | None" = None,
        project_id: "str | Variable | None" = None,
        name: "str | Variable | None" = "ag2",
        image: "str | Variable | None" = None,
        env_vars: "dict[str, str] | Variable | None" = None,
        resources: TenkiResources | None = None,
        timeout: float = 60,
        max_duration: float = 900,
        workdir: str = "/home/tenki",
    ) -> None:
        if timeout <= 0:
            raise ValueError("`timeout` must be greater than 0 seconds.")
        if max_duration <= 0:
            raise ValueError("`max_duration` must be greater than 0 seconds.")

        self._api_key = api_key
        self._api_url = api_url
        self._project_id = project_id
        self._name = name
        self._image = image
        self._env_vars = env_vars
        self._resources = resources
        self._timeout = timeout
        self._max_duration = max_duration
        self._workdir = workdir

        self._cache: dict[Hashable, TenkiSandbox] = {}
        self._cache_lock = threading.Lock()

    @property
    def workdir(self) -> PurePosixPath:
        """Sandbox-side working directory advertised to AG2 tools."""
        return PurePosixPath(self._workdir)

    @property
    def code_runners(self) -> dict[CodeLanguage, LanguageRunner]:
        """Interpreter overrides for Tenki's default guest image."""
        return {"python": LanguageRunner(inline_argv=("python3", "-c"))}

    @asynccontextmanager
    async def open(
        self,
        context: "ConversationContext | None" = None,
    ) -> AsyncIterator[TenkiSandbox]:
        api_key = resolve_variable(self._api_key, context, param_name="api_key") if context else self._api_key
        api_url = resolve_variable(self._api_url, context, param_name="api_url") if context else self._api_url
        project_id = (
            resolve_variable(self._project_id, context, param_name="project_id") if context else self._project_id
        )
        name = resolve_variable(self._name, context, param_name="name") if context else self._name
        image = resolve_variable(self._image, context, param_name="image") if context else self._image
        env_vars = (
            resolve_variable(self._env_vars, context, param_name="env_vars") if context else self._env_vars
        ) or {}

        unresolved = (api_key, api_url, project_id, name, image, env_vars)
        if any(isinstance(value, Variable) for value in unresolved):
            raise RuntimeError(
                "Tenki parameters given as Variable but no Context is available to resolve them. "
                "Variables are only resolvable when a sandbox tool is invoked through an Agent."
            )
        assert api_key is None or isinstance(api_key, str)
        assert api_url is None or isinstance(api_url, str)
        assert project_id is None or isinstance(project_id, str)
        assert name is None or isinstance(name, str)
        assert image is None or isinstance(image, str)
        assert isinstance(env_vars, dict)

        resources = self._resources or TenkiResources()
        key: Hashable = (
            api_key,
            api_url,
            project_id,
            name,
            image,
            tuple(sorted(env_vars.items())),
            resources.cpu_cores,
            resources.memory_mb,
            resources.disk_size_gb,
            self._timeout,
            self._max_duration,
            self._workdir,
        )

        with self._cache_lock:
            sandbox = self._cache.get(key)
            if sandbox is None or sandbox.closed:
                client = AsyncClient(auth_token=api_key, base_url=api_url)
                sandbox = TenkiSandbox(
                    client=client,
                    create_options={
                        "project_id": project_id,
                        "name": name,
                        "image": image,
                        "env": env_vars,
                        "cpu_cores": resources.cpu_cores,
                        "memory_mb": resources.memory_mb,
                        "disk_size_gb": resources.disk_size_gb,
                        "allow_inbound": False,
                        "allow_outbound": True,
                        "max_duration": self._max_duration,
                        "tags": ["ag2"],
                    },
                    timeout=self._timeout,
                    workdir=self._workdir,
                )
                self._cache[key] = sandbox

        try:
            await sandbox.__aenter__()
        except BaseException:
            with self._cache_lock:
                if self._cache.get(key) is sandbox:
                    self._cache.pop(key)
            with suppress(BaseException):
                await sandbox.aclose()
            raise
        # The factory owns the lifecycle so cached state survives this scope.
        yield sandbox

    async def aclose(self) -> None:
        """Terminate every cached sandbox. Safe to call multiple times."""
        with self._cache_lock:
            sandboxes = list(self._cache.values())
            self._cache.clear()
        errors: list[Exception] = []
        for sandbox in sandboxes:
            try:
                await sandbox.aclose()
            except Exception as e:
                errors.append(e)
        if errors:
            raise errors[0]

    async def __aenter__(self) -> "TenkiEnvironment":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __deepcopy__(self, memo: dict) -> "TenkiEnvironment":  # type: ignore[type-arg]
        return self
