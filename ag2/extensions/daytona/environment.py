# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import threading
from collections.abc import AsyncIterator, Hashable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from daytona import (
    AsyncDaytona,
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    DaytonaConfig,
    Resources,
)

from ag2.annotations import Variable
from ag2.tools.builtin._resolve import resolve_variable

from .sandbox import DaytonaSandbox

if TYPE_CHECKING:
    from daytona import Image

    from ag2.context import ConversationContext


@dataclass(slots=True)
class DaytonaResources:
    """Resource limits for a Daytona sandbox.

    Only applied when ``image`` is set on the factory; ignored when a
    ``snapshot`` is used (snapshots carry their own resource profile).
    """

    cpu: int | None = None
    memory: int | None = None
    disk: int | None = None


class DaytonaEnvironment:
    """:class:`SandboxFactory` for :class:`DaytonaSandbox`.

    This is the backend object you hand to a tool::

        env = DaytonaEnvironment(api_key=Variable("daytona_key"), image="python:3.12")
        shell = SandboxShellTool(env)
        code = SandboxCodeTool(env, languages=("python", "typescript"))

    All shaping parameters (``api_key``, ``api_url``, ``target``,
    ``snapshot``, ``image``, ``env_vars``) accept a
    :class:`~ag2.annotations.Variable` for deferred resolution
    from ``context.variables`` — useful for per-tenant credentials or
    A/B-tested images. Variables are resolved on each :meth:`open` call.

    Sandboxes are **cached** by resolved parameters: opens with the same
    resolved values reuse one cloud sandbox (state persists across tool
    calls); distinct values (e.g. per-tenant ``image=Variable(...)``) get
    distinct sandboxes. Cached sandboxes live until :meth:`aclose` (or
    per-sandbox atexit).
    """

    def __init__(
        self,
        *,
        api_key: "str | Variable | None" = None,  # pragma: allowlist secret
        api_url: "str | Variable | None" = None,
        target: "str | Variable | None" = None,
        snapshot: "str | Variable | None" = None,
        image: "str | Image | Variable | None" = None,
        env_vars: "dict[str, str] | Variable | None" = None,
        resources: DaytonaResources | None = None,
        timeout: int = 60,
        workdir: str = "/workspace",
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
        self._workdir = workdir

        self._cache: dict[Hashable, DaytonaSandbox] = {}
        self._cache_lock = threading.Lock()

    @asynccontextmanager
    async def open(
        self,
        context: "ConversationContext | None" = None,
    ) -> AsyncIterator[DaytonaSandbox]:
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

        assert isinstance(env_vars, dict)

        key: Hashable = (
            api_key,
            api_url,
            target,
            snapshot,
            repr(image),
            tuple(sorted(env_vars.items())),
            self._workdir,
            self._timeout,
        )

        with self._cache_lock:
            sandbox = self._cache.get(key)
            if sandbox is None or sandbox.closed:
                # Build the client + params only on a cache miss — a cache hit
                # reuses the existing sandbox and its client untouched.
                config_kwargs: dict[str, str] = {}
                if api_key is not None:
                    config_kwargs["api_key"] = api_key
                if api_url is not None:
                    config_kwargs["api_url"] = api_url
                if target is not None:
                    config_kwargs["target"] = target
                client = AsyncDaytona(DaytonaConfig(**config_kwargs))

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

                sandbox = DaytonaSandbox(
                    client=client,
                    params=params,
                    timeout=self._timeout,
                    workdir=self._workdir,
                )
                self._cache[key] = sandbox

        await sandbox.__aenter__()
        # Factory owns the lifecycle; do not delete on scope exit.
        yield sandbox

    async def aclose(self) -> None:
        """Delete every cached sandbox. Safe to call multiple times."""
        with self._cache_lock:
            sandboxes = list(self._cache.values())
            self._cache.clear()
        for sandbox in sandboxes:
            await sandbox.aclose()

    async def __aenter__(self) -> "DaytonaEnvironment":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __deepcopy__(self, memo: dict) -> "DaytonaEnvironment":  # type: ignore[type-arg]
        # Shared resource handle: a copy is the SAME factory. The cached cloud
        # sandboxes (and the threading.Lock guarding them — not deepcopy-able)
        # are reused, not duplicated. This is what lets Agent.add_tool deepcopy
        # a SandboxShellTool / SandboxCodeTool backed by this environment.
        return self
