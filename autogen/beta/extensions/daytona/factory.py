# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
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

from autogen.beta.annotations import Variable
from autogen.beta.tools.builtin._resolve import resolve_variable

from .sandbox import DaytonaSandbox

if TYPE_CHECKING:
    from daytona import Image

    from autogen.beta.context import ConversationContext


@dataclass(slots=True)
class DaytonaResources:
    """Resource limits for a Daytona sandbox.

    Only applied when ``image`` is set on the factory; ignored when a
    ``snapshot`` is used (snapshots carry their own resource profile).
    """

    cpu: int | None = None
    memory: int | None = None
    disk: int | None = None


class DaytonaSandboxFactory:
    """:class:`SandboxFactory` for :class:`DaytonaSandbox`.

    All shaping parameters (``api_key``, ``api_url``, ``target``,
    ``snapshot``, ``image``, ``env_vars``) accept a
    :class:`~autogen.beta.annotations.Variable` for deferred resolution
    from ``context.variables`` — useful for per-tenant credentials or
    A/B-tested images. Variables are resolved on each :meth:`open` call,
    and a fresh :class:`DaytonaSandbox` is created and torn down per
    scope.
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
        async with sandbox:
            yield sandbox
