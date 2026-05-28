# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from autogen.beta.annotations import Variable
from autogen.beta.tools.builtin._resolve import resolve_variable

from .sandbox import DockerSandbox

if TYPE_CHECKING:
    from autogen.beta.context import ConversationContext


class DockerSandboxFactory:
    """:class:`~autogen.beta.tools.sandbox.SandboxFactory` for
    :class:`DockerSandbox`.

    All :class:`~autogen.beta.annotations.Variable`-capable parameters
    (``image``, ``env_vars``, ``network_mode``) are resolved on
    :meth:`open` against the active
    :class:`~autogen.beta.context.ConversationContext`. The resulting
    :class:`DockerSandbox` receives only concrete values, so the backend
    never has to know about Variables or Context.

    The factory owns the sandbox lifecycle: every :meth:`open` constructs
    a fresh :class:`DockerSandbox`, enters it (starting the container),
    and tears it down on exit.
    """

    def __init__(
        self,
        *,
        image: "str | Variable" = "python:3.12-slim",
        env_vars: "dict[str, str] | Variable | None" = None,
        timeout: float = 60,
        network_mode: "str | Variable" = "none",
        mem_limit: str | None = "512m",
        cpu_quota: int | None = None,
        user: str | None = None,
        auto_remove: bool = True,
        host_path: str | os.PathLike[str] | None = None,
        workdir: str = "/workspace",
    ) -> None:
        self._image = image
        self._env_vars = env_vars
        self._timeout = timeout
        self._network_mode = network_mode
        self._mem_limit = mem_limit
        self._cpu_quota = cpu_quota
        self._user = user
        self._auto_remove = auto_remove
        self._host_path = host_path
        self._workdir = workdir

    @asynccontextmanager
    async def open(
        self,
        context: "ConversationContext | None" = None,
    ) -> AsyncIterator[DockerSandbox]:
        image = resolve_variable(self._image, context, param_name="image") if context else self._image
        env_vars = (
            resolve_variable(self._env_vars, context, param_name="env_vars") if context else self._env_vars
        ) or {}
        network_mode = (
            resolve_variable(self._network_mode, context, param_name="network_mode") if context else self._network_mode
        )

        for value, name in ((image, "image"), (network_mode, "network_mode")):
            if isinstance(value, Variable):
                raise RuntimeError(
                    f"Docker `{name}` given as Variable but no Context available to resolve it. "
                    "Variables are only resolvable when the sandbox is driven from an Agent "
                    "(SandboxCodeTool / LocalShellTool wrappers forward the active Context)."
                )

        assert isinstance(image, str)
        assert isinstance(network_mode, str)
        assert isinstance(env_vars, dict)
        sandbox = DockerSandbox(
            image=image,
            env_vars=dict(env_vars),
            timeout=self._timeout,
            network_mode=network_mode,
            mem_limit=self._mem_limit,
            cpu_quota=self._cpu_quota,
            user=self._user,
            auto_remove=self._auto_remove,
            host_path=self._host_path,
            workdir=self._workdir,
        )
        async with sandbox:
            yield sandbox
