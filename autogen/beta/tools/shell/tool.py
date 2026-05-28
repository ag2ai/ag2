# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterable
from contextlib import AsyncExitStack, ExitStack
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from autogen.beta.annotations import Context
from autogen.beta.middleware import BaseMiddleware, ToolMiddleware
from autogen.beta.tools.final import tool
from autogen.beta.tools.final.function_tool import FunctionTool
from autogen.beta.tools.sandbox import LocalSandbox, ShellAdapter
from autogen.beta.tools.tool import Tool

from .environment import LocalShellEnvironment, ShellEnvironment

if TYPE_CHECKING:
    from autogen.beta.annotations import Variable


class LocalShellTool(Tool):
    """Exposes a single ``shell`` function that runs commands in whatever
    sandbox is provided — local subprocess, Docker container, SSH, etc.

    Three ways to construct:

    1. **Path / no arg** (legacy): wraps a :class:`LocalShellEnvironment`.
       ``LocalShellTool()`` / ``LocalShellTool("/tmp/proj")``.
    2. **ShellEnvironment instance** (legacy): passed through.
    3. **adapter=ShellAdapter(...)** (preferred): uses any backend.

    Convenience class-methods for the common cases:
    :meth:`LocalShellTool.local`, :meth:`LocalShellTool.docker`.
    """

    def __init__(
        self,
        environment: "ShellEnvironment | ShellAdapter | str | os.PathLike[str] | None" = None,
        name: str = "run_shell_command",
        *,
        description: str = "Execute a shell command in the working directory: {workdir}",
        middleware: Iterable["ToolMiddleware"] = (),
        adapter: ShellAdapter | None = None,
    ) -> None:
        env: ShellEnvironment | ShellAdapter
        if adapter is not None:
            if environment is not None:
                raise ValueError("Pass either `environment` or `adapter`, not both.")
            env = adapter
        elif isinstance(environment, ShellAdapter):
            env = environment
        elif environment is None:
            env = LocalShellEnvironment()
        elif isinstance(environment, ShellEnvironment):
            env = environment
        else:
            env = LocalShellEnvironment.ensure_env(environment)

        def run_shell_command(command: str, ctx: Context) -> str:
            if isinstance(env, ShellAdapter):
                return env.run_sync(command, context=ctx)
            return env.run(command, context=ctx)

        self._env = env
        self._tool: FunctionTool = tool(
            run_shell_command,
            name=name,
            description=description.format(workdir=env.workdir),
            middleware=middleware,
        )

        self._workdir = env.workdir
        self.name = name

    @property
    def workdir(self) -> "Path | PurePosixPath":
        """The working directory of the underlying environment / adapter."""
        return self._workdir

    @classmethod
    def local(
        cls,
        path: str | os.PathLike[str] | None = None,
        *,
        name: str = "run_shell_command",
        description: str = "Execute a shell command in the working directory: {workdir}",
        middleware: Iterable["ToolMiddleware"] = (),
        allowed: list[str] | None = None,
        blocked: list[str] | None = None,
        ignore: list[str] | None = None,
        readonly: bool = False,
        env: dict[str, str] | None = None,
        timeout: float = 60,
        max_output: int = 100_000,
    ) -> "LocalShellTool":
        """Shorthand for ``LocalShellTool(adapter=ShellAdapter(LocalSandbox(...)))``."""
        sandbox = LocalSandbox(path=path, timeout=timeout, max_output=max_output)
        adapter = ShellAdapter(
            sandbox,
            allowed=allowed,
            blocked=blocked,
            ignore=ignore,
            readonly=readonly,
            env=env,
            timeout=timeout,
        )
        return cls(name=name, description=description, middleware=middleware, adapter=adapter)

    @classmethod
    def docker(
        cls,
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
        allowed: list[str] | None = None,
        blocked: list[str] | None = None,
        ignore: list[str] | None = None,
        readonly: bool = False,
        env: dict[str, str] | None = None,
        name: str = "run_shell_command",
        description: str = "Execute a shell command in the working directory: {workdir}",
        middleware: Iterable["ToolMiddleware"] = (),
    ) -> "LocalShellTool":
        """Shorthand for ``LocalShellTool(adapter=ShellAdapter(DockerSandboxFactory(...)))``."""
        from autogen.beta.extensions.docker import DockerSandboxFactory

        factory = DockerSandboxFactory(
            image=image,
            env_vars=env_vars,
            timeout=timeout,
            network_mode=network_mode,
            mem_limit=mem_limit,
            cpu_quota=cpu_quota,
            user=user,
            auto_remove=auto_remove,
            host_path=host_path,
            workdir=workdir,
        )
        adapter = ShellAdapter(
            factory,
            allowed=allowed,
            blocked=blocked,
            ignore=ignore,
            readonly=readonly,
            env=env,
            timeout=timeout,
        )
        return cls(name=name, description=description, middleware=middleware, adapter=adapter)

    @classmethod
    def daytona(
        cls,
        *,
        api_key: "str | Variable | None" = None,  # pragma: allowlist secret
        api_url: "str | Variable | None" = None,
        target: "str | Variable | None" = None,
        snapshot: "str | Variable | None" = None,
        image: "str | Variable | None" = None,
        env_vars: "dict[str, str] | Variable | None" = None,
        timeout: int = 60,
        workdir: str = "/workspace",
        allowed: list[str] | None = None,
        blocked: list[str] | None = None,
        ignore: list[str] | None = None,
        readonly: bool = False,
        env: dict[str, str] | None = None,
        name: str = "run_shell_command",
        description: str = "Execute a shell command in the working directory: {workdir}",
        middleware: Iterable["ToolMiddleware"] = (),
    ) -> "LocalShellTool":
        """Shorthand for ``LocalShellTool(adapter=ShellAdapter(DaytonaSandboxFactory(...)))``."""
        from autogen.beta.extensions.daytona import DaytonaSandboxFactory

        factory = DaytonaSandboxFactory(
            api_key=api_key,
            api_url=api_url,
            target=target,
            snapshot=snapshot,
            image=image,
            env_vars=env_vars,
            timeout=timeout,
            workdir=workdir,
        )
        adapter = ShellAdapter(
            factory,
            allowed=allowed,
            blocked=blocked,
            ignore=ignore,
            readonly=readonly,
            env=env,
            timeout=timeout,
        )
        return cls(name=name, description=description, middleware=middleware, adapter=adapter)

    async def schemas(self, context: "Context") -> list:  # type: ignore[type-arg]
        return await self._tool.schemas(context)

    def register(
        self,
        stack: "ExitStack | AsyncExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        self._tool.register(stack, context, middleware=middleware)
