# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterable
from contextlib import AsyncExitStack, ExitStack
from typing import TYPE_CHECKING

from autogen.beta.annotations import Context
from autogen.beta.middleware import BaseMiddleware, ToolMiddleware
from autogen.beta.tools.final import tool
from autogen.beta.tools.final.function_tool import FunctionTool
from autogen.beta.tools.sandbox import CodeAdapter
from autogen.beta.tools.tool import Tool

from .environment import CodeEnvironment, CodeLanguage

if TYPE_CHECKING:
    from autogen.beta.annotations import Variable


class SandboxCodeTool(Tool):
    """Exposes a single ``run_code(code, language)`` function backed by a
    :class:`CodeEnvironment` — Daytona, Docker, or any other implementation
    of the protocol.

    Unlike :class:`CodeExecutionTool` (which delegates execution to the LLM
    provider's built-in sandbox), ``SandboxCodeTool`` runs client-side, so
    it works on every provider regardless of native code-execution support.

    There is no default backend: ``environment`` (or ``adapter``) is required.
    The class name is a contract — it executes whatever the model writes,
    so it should only be wired to a backend that genuinely sandboxes
    execution. Convenience class-methods build the adapter for you:
    :meth:`SandboxCodeTool.docker`, :meth:`SandboxCodeTool.daytona`.

    Examples::

        from autogen.beta.extensions.daytona import DaytonaCodeEnvironment
        from autogen.beta.extensions.docker import DockerCodeEnvironment
        from autogen.beta.tools.sandbox import CodeAdapter, LocalSandbox

        # Preferred: adapter over any backend
        code = SandboxCodeTool(adapter=CodeAdapter(LocalSandbox(), languages=("python",)))

        # Convenience shortcuts
        code = SandboxCodeTool.docker(image="python:3.12-slim")

        # Legacy: wrap an environment directly
        code = SandboxCodeTool(DockerCodeEnvironment(image="python:3.12-slim"))
    """

    def __init__(
        self,
        environment: "CodeEnvironment | CodeAdapter | None" = None,
        name: str = "run_code",
        *,
        description: str = "Execute code in a sandboxed environment. Supported languages: {languages}.",
        middleware: Iterable[ToolMiddleware] = (),
        adapter: CodeAdapter | None = None,
    ) -> None:
        if adapter is not None and environment is not None:
            raise ValueError("Pass either `environment` or `adapter`, not both.")
        env: CodeEnvironment | CodeAdapter
        if adapter is not None:
            env = adapter
        elif environment is None:
            raise TypeError("SandboxCodeTool requires an `environment` or `adapter`.")
        else:
            env = environment

        async def run_code(code: str, language: CodeLanguage, ctx: Context) -> str:
            result = await env.run(code, language, context=ctx)
            if result.exit_code != 0:
                suffix = f"[exit code: {result.exit_code}]"
                return f"{result.output}\n{suffix}" if result.output else suffix
            return result.output

        self._env = env
        self._tool: FunctionTool = tool(
            run_code,
            name=name,
            description=description.format(languages=", ".join(env.supported_languages)),
            middleware=middleware,
        )
        self.name = name

    @property
    def environment(self) -> "CodeEnvironment | CodeAdapter":
        """The underlying execution environment."""
        return self._env

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
        languages: tuple[CodeLanguage, ...] = ("python", "bash"),
        name: str = "run_code",
        description: str = "Execute code in a sandboxed environment. Supported languages: {languages}.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> "SandboxCodeTool":
        """Shorthand for ``SandboxCodeTool(adapter=CodeAdapter(DockerSandboxFactory(...)))``."""
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
        adapter = CodeAdapter(factory, languages=languages, timeout=timeout)
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
        languages: tuple[CodeLanguage, ...] = ("python", "bash", "javascript", "typescript"),
        workdir: str = "/workspace",
        name: str = "run_code",
        description: str = "Execute code in a sandboxed environment. Supported languages: {languages}.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> "SandboxCodeTool":
        """Shorthand for ``SandboxCodeTool(adapter=CodeAdapter(DaytonaSandboxFactory(...)))``."""
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
        adapter = CodeAdapter(factory, languages=languages, timeout=timeout)
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
