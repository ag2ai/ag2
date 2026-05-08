# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack

from autogen.beta.annotations import Context
from autogen.beta.middleware import BaseMiddleware, ToolMiddleware
from autogen.beta.tools.final import tool
from autogen.beta.tools.final.function_tool import FunctionTool
from autogen.beta.tools.tool import Tool

from .environment import CodeEnvironment, CodeLanguage


class SandboxCodeTool(Tool):
    """Exposes a single ``run_code(code, language)`` function backed by a
    :class:`CodeEnvironment` — Daytona, Docker, or any other implementation
    of the protocol.

    Unlike :class:`CodeExecutionTool` (which delegates execution to the LLM
    provider's built-in sandbox), ``SandboxCodeTool`` runs client-side, so
    it works on every provider regardless of native code-execution support.

    There is no default backend: ``environment`` is required. The class
    name is a contract — it executes whatever the model writes, so it
    should only be wired to a backend that genuinely sandboxes execution.
    Use :class:`~autogen.beta.extensions.daytona.DaytonaCodeEnvironment`
    or :class:`~autogen.beta.extensions.docker.DockerCodeEnvironment` (or
    your own implementation of :class:`CodeEnvironment`).

    Args:
        environment: The execution backend.
        name: Tool name shown to the model. Defaults to ``"run_code"``.
        description: Tool description shown to the model. ``{languages}`` is
                     substituted with the environment's supported languages.
        middleware: Tool middleware applied around each invocation.

    Examples::

        from autogen.beta.extensions.daytona import DaytonaCodeEnvironment
        from autogen.beta.extensions.docker import DockerCodeEnvironment

        # Hosted sandbox
        code = SandboxCodeTool(DaytonaCodeEnvironment(image="python:3.12"))

        # Local container
        code = SandboxCodeTool(DockerCodeEnvironment(image="python:3.12-slim"))
    """

    def __init__(
        self,
        environment: CodeEnvironment,
        name: str = "run_code",
        *,
        description: str = "Execute code in a sandboxed environment. Supported languages: {languages}.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        async def run_code(code: str, language: CodeLanguage, ctx: Context) -> str:
            result = await environment.run(code, language, context=ctx)
            if result.exit_code != 0:
                suffix = f"[exit code: {result.exit_code}]"
                return f"{result.output}\n{suffix}" if result.output else suffix
            return result.output

        self._env = environment
        self._tool: FunctionTool = tool(
            run_code,
            name=name,
            description=description.format(languages=", ".join(environment.supported_languages)),
            middleware=middleware,
        )
        self.name = name

    @property
    def environment(self) -> CodeEnvironment:
        """The underlying execution environment."""
        return self._env

    async def schemas(self, context: "Context") -> list:
        return await self._tool.schemas(context)

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        self._tool.register(stack, context, middleware=middleware)
