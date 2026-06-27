# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import AsyncExitStack, ExitStack
from typing import TYPE_CHECKING

from ag2.annotations import Context
from ag2.middleware import BaseMiddleware, ToolMiddleware
from ag2.tools.final import tool
from ag2.tools.final.function_tool import FunctionTool
from ag2.tools.sandbox import CodeAdapter, SandboxFactory
from ag2.tools.tool import Tool

from .environment import CodeEnvironment, CodeLanguage

if TYPE_CHECKING:
    from ag2.tools.sandbox import LanguageRunner


class SandboxCodeTool(Tool):
    """Exposes a single ``run_code(code, language)`` function that runs
    code inside an *environment* you choose — Docker, Daytona, or any
    custom backend.

    The **environment** decides *where* code runs and carries all backend
    configuration; the **tool** decides which ``languages`` are accepted
    and how each maps to a runner. The same environment can back both a
    :class:`SandboxCodeTool` and a
    :class:`~ag2.tools.SandboxShellTool`.

    Unlike :class:`~ag2.tools.CodeExecutionTool` (which delegates
    execution to the LLM provider's built-in sandbox), ``SandboxCodeTool``
    runs client-side, so it works on every provider regardless of native
    code-execution support.

    There is **no default backend**: ``environment`` is required. The class
    name is a contract — it executes whatever the model writes, so it must
    be wired to a backend that genuinely sandboxes execution. A
    :class:`~ag2.tools.LocalEnvironment` is accepted but only when
    passed explicitly (it offers no isolation).

    Examples::

        from ag2.tools import SandboxCodeTool
        from ag2.extensions.docker import DockerEnvironment

        docker = DockerEnvironment(image="python:3.12-slim")
        code = SandboxCodeTool(docker, languages=("python", "bash"))

        # Advanced: pass a pre-built CodeAdapter (custom runners):
        from ag2.tools.sandbox import CodeAdapter, LocalEnvironment, LanguageRunner

        code = SandboxCodeTool(
            CodeAdapter(
                LocalEnvironment(),
                languages=("typescript",),
                runners={"typescript": LanguageRunner(file_extension="ts", file_runner_argv=("tsx",))},
            )
        )

    Args:
        environment: What runs the code. Either a **backend** — a
                     :class:`~ag2.tools.sandbox.SandboxFactory`
                     (``DockerEnvironment`` / ``DaytonaEnvironment`` /
                     ``LocalEnvironment``), wrapped in a :class:`CodeAdapter`
                     using ``languages`` / ``runners`` — or a ready
                     :class:`~ag2.tools.code.CodeEnvironment` (incl.
                     a pre-built :class:`CodeAdapter`), used as-is. Required.
        languages: Languages this tool accepts (backend form only).
        runners: Override / extend the default language→runner mapping
                 (backend form only).
        name / description / middleware: Tool wiring.
    """

    def __init__(
        self,
        environment: "SandboxFactory | CodeEnvironment",
        *,
        languages: tuple[CodeLanguage, ...] = ("python", "bash"),
        runners: "dict[CodeLanguage, LanguageRunner] | None" = None,
        name: str = "run_code",
        description: str = "Execute code in a sandboxed environment. Supported languages: {languages}.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        env: CodeEnvironment
        if isinstance(environment, SandboxFactory):
            env = CodeAdapter(environment, languages=languages, runners=runners)
        else:
            # Already a CodeEnvironment (incl. CodeAdapter) — use as-is.
            if runners is not None:
                raise ValueError(
                    "`runners` only applies when `environment` is a backend "
                    "(SandboxFactory); configure runners on your "
                    "CodeAdapter / CodeEnvironment instead."
                )
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
    def environment(self) -> "CodeEnvironment":
        """The underlying code environment (a :class:`CodeAdapter` when a
        backend was passed, otherwise the object you supplied)."""
        return self._env

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
