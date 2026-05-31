# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import AsyncExitStack, ExitStack
from pathlib import Path, PurePosixPath

from autogen.beta.annotations import Context
from autogen.beta.middleware import BaseMiddleware, ToolMiddleware
from autogen.beta.tools.final import tool
from autogen.beta.tools.final.function_tool import FunctionTool
from autogen.beta.tools.sandbox import LocalEnvironment, Sandbox, SandboxFactory, ShellAdapter
from autogen.beta.tools.tool import Tool


class SandboxShellTool(Tool):
    """Exposes a single ``run_shell_command(command)`` function that runs
    shell commands inside an *environment* you choose — local subprocess,
    Docker container, Daytona sandbox, or any custom backend.

    The **environment** decides *where* commands run and carries all
    backend configuration (image, env vars, network, timeout, …). The
    **tool** decides the agent-facing policy (``allowed`` / ``blocked`` /
    ``ignore`` / ``readonly``). The two are orthogonal: the same
    environment can back both a :class:`SandboxShellTool` and a
    :class:`~autogen.beta.tools.SandboxCodeTool`.

    Not to be confused with :class:`~autogen.beta.tools.ShellTool`, which is
    a provider-executed (server-side) shell capability flag.
    ``SandboxShellTool`` runs commands client-side, so it works with any
    provider.

    Examples::

        from autogen.beta.tools import SandboxShellTool, LocalEnvironment
        from autogen.beta.extensions.docker import DockerEnvironment

        # Local subprocess (default):
        shell = SandboxShellTool()
        shell = SandboxShellTool(LocalEnvironment("/tmp/proj"), allowed=["git", "ls"])

        # Docker — configure the backend once, hand it over:
        docker = DockerEnvironment(image="python:3.12-slim", network_mode="none")
        shell = SandboxShellTool(docker, readonly=True)

        # Advanced: pass a pre-built ShellAdapter (custom backend / filters):
        from autogen.beta.tools.sandbox import ShellAdapter, CallableFactory

        shell = SandboxShellTool(ShellAdapter(CallableFactory(make_sandbox), allowed=["node"]))

    Args:
        environment: The execution backend — a
                     :class:`~autogen.beta.tools.sandbox.SandboxFactory`
                     (``LocalEnvironment`` / ``DockerEnvironment`` /
                     ``DaytonaEnvironment``), a bare
                     :class:`~autogen.beta.tools.sandbox.Sandbox`, or a
                     pre-built :class:`ShellAdapter`. ``None`` defaults to a
                     local subprocess (``LocalEnvironment()``).
        allowed: Whitelist of command prefixes. When ``readonly`` is set and
                 ``allowed`` is ``None``, a read-only command set is used.
        blocked: Blacklist of command prefixes.
        ignore: Glob patterns of paths that may not appear in a command.
        readonly: Restrict to read-only commands (cat/ls/grep/…).
        name / description / middleware: Tool wiring.
    """

    def __init__(
        self,
        environment: "SandboxFactory | Sandbox | ShellAdapter | None" = None,
        *,
        allowed: list[str] | None = None,
        blocked: list[str] | None = None,
        ignore: list[str] | None = None,
        readonly: bool = False,
        name: str = "run_shell_command",
        description: str = "Execute a shell command in the working directory: {workdir}",
        middleware: Iterable["ToolMiddleware"] = (),
    ) -> None:
        if isinstance(environment, ShellAdapter):
            if allowed is not None or blocked is not None or ignore is not None or readonly:
                raise ValueError(
                    "Filter arguments (allowed/blocked/ignore/readonly) are not allowed when "
                    "passing a pre-built ShellAdapter — configure them on the adapter instead."
                )
            adapter = environment
        else:
            backend: Sandbox | SandboxFactory = environment if environment is not None else LocalEnvironment()
            adapter = ShellAdapter(
                backend,
                allowed=allowed,
                blocked=blocked,
                ignore=ignore,
                readonly=readonly,
            )

        def run_shell_command(command: str, ctx: Context) -> str:
            return adapter.run_sync(command, context=ctx)

        self._adapter = adapter
        self._workdir = adapter.workdir
        self._tool: FunctionTool = tool(
            run_shell_command,
            name=name,
            description=description.format(workdir=adapter.workdir),
            middleware=middleware,
        )
        self.name = name

    @property
    def workdir(self) -> "Path | PurePosixPath":
        """The working directory of the underlying environment."""
        return self._workdir

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
