# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from os import PathLike
from pathlib import Path

from autogen.beta.annotations import Context
from autogen.beta.middleware import BaseMiddleware, ToolMiddleware
from autogen.beta.tools.final import tool
from autogen.beta.tools.final.function_tool import FunctionTool
from autogen.beta.tools.tool import Tool

from .environment import LocalShellEnvironment, ShellEnvironment


class LocalShellTool(Tool):
    """The tool exposes a single ``shell`` function that runs commands in whatever
    environment is provided — local subprocess, Docker container, SSH, etc.
    All execution details are encapsulated inside the environment.

    Args:
        environment: The execution environment, a path to a working directory,
                     or ``None``. When a path (``str`` or :class:`~pathlib.Path`)
                     is given, a :class:`LocalShellEnvironment` is created in
                     that directory automatically. Defaults to
                     :class:`LocalShellEnvironment` with a temporary directory.

    Examples::

        # Auto temp dir — cleaned up on exit
        sh = LocalShellTool()

        # Pass a path directly — creates LocalShellEnvironment for you
        sh = LocalShellTool("/tmp/my_project")
        sh = LocalShellTool(Path("/tmp/my_project"))

        # Full control via explicit environment
        sh = LocalShellTool(LocalShellEnvironment(path="/tmp/my_project"))

        # Read-only local inspection
        sh = LocalShellTool(LocalShellEnvironment(path="/tmp/my_project", readonly=True))

        # Future: Docker or SSH (not yet implemented)
        # sh = LocalShellTool(DockerEnvironment(image="python:3.12"))
        # sh = LocalShellTool(SSHEnvironment(host="server.com", user="ubuntu"))
    """

    def __init__(
        self,
        environment: ShellEnvironment | str | PathLike[str] | None = None,
        name: str = "run_shell_command",
        *,
        description: str = "Execute a shell command in the working directory: {workdir}",
        middleware: Iterable["ToolMiddleware"] = (),
    ) -> None:
        if environment:
            env: ShellEnvironment = LocalShellEnvironment.ensure_env(environment)
        else:
            env = LocalShellEnvironment()

        self._tool: FunctionTool = tool(
            env.run,
            name=name,
            description=description.format(workdir=env.workdir),
            middleware=middleware,
        )

        self._workdir = env.workdir
        self.name = name

    @property
    def workdir(self) -> Path:
        """The working directory of the underlying environment."""
        return self._workdir

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
