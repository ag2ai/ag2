# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os

from ag2.tools.code.environment.base import CodeLanguage

from .factory import SingletonFactory
from .local import LocalSandbox


class LocalEnvironment(SingletonFactory):
    """Local-subprocess backend — the default environment for
    :class:`~ag2.tools.SandboxShellTool`.

    A :class:`~ag2.tools.sandbox.SandboxFactory` over a single
    :class:`LocalSandbox` with a fixed working directory. Hand it to a tool
    the same way you would a :class:`DockerEnvironment` /
    :class:`DaytonaEnvironment`::

        shell = SandboxShellTool(LocalEnvironment("/tmp/proj"), allowed=["git"])

    Commands run via ``subprocess`` on the host, so there is no real
    isolation — that is why :class:`SandboxShellTool` (which filters
    commands) defaults to it, but :class:`SandboxCodeTool` (which runs
    arbitrary model-written code) does not and requires an explicit backend.

    Args:
        path: Working directory. ``None`` creates a temporary directory.
        cleanup: Delete ``path`` on close. Defaults to ``True`` for an
                 auto temp dir, ``False`` for an explicit path.
        timeout: Default per-command timeout in seconds.
        max_output: Maximum characters in a single command's output.
        env_vars: Environment variables merged into every command.
        languages: Informational language list for the sandbox.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        cleanup: bool | None = None,
        timeout: float = 60,
        max_output: int = 100_000,
        env_vars: dict[str, str] | None = None,
        languages: tuple[CodeLanguage, ...] = ("python", "bash"),
    ) -> None:
        super().__init__(
            LocalSandbox(
                path,
                cleanup=cleanup,
                timeout=timeout,
                max_output=max_output,
                env_vars=env_vars,
                languages=languages,
            )
        )

    async def aclose(self) -> None:
        """Close the underlying :class:`LocalSandbox` (deletes its workdir
        when ``cleanup`` was set). Safe to call multiple times."""
        await self.sandbox.aclose()


__all__ = ("LocalEnvironment",)
