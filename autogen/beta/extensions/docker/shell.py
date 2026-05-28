# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import TYPE_CHECKING

from autogen.beta.annotations import Variable
from autogen.beta.tools.code import CodeLanguage
from autogen.beta.tools.sandbox import ShellAdapter
from autogen.beta.tools.shell.environment.base import ShellEnvironment

from .factory import DockerSandboxFactory

if TYPE_CHECKING:
    from autogen.beta.context import ConversationContext


class DockerShellEnvironment(ShellEnvironment):
    """Deprecated alias — use ``ShellAdapter(DockerSandboxFactory(...))``."""

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
        languages: tuple[CodeLanguage, ...] = ("python", "bash"),
        host_path: str | os.PathLike[str] | None = None,
        workdir: str = "/workspace",
        allowed: list[str] | None = None,
        blocked: list[str] | None = None,
        ignore: list[str] | None = None,
        readonly: bool = False,
        env: dict[str, str] | None = None,
    ) -> None:
        del languages  # accepted for v1 compat; the language matrix lives on CodeAdapter.
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
        self._workdir = Path(workdir)
        self._adapter = ShellAdapter(
            factory,
            allowed=allowed,
            blocked=blocked,
            ignore=ignore,
            readonly=readonly,
            env=env,
            timeout=timeout,
        )

    @property
    def workdir(self) -> Path:
        return self._workdir

    def run(self, command: str, *, context: "ConversationContext | None" = None) -> str:
        return self._adapter.run_sync(command, context=context)

    async def aclose(self) -> None:
        """No-op: factory opens / closes a container per :meth:`run`."""
